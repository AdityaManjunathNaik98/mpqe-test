#------------------------------
# IMPORTS AND CONFIGURATION
#------------------------------
import os
import os.path as osp
import pickle as pkl
import numpy as np
import yaml
import torch 
from torch import optim

import utils as utils
from utils.data_utils import load_queries_by_formula, load_test_queries_by_formula, load_graph
from rgcn import RGCNEncoderDecoder
from sacred import Experiment
from sacred.observers import MongoObserver
from utils.utils import check_conv, update_loss

class RGCNTrainingData:
    def __init__(self, train_queries=None, val_queries=None, test_queries=None, 
                 batch_size=512, current_iteration=0, past_burn_in=False):
        self.train_queries = train_queries
        self.val_queries = val_queries
        self.test_queries = test_queries
        self.batch_size = batch_size
        self.current_iteration = current_iteration
        self.past_burn_in = past_burn_in

#------------------------------
# CONFIG LOADING (NO ARGUMENTS)
#------------------------------
CONFIG_FILE = "config.yaml"  # Fixed config file name

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def validate_config(config):
    """Validate configuration and set defaults."""
    # Required sections
    required_sections = ['model', 'training', 'rgcn', 'data', 'hardware', 'logging']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # Set device based on hardware config
    if config['hardware']['device'] == 'auto':
        config['hardware']['device'] = 'cuda' if (
            config['hardware']['use_cuda'] and torch.cuda.is_available()
        ) else 'cpu'
    
    # Validate paths
    if not os.path.exists(config['data']['data_dir']):
        raise ValueError(f"Data directory does not exist: {config['data']['data_dir']}")
    
    return config

# Load and process configuration
print(f"Loading configuration from {CONFIG_FILE}")
config = load_config(CONFIG_FILE)
config = validate_config(config)

print("Configuration loaded:")
print(yaml.dump(config, default_flow_style=False, indent=2))

#------------------------------
# DATA LOADING   taking from multiple files for now
#------------------------------
print("Loading graph data...")
graph, feature_modules, node_maps = load_graph(
    config['data']['data_dir'], 
    config['model']['embed_dim']
)

if config['hardware']['use_cuda']:
    graph.features = utils.cudify(feature_modules, node_maps)
    for key in node_maps:
        node_maps[key] = node_maps[key].cuda()

out_dims = {mode: config['model']['embed_dim'] for mode in graph.relations}

# Get encoder AFTER moving node_maps to GPU
enc = utils.get_encoder(
    config['data']['depth'], 
    graph, 
    out_dims, 
    feature_modules, 
    config['hardware']['use_cuda']
)

print("Loading queries...")
data_dir = config['data']['data_dir']
train_queries = load_queries_by_formula(data_dir + "/train_edges.pkl")
val_queries = load_test_queries_by_formula(data_dir + "/val_edges.pkl")
test_queries = load_test_queries_by_formula(data_dir + "/test_edges.pkl")

# Load additional query files
for i in range(2, 4):
    train_queries.update(load_queries_by_formula(data_dir + "/train_queries_{:d}.pkl".format(i)))
    i_val_queries = load_test_queries_by_formula(data_dir + "/val_queries_{:d}.pkl".format(i))
    val_queries["one_neg"].update(i_val_queries["one_neg"])
    val_queries["full_neg"].update(i_val_queries["full_neg"])
    i_test_queries = load_test_queries_by_formula(data_dir + "/test_queries_{:d}.pkl".format(i))
    test_queries["one_neg"].update(i_test_queries["one_neg"])
    test_queries["full_neg"].update(i_test_queries["full_neg"])

print("Data loaded successfully.")

#------------------------------
# MODEL SETUP
#------------------------------
# Flatten the nested config structure
flat_config = {
    **config['model'],      # num_layers, adaptive, etc.
    **config['training'],   # learning_rate, etc.
    **config['rgcn'],       # readout, scatter_op
    **config['hardware'],   # use_cuda, device
    **config['logging'],    # log_every, val_every
    **config['data'],       # data_dir, depth
    'graph': graph,
    'enc': enc
}

# Initialize model with flattened config
enc_dec = RGCNEncoderDecoder(flat_config)

# Create optimizer (use model's built-in optimizer)
enc_dec._init_optimizer_and_criterion()
optimizer = enc_dec.optimizer

# Setup logging
logger = utils.setup_logging(config['logging']['log_dir'] + "/training.log")

# Print model information
print("\nModel Information:")
model_info = enc_dec.get_model_info()
for key, value in model_info.items():
    print(f"  {key}: {value}")

print(f"\nTotal trainable parameters: {enc_dec.get_num_parameters():,}")

#------------------------------
# SACRED EXPERIMENT SETUP
#------------------------------
ex = Experiment(config.get('experiment', {}).get('name', 'rgcn_experiment'))

# Add MongoDB observer if configured
uri = os.environ.get('MLAB_URI')
database = os.environ.get('MLAB_DB')
if all([uri, database]):
    ex.observers.append(MongoObserver.create(uri, database))

@ex.config
def sacred_config():
    # Sacred configuration from YAML
    model_name = config.get('experiment', {}).get('name', 'qrgcn')
    learning_rate = config['training']['learning_rate']
    num_layers = config['model']['num_layers']
    max_burn_in = config['training']['max_burn_in']
    max_iter = config['training']['max_iter']
    batch_size = config['training']['batch_size']
    embed_dim = config['model']['embed_dim']
    readout = config['rgcn']['readout']
    
    # Add all config as sacred config
    yaml_config = config

@ex.main
def main(_run):
    print("Starting training...")
    
    # Log configuration to Sacred
    if _run is not None:
        for section, values in config.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    if isinstance(value, (int, float, str, bool)):
                        _run.log_scalar(f'config.{section}.{key}', value)
    
    # Setup output directory
    exp_id = '-' + str(_run._id) if _run._id is not None else ''
    db_name = database if database is not None else ''
    folder_path = osp.join(config['logging']['log_dir'], db_name, 'output' + exp_id)
    if not osp.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    
    model_path = osp.join(folder_path, "model.pt")
    config_save_path = osp.join(folder_path, "config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    # Generic training loop using train_step/eval_step
    print('Training RGCN-Enc-Dec (Generic)')
    
    edge_conv = False
    ema_loss = None
    vals = []
    losses = []
    conv_test = None
    
    max_iter = flat_config['max_iter']
    max_burn_in = flat_config['max_burn_in']
    log_every = flat_config['log_every']
    val_every = flat_config['val_every']
    batch_size = flat_config['batch_size']
    tolerance = flat_config.get('tolerance', flat_config.get('tol', 1e-6))
    
    for i in range(max_iter):
        # Determine if past burn-in
        past_burn_in = edge_conv or len(losses) >= max_burn_in
        
        # Create training data package
        train_data = RGCNTrainingData(
            train_queries=train_queries,
            batch_size=batch_size,
            current_iteration=i,
            past_burn_in=past_burn_in
        )
        
        # Single training step
        loss = enc_dec.train_step(train_data)
        
        # Check for edge convergence
        if not edge_conv and (check_conv(vals) or len(losses) >= max_burn_in):
            logger.info("Edge converged at iteration {:d}".format(i-1))
            logger.info("Testing at edge conv...")
            
            test_data = RGCNTrainingData(
                test_queries=test_queries,
                batch_size=128,
                current_iteration=i
            )
            # test_result = enc_dec.eval_step(test_data)
            # conv_test = test_result['accuracy']
            print(f"Final test_queries structure: {list(test_queries.keys())}")
            final_result = enc_dec.eval_step(test_data)
            test_avg_auc = final_result['accuracy']
            
            
            edge_conv = True
            losses = []
            ema_loss = None
            vals = []
            
            if model_path is not None:
                torch.save(enc_dec.state_dict(), model_path+"-edge_conv")
        
        # Update loss tracking
        losses, ema_loss = update_loss(loss, losses, ema_loss)
        
        # Check final convergence
        if edge_conv and check_conv(vals):
            logger.info("Fully converged at iteration {:d}".format(i))
            break
        
        # Logging
        if i % log_every == 0:
            logger.info("Iter: {:d}; ema_loss: {:f}".format(i, ema_loss))
            if _run is not None:
                _run.log_scalar('ema_loss', ema_loss, i)
        
        # Validation
        if i >= val_every and i % val_every == 0:
            val_data = RGCNTrainingData(
                val_queries=val_queries,
                batch_size=128,
                current_iteration=i
            )
            val_result = enc_dec.eval_step(val_data)
            
            if edge_conv:
                vals.append(val_result['accuracy'])
            else:
                vals.append(val_result['accuracy'])
    
    # Final evaluation
    test_data = RGCNTrainingData(
        val_queries=test_queries,
        batch_size=128,
        current_iteration=i
    )
    print(f"Final test_queries structure: {list(test_queries.keys())}")
    final_result = enc_dec.eval_step(test_data)
    test_avg_auc = final_result['accuracy']
    
    logger.info("Test macro-averaged val: {:f}".format(test_avg_auc))
    if _run is not None:
        _run.log_scalar('test_auc', test_avg_auc)
    
    # Save final results
    torch.save(enc_dec.state_dict(), model_path)
    
    vocab_path = osp.join(folder_path, 'training_vocabulary.pkl')
    with open(vocab_path, 'wb') as f:
        pkl.dump(enc_dec.graph.full_sets, f)
    
    model_info_path = osp.join(folder_path, 'model_info.yaml')
    with open(model_info_path, 'w') as f:
        yaml.dump(enc_dec.get_model_info(), f, default_flow_style=False, indent=2)
    
    # Log final results
    final_params = enc_dec.get_num_parameters()
    print(f"Training completed. Final test AUC: {test_avg_auc:.4f}")
    print(f"Results saved to: {folder_path}")
    
    # Log to Sacred
    if _run is not None:
        _run.log_scalar('final_test_auc', test_avg_auc)
        _run.log_scalar('model_parameters', final_params)
        _run.add_artifact(config_save_path, 'config.yaml')
        _run.add_artifact(model_info_path, 'model_info.yaml')
    
    return test_avg_auc

#------------------------------
# UTILITY FUNCTIONS
#------------------------------
def load_pretrained_model(model_path, config_path=CONFIG_FILE):
    """Load a pretrained model using YAML configuration."""
    config = load_config(config_path)
    config = validate_config(config)
    
    # Recreate data loading
    graph, feature_modules, node_maps = load_graph(
        config['data']['data_dir'], 
        config['model']['embed_dim']
    )
    
    if config['hardware']['use_cuda']:
        graph.features = utils.cudify(feature_modules, node_maps)
        for key in node_maps:
            node_maps[key] = node_maps[key].cuda()
    
    out_dims = {mode: config['model']['embed_dim'] for mode in graph.relations}
    enc = utils.get_encoder(
        config['data']['depth'], 
        graph, 
        out_dims, 
        feature_modules, 
        config['hardware']['use_cuda']
    )
    
    # Flatten config and create model
    flat_config = {
        **config['model'],
        **config['training'],
        **config['rgcn'],
        **config['hardware'],
        **config['logging'],
        **config['data'],
        'graph': graph,
        'enc': enc
    }
    
    model = RGCNEncoderDecoder(flat_config)
    model.load_model(model_path)
    return model

if __name__ == "__main__":
    ex.run()