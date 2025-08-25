#------------------------------
# IMPORTS AND CONFIGURATION
#------------------------------
import os
import os.path as osp
import pickle as pkl
import numpy as np
from argparse import ArgumentParser
import torch 
from torch import optim

import utils as utils
from data_utils import load_queries_by_formula, load_test_queries_by_formula, load_graph
from model import RGCNEncoderDecoder, QueryEncoderDecoder
from train_helpers import train_ingredient, run_train
from sacred import Experiment
from sacred.observers import MongoObserver

# Configuration - only essential arguments
parser = ArgumentParser()
parser.add_argument("--model", type=str, default="qrgcn", help="Model type: qrgcn or gqe")
parser.add_argument("--max_iter", type=int, default=10000000, help="Maximum training iterations")
parser.add_argument("--max_burn_in", type=int, default=1000000, help="Maximum burn-in iterations")
parser.add_argument("--num_layers", type=int, default=2, help="Number of model layers")
args = parser.parse_args()

# Fixed hyperparameters
EMBED_DIM = 128
DATA_DIR = "F:/cuda-environment/AIFB/processed"
LEARNING_RATE = 0.01
BATCH_SIZE = 512
VAL_EVERY = 5000
TOLERANCE = 0.0001
USE_CUDA = True
LOG_DIR = "./"
MODEL_DIR = "./"
DECODER = "bilinear"
READOUT = "sum"
INTER_DECODER = "mean"
OPTIMIZER = "adam"
DROPOUT = 0.0
WEIGHT_DECAY = 0.0
NUM_BASES = 0
SCATTER_OP = 'add'
PATH_WEIGHT = 0.01
DEPTH = 0
SHARED_LAYERS = False
ADAPTIVE = False

#------------------------------
# DATA INGESTION
#------------------------------
print("Loading graph data...")
graph, feature_modules, node_maps = load_graph(DATA_DIR, EMBED_DIM)
if USE_CUDA:
    graph.features = utils.cudify(feature_modules, node_maps)
out_dims = {mode: EMBED_DIM for mode in graph.relations}

print("Loading edge data...")
train_queries = load_queries_by_formula(DATA_DIR + "/train_edges.pkl")
val_queries = load_test_queries_by_formula(DATA_DIR + "/val_edges.pkl")
test_queries = load_test_queries_by_formula(DATA_DIR + "/test_edges.pkl")

#------------------------------
# DATA PREPROCESSING
#------------------------------
print("Loading and merging query data...")
for i in range(2, 4):
    # Load training queries
    train_queries.update(load_queries_by_formula(DATA_DIR + "/train_queries_{:d}.pkl".format(i)))
    
    # Load and merge validation queries
    i_val_queries = load_test_queries_by_formula(DATA_DIR + "/val_queries_{:d}.pkl".format(i))
    val_queries["one_neg"].update(i_val_queries["one_neg"])
    val_queries["full_neg"].update(i_val_queries["full_neg"])
    
    # Load and merge test queries
    i_test_queries = load_test_queries_by_formula(DATA_DIR + "/test_queries_{:d}.pkl".format(i))
    test_queries["one_neg"].update(i_test_queries["one_neg"])
    test_queries["full_neg"].update(i_test_queries["full_neg"])


#------------------------------
# MODEL INSTANTIATION
#------------------------------
print(f"Initializing {args.model} model...")
enc = utils.get_encoder(DEPTH, graph, out_dims, feature_modules, USE_CUDA)

if args.model == 'qrgcn':
    enc_dec = RGCNEncoderDecoder(
        graph, enc, READOUT, SCATTER_OP,
        DROPOUT, WEIGHT_DECAY,
        args.num_layers, SHARED_LAYERS, ADAPTIVE
    )
elif args.model == 'gqe':
    dec = utils.get_metapath_decoder(
        graph,
        enc.out_dims if DEPTH > 0 else out_dims,
        DECODER
    )
    inter_dec = utils.get_intersection_decoder(graph, out_dims, INTER_DECODER)
    enc_dec = QueryEncoderDecoder(graph, enc, dec, inter_dec)
else:
    raise ValueError(f'Unknown model {args.model}')

if USE_CUDA:
    enc_dec.cuda()

# Initialize optimizer
if OPTIMIZER == "sgd":
    optimizer = optim.SGD([p for p in enc_dec.parameters() if p.requires_grad],
                          lr=LEARNING_RATE, momentum=0)
elif OPTIMIZER == "adam":
    optimizer = optim.Adam([p for p in enc_dec.parameters() if p.requires_grad],
                           lr=LEARNING_RATE)

print(f"Model initialized with {sum(p.numel() for p in enc_dec.parameters())} parameters")

#------------------------------
# LOGGING AND EXPERIMENT SETUP
#------------------------------
fname = "{data:s}{depth:d}-{embed_dim:d}-{lr:f}-{model}-{decoder}-{readout}."
log_file = (LOG_DIR + fname + "log").format(
    data=DATA_DIR.strip().split("/")[-1],
    depth=DEPTH,
    embed_dim=EMBED_DIM,
    lr=LEARNING_RATE,
    decoder=DECODER,
    model=args.model,
    readout=READOUT
)

model_file = "model.pt"
logger = utils.setup_logging(log_file)

# Sacred experiment setup
ex = Experiment(ingredients=[train_ingredient])
uri = os.environ.get('MLAB_URI')
database = os.environ.get('MLAB_DB')
if all([uri, database]):
    ex.observers.append(MongoObserver.create(uri, database))
else:
    print('Running without Sacred observers')

@ex.config
def config():
    model = args.model
    lr = LEARNING_RATE
    num_layers = args.num_layers
    shared_layers = SHARED_LAYERS
    adaptive = ADAPTIVE
    readout = READOUT
    dropout = DROPOUT
    weight_decay = WEIGHT_DECAY
    max_burn_in = args.max_burn_in
    num_basis = NUM_BASES
    scatter_op = SCATTER_OP
    opt = OPTIMIZER
    data_dir = DATA_DIR
    path_weight = PATH_WEIGHT
    decoder = DECODER

#------------------------------
# MODEL TRAINING AND VALIDATION
#------------------------------
@ex.main
def main(data_dir, _run):
    print("Starting training...")
    
    # Setup output directory
    exp_id = '-' + str(_run._id) if _run._id is not None else ''
    db_name = database if database is not None else ''
    folder_path = osp.join(LOG_DIR, db_name, 'output' + exp_id)
    if not osp.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    model_path = osp.join(folder_path, model_file)

    # Run training
    run_train(
        enc_dec, optimizer, train_queries, val_queries, test_queries,
        logger, batch_size=BATCH_SIZE, max_burn_in=args.max_burn_in,
        val_every=VAL_EVERY, max_iter=args.max_iter,
        model_file=model_path, path_weight=PATH_WEIGHT
    )

    print("Training completed!")

    # Export embeddings for downstream tasks
    entity_ids_path = osp.join(data_dir, 'entity_ids.pkl')
    if osp.exists(entity_ids_path):
        print("Exporting embeddings...")
        entity_ids = pkl.load(open(entity_ids_path, 'rb'))
        embeddings = np.zeros((len(entity_ids), 1 + EMBED_DIM))

        for i, ent_id in enumerate(entity_ids.values()):
            for mode in enc_dec.graph.full_sets:
                if ent_id in enc_dec.graph.full_sets[mode]:
                    embeddings[i, 0] = ent_id
                    id_tensor = torch.tensor([ent_id])
                    emb = enc_dec.enc(id_tensor, mode).detach().cpu().numpy()
                    embeddings[i, 1:] = emb.reshape(-1)

        file_path = osp.join(folder_path, 'embeddings.npy')
        np.save(file_path, embeddings)
        print(f'Saved embeddings at {file_path}')
    else:
        print('Did not find entity_ids dictionary. Files found:')
        print(os.listdir(data_dir))

#------------------------------
# EXECUTION
#------------------------------
if __name__ == "__main__":
    ex.run()