import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from torch_scatter import scatter_add, scatter_max, scatter_mean
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Parameter as Param
from torch_geometric.nn import inits
from utils.data_utils import *
from utils.utils import *
from typing import Dict, Any

from base import BaseGNN


class MLPReadout(nn.Module):
    def __init__(self, input_dim, output_dim, scatter_fn):
        super(MLPReadout, self).__init__()
        self.layers = nn.Sequential(nn.Linear(in_features=input_dim,
                                              out_features=output_dim),
                                    nn.ReLU(),
                                    nn.Linear(in_features=output_dim,
                                              out_features=output_dim))
        self.scatter_fn = scatter_fn

    def forward(self, embs, batch_idx, **kwargs):
        x = self.layers(embs)
        x = self.scatter_fn(x, batch_idx, dim=0)

        # If scatter_fn is max or min, values and indices are returned
        if isinstance(x, tuple):
            x = x[0]

        return x


class TargetMLPReadout(nn.Module):
    def __init__(self, dim, scatter_fn):
        super(TargetMLPReadout, self).__init__()
        self.layers = nn.Sequential(nn.Linear(in_features=2*dim,
                                              out_features=dim),
                                    nn.ReLU(),
                                    nn.Linear(in_features=dim,
                                              out_features=dim))
        self.scatter_fn = scatter_fn

    def forward(self, embs, batch_idx, batch_size, num_nodes, num_anchors,
                **kwargs):
        device = embs.device

        non_target_idx = torch.ones(num_nodes, dtype=torch.bool)
        non_target_idx[num_anchors] = 0
        non_target_idx.to(device)

        batch_idx = batch_idx.reshape(batch_size, -1)
        batch_idx = batch_idx[:, non_target_idx].reshape(-1)

        embs = embs.reshape(batch_size, num_nodes, -1)
        non_targets = embs[:, non_target_idx]
        targets = embs[:, ~non_target_idx].expand_as(non_targets)

        x = torch.cat((targets, non_targets), dim=-1)
        x = x.reshape(batch_size * (num_nodes - 1), -1).contiguous()

        x = self.layers(x)
        x = self.scatter_fn(x, batch_idx, dim=0)

        # If scatter_fn is max or min, values and indices are returned
        if isinstance(x, tuple):
            x = x[0]

        return x



class RGCNConv(MessagePassing):
    r"""The relational graph convolutional operator from the `"Modeling
    Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_0 \cdot \mathbf{x}_i +
        \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
        \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j,

    where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
    Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
    stores a relation identifier
    :math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}` for each edge.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        num_bases (int): Number of bases used for basis-decomposition.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_relations,
                 num_bases,
                 bias=True):
        super(RGCNConv, self).__init__('add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases

        if num_bases == 0:
            self.basis = Param(torch.Tensor(num_relations, in_channels, out_channels))
            self.att = None
        else:
            self.basis = Param(torch.Tensor(num_bases, in_channels, out_channels))
            self.att = Param(torch.Tensor(num_relations, num_bases))
        self.root = Param(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Param(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.att is None:
            size = self.num_relations * self.in_channels
        else:
            size = self.num_bases * self.in_channels
            inits.uniform(size, self.att)

        inits.uniform(size, self.basis)
        inits.uniform(size, self.root)
        inits.uniform(size, self.bias)

    def forward(self, x, edge_index, edge_type, edge_norm=None):
        """"""
        if x is None:
            x = torch.arange(
                edge_index.max().item() + 1,
                dtype=torch.long,
                device=edge_index.device)

        return self.propagate(
            edge_index, x=x, edge_type=edge_type, edge_norm=edge_norm)

    def message(self, x_j, edge_type, edge_norm):
        if self.att is None:
            w = self.basis.view(self.num_relations, -1)
        else:
            w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))

        if x_j.dtype == torch.long:
            w = w.view(-1, self.out_channels)
            index = edge_type * self.in_channels + x_j
            out = torch.index_select(w, 0, index)
            return out if edge_norm is None else out * edge_norm.view(-1, 1)
        else:
            w = w.view(self.num_relations, self.in_channels, self.out_channels)
            w = torch.index_select(w, 0, edge_type)
            out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)
            return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, x):
        if x.dtype == torch.long:
            out = aggr_out + self.root
        else:
            out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)


class RGCNEncoderDecoder(BaseGNN):
    """
    RGCN Encoder-Decoder that inherits from BaseGNN with YAML configuration support.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize RGCN Encoder-Decoder.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        # Store RGCN-specific components temporarily
        graph = config['graph']
        enc = config['enc']
        
        # Set embedding dimension from graph
        emb_dim = graph.feature_dims[next(iter(graph.feature_dims))]
        
        # Set BaseGNN required parameters
        config['input_dim'] = emb_dim
        config['hidden_dim'] = emb_dim  
        config['output_dim'] = emb_dim
        
        # Call parent constructor FIRST
        super(RGCNEncoderDecoder, self).__init__(config)
        
        # NOW assign the modules after nn.Module is initialized
        self.graph = graph
        self.enc = enc
        self.emb_dim = emb_dim
        
        # Extract RGCN-specific parameters using config.get() with defaults
        self.readout_str = config.get('readout', 'mp')
        self.scatter_op = config.get('scatter_op', 'add')
        self.shared_layers = config.get('shared_layers', True)
        self.adaptive = config.get('adaptive', True)
        
        # Training parameters
        self.max_burn_in = config.get('max_burn_in', 100000)
        self.batch_size = config.get('batch_size', 512)
        self.log_every = config.get('log_every', 500)
        self.val_every = config.get('val_every', 1000)
        self.tol = config.get('tol', 1e-6)
        self.inter_weight = config.get('inter_weight', 0.005)
        self.path_weight = config.get('path_weight', 0.01)
        
        # Training state variables
        self.edge_conv = False
        self.ema_loss = None
        self.vals = []
        self.losses = []
        self.conv_test = None
        self.train_iterators = {}
        self.current_iter = 0
        
        # Initialize RGCN components
        self._setup_rgcn_components()
        
        # Initialize optimizer and criterion
        self._init_optimizer_and_criterion()
    
    def _setup_rgcn_components(self):
        """Set up RGCN-specific components."""
        # Initialize mode embeddings
        self.mode_embeddings = nn.Embedding(len(self.graph.mode_weights), self.emb_dim)
        
        # Create mode and relation ID mappings
        self.mode_ids = {}
        mode_id = 0
        for mode in self.graph.mode_weights:
            self.mode_ids[mode] = mode_id
            mode_id += 1

        self.rel_ids = {}
        id_rel = 0
        for r1 in self.graph.relations:
            for r2 in self.graph.relations[r1]:
                rel = (r1, r2[1], r2[0])
                self.rel_ids[rel] = id_rel
                id_rel += 1

        # Initialize RGCN layers
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            if len(self.layers) == 0 or not self.shared_layers:
                rgcn = RGCNConv(in_channels=self.emb_dim,
                                out_channels=self.emb_dim,
                                num_relations=len(self.graph.rel_edges),
                                num_bases=0)
            self.layers.append(rgcn)

        # Set up scatter function
        if self.scatter_op == 'add':
            scatter_fn = scatter_add
        elif self.scatter_op == 'max':
            scatter_fn = scatter_max
        elif self.scatter_op == 'mean':
            scatter_fn = scatter_mean
        else:
            raise ValueError(f'Unknown scatter op {self.scatter_op}')

        # Initialize readout function
        if self.readout_str == 'sum':
            self.readout = self.sum_readout
        elif self.readout_str == 'max':
            self.readout = self.max_readout
        elif self.readout_str == 'mlp':
            self.readout = MLPReadout(self.emb_dim, self.emb_dim, scatter_fn)
        elif self.readout_str == 'targetmlp':
            self.readout = TargetMLPReadout(self.emb_dim, scatter_fn)
        elif self.readout_str == 'concat':
            self.readout = MLPReadout(self.emb_dim * self.num_layers, self.emb_dim, scatter_fn)
        elif self.readout_str == 'mp':
            self.readout = self.target_message_readout
        else:
            raise ValueError(f'Unknown readout function {self.readout_str}')
    
    def _init_optimizer_and_criterion(self):
        """Initialize optimizer and criterion - override for RGCN-specific needs."""
        if self.optimizer is None:
            self.optimizer = self._create_optimizer()
        
        if self.criterion is None:
            # RGCN uses custom loss, but set a default for BaseGNN compatibility
            self.criterion = nn.MSELoss()  # Default, margin_loss is used in practice
    
    def forward(self, x, edge_index, **kwargs):
        """
        Standard forward pass to satisfy BaseGNN interface.
        This is a simplified version for basic GNN operations.
        For full query processing, use forward_query method.
        """
        # Basic RGCN forward pass
        h = x
        for i in range(self.num_layers - 1):
            h = self.layers[i](h, edge_index, kwargs.get('edge_type'))
            h = F.relu(h)
        
        h = self.layers[-1](h, edge_index, kwargs.get('edge_type'))
        return h

    def forward_query(self, formula, queries, target_nodes,
                      anchor_ids=None, var_ids=None, q_graphs=None,
                      neg_nodes=None, neg_lengths=None):
        """
        Original RGCN forward method for query processing.
        """
        if anchor_ids is None or var_ids is None or q_graphs is None:
            from utils.data_utils import RGCNQueryDataset
            query_data = RGCNQueryDataset.get_query_graph(formula, queries,
                                                          self.rel_ids,
                                                          self.mode_ids)
            anchor_ids, var_ids, q_graphs = query_data

        device = next(self.parameters()).device
        var_ids = var_ids.to(device)
        q_graphs = q_graphs.to(device)

        batch_size, num_anchors = anchor_ids.shape
        n_vars = var_ids.shape[0]
        n_nodes = num_anchors + n_vars

        x = torch.empty(batch_size, n_nodes, self.emb_dim).to(var_ids.device)
        for i, anchor_mode in enumerate(formula.anchor_modes):
            x[:, i] = self.enc(anchor_ids[:, i], anchor_mode).t()
        x[:, num_anchors:] = self.mode_embeddings(var_ids)
        x = x.reshape(-1, self.emb_dim)
        q_graphs.x = x

        if self.adaptive:
            from utils.data_utils import RGCNQueryDataset
            num_passes = RGCNQueryDataset.query_diameters[formula.query_type]
            if num_passes > len(self.layers):
                raise ValueError(f'RGCN is adaptive with {len(self.layers)}'
                                 f' layers, but query requires {num_passes}.')
        else:
            num_passes = self.num_layers

        h1 = q_graphs.x
        h_layers = []
        for i in range(num_passes - 1):
            h1 = self.layers[i](h1, q_graphs.edge_index, q_graphs.edge_type)
            h1 = F.relu(h1)
            if self.readout_str == 'concat':
                h_layers.append(h1)

        h1 = self.layers[-1](h1, q_graphs.edge_index, q_graphs.edge_type)

        if self.readout_str == 'concat':
            h_layers.append(h1)
            h1 = torch.cat(h_layers, dim=1)

        out = self.readout(embs=h1, batch_idx=q_graphs.batch,
                           batch_size=batch_size, num_nodes=n_nodes,
                           num_anchors=num_anchors)

        target_embeds = self.enc(target_nodes, formula.target_mode).t()
        scores = F.cosine_similarity(out, target_embeds, dim=1)

        if neg_nodes is not None:
            neg_embeds = self.enc(neg_nodes, formula.target_mode).t()
            out = out.repeat_interleave(torch.tensor(neg_lengths).to(device),
                                        dim=0)
            neg_scores = F.cosine_similarity(out, neg_embeds)
            scores = torch.cat((scores, neg_scores), dim=0)

        return scores

    # Keep all the existing readout methods
    def sum_readout(self, embs, batch_idx, **kwargs):
        return scatter_add(embs, batch_idx, dim=0)

    def max_readout(self, embs, batch_idx, **kwargs):
        out, argmax = scatter_max(embs, batch_idx, dim=0)
        return out

    def target_message_readout(self, embs, batch_size, num_nodes, num_anchors, **kwargs):
        device = embs.device
        non_target_idx = torch.ones(num_nodes, dtype=torch.bool)
        non_target_idx[num_anchors] = 0
        non_target_idx.to(device)

        embs = embs.reshape(batch_size, num_nodes, -1)
        targets = embs[:, ~non_target_idx].reshape(batch_size, -1)
        return targets

    def margin_loss(self, formula, queries, anchor_ids=None, var_ids=None,
                    q_graphs=None, hard_negatives=False, margin=1):
        """RGCN-specific margin loss function."""
        if not "inter" in formula.query_type and hard_negatives:
            raise Exception("Hard negative examples can only be used with "
                            "intersection queries")
        elif hard_negatives:
            neg_nodes = [random.choice(query.hard_neg_samples)
                         for query in queries]
        elif formula.query_type == "1-chain":
            neg_nodes = [random.choice(self.graph.full_lists[formula.target_mode]) for _ in queries]
        else:
            neg_nodes = [random.choice(query.neg_samples) for query in queries]

        affs = self.forward_query(formula, queries,
                                  [query.target_node for query in queries],
                                  anchor_ids, var_ids, q_graphs)
        neg_affs = self.forward_query(formula, queries, neg_nodes,
                                      anchor_ids, var_ids, q_graphs)
        loss = margin - (affs - neg_affs)
        loss = torch.clamp(loss, min=0)
        loss = loss.mean()

        if isinstance(self.readout, nn.Module) and self.weight_decay > 0:
            l2_reg = 0
            for param in self.readout.parameters():
                l2_reg += torch.norm(param)
            loss += self.weight_decay * l2_reg

        return loss

    def train_step(self, data, mask=None):
        """
        Single training step for RGCN query reasoning.
        
        Args:
            data: Should contain train_queries, current_iteration, and other training state
            mask: Not used for RGCN, kept for interface compatibility
        
        Returns:
            Average loss for this step
        """
        self.train()
        
        # Extract training data
        train_queries = data.train_queries
        batch_size = getattr(data, 'batch_size', self.batch_size)
        
        # Initialize iterators if first call
        if not hasattr(self, 'train_iterators') or not self.train_iterators:
            self.train_iterators = {}
            for query_type in train_queries:
                queries = train_queries[query_type]
                self.train_iterators[query_type] = get_queries_iterator(queries, batch_size, self)
        
        self.optimizer.zero_grad()
        
        # Always train on 1-chain
        loss = run_batch_v2(self.train_iterators['1-chain'], self)
        
        # If past burn-in, add other query types
        if getattr(data, 'past_burn_in', False):
            for query_type in train_queries:
                if query_type == "1-chain":
                    continue
                if "inter" in query_type:
                    loss += self.inter_weight * run_batch_v2(self.train_iterators[query_type], self)
                    loss += self.inter_weight * run_batch_v2(self.train_iterators[query_type], self, hard_negatives=True)
                else:
                    loss += self.path_weight * run_batch_v2(self.train_iterators[query_type], self)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def eval_step(self, data, mask=None):
        """Single evaluation step for RGCN query reasoning."""
        self.eval()
        
        queries = getattr(data, 'val_queries', getattr(data, 'test_queries', None))
        iteration = getattr(data, 'current_iteration', 0)
        batch_size = getattr(data, 'batch_size', 128)
        
        if queries is None:
            print("ERROR: No queries found in data")
            return {'loss': float('inf'), 'accuracy': 0.0}
        
        vals = {}
        total_loss = 0.0
        num_query_types = 0
        
        with torch.no_grad():
            for query_type in queries["one_neg"]:
                try:
                    auc, rel_aucs = eval_auc_queries(queries["one_neg"][query_type], self)
                    perc = eval_perc_queries(queries["full_neg"][query_type], self, batch_size)
                    vals[query_type] = auc
                    total_loss += (1.0 - auc)
                    num_query_types += 1
                    
                    # Log like the original method
                    print(f"{query_type} val AUC: {auc:.6f} val perc {perc}; iteration: {iteration}")
                    
                    if "inter" in query_type:
                        auc_hard, _ = eval_auc_queries(queries["one_neg"][query_type], self, hard_negatives=True)
                        vals[query_type + "hard"] = auc_hard
                        total_loss += (1.0 - auc_hard)
                        num_query_types += 1
                        print(f"Hard-{query_type} val AUC: {auc_hard:.6f}; iteration: {iteration}")
                        
                except Exception as e:
                    print(f"ERROR in eval_step for {query_type}: {e}")
                    import traceback
                    traceback.print_exc()
                    vals[query_type] = 0.0
                    total_loss += 1.0
                    num_query_types += 1
        
        avg_loss = total_loss / max(num_query_types, 1)
        avg_auc = np.mean(list(vals.values())) if vals else 0.0
        
        print(f"Final evaluation - avg_auc: {avg_auc}, vals: {vals}")
        
        return {'loss': avg_loss, 'accuracy': avg_auc}

    def get_model_info(self):
        """Override to include RGCN-specific information."""
        info = super().get_model_info()
        info.update({
            'emb_dim': self.emb_dim,
            'readout_function': self.readout_str,
            'scatter_operation': self.scatter_op,
            'adaptive': self.adaptive,
            'shared_layers': self.shared_layers,
            'num_relations': len(self.graph.rel_edges) if hasattr(self.graph, 'rel_edges') else 'unknown'
        })
        return info