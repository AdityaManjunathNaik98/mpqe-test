import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import random
import scipy.stats as stats
from sacred import Ingredient
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
import itertools
import math
import logging
from torch.autograd import Variable

import warnings
warnings.filterwarnings("ignore")


train_ingredient = Ingredient('train')


class DirectEncoder(nn.Module):
    """
    Encodes a node as a embedding via direct lookup.
    (i.e., this is just like basic node2vec or matrix factorization)
    """
    def __init__(self, features, feature_modules): 
        """
        Initializes the model for a specific graph.

        features         -- function mapping (node_list, features, offset) to feature values
                            see torch.nn.EmbeddingBag and forward function below docs for offset meaning.
        feature_modules  -- This should be a map from mode -> torch.nn.EmbeddingBag 
        """
        super(DirectEncoder, self).__init__()
        for name, module in feature_modules.items():
            self.add_module("feat-"+name, module)
        self.features = features

    def forward(self, nodes, mode, offset=None, **kwargs):
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes
        mode      -- string desiginating the mode of the nodes
        offsets   -- specifies how the embeddings are aggregated. 
                     see torch.nn.EmbeddingBag for format. 
                     No aggregation if offsets is None
        """

        if offset is None:
            embeds = self.features(nodes, mode).t()
            norm = embeds.norm(p=2, dim=0, keepdim=True)
            return embeds.div(norm.expand_as(embeds))
        else:
            return self.features(nodes, mode, offset).t()

class Encoder(nn.Module):
    """
    Encodes a node's using a GCN/GraphSage approach
    """
    def __init__(self, features, feature_dims, 
            out_dims, relations, adj_lists, aggregator,
            base_model=None, cuda=False, 
            layer_norm=False,
            feature_modules={}): 
        """
        Initializes the model for a specific graph.

        features         -- function mapping (node_list, features, offset) to feature values
                            see torch.nn.EmbeddingBag and forward function below docs for offset meaning.
        feature_dims     -- output dimension of each of the feature functions. 
        out_dims         -- embedding dimensions for each mode (i.e., output dimensions)
        relations        -- map from mode -> out_going_relations
        adj_lists        -- map from relation_tuple -> node -> list of node's neighbors
        base_model       -- if features are from another encoder, pass it here for training
        cuda             -- whether or not to move params to the GPU
        feature_modules  -- if features come from torch.nn module, pass the modules here for training
        """

        super(Encoder, self).__init__()

        self.features = features
        self.feat_dims = feature_dims
        self.adj_lists = adj_lists
        self.relations = relations
        self.aggregator = aggregator
        for name, module in feature_modules.items():
            self.add_module("feat-"+name, module)
        if base_model != None:
            self.base_model = base_model

        self.out_dims = out_dims
        self.cuda = cuda
        self.aggregator.cuda = cuda
        self.layer_norm = layer_norm
        self.compress_dims = {}
        for source_mode in relations:
            self.compress_dims[source_mode] = self.feat_dims[source_mode]
            for (to_mode, _) in relations[source_mode]:
                self.compress_dims[source_mode] += self.feat_dims[to_mode]

        self.self_params = {}
        self.compress_params = {}
        self.lns = {}
        for mode, feat_dim in self.feat_dims.items():
            if self.layer_norm:
                self.lns[mode] = LayerNorm(out_dims[mode])
                self.add_module(mode+"_ln", self.lns[mode])
            self.compress_params[mode] = nn.Parameter(
                    torch.FloatTensor(out_dims[mode], self.compress_dims[mode]))
            init.xavier_uniform(self.compress_params[mode])
            self.register_parameter(mode+"_compress", self.compress_params[mode])

    def forward(self, nodes, mode, keep_prob=0.5, max_keep=10):
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes
        mode      -- string desiginating the mode of the nodes
        """
        self_feat = self.features(nodes, mode).t()
        neigh_feats = []
        for to_r in self.relations[mode]:
            rel = (mode, to_r[1], to_r[0])
            to_neighs = [[-1] if node == -1 else self.adj_lists[rel][node] for node in nodes]
            
            # Special null neighbor for nodes with no edges of this type
            to_neighs = [[-1] if len(l) == 0 else l for l in to_neighs]
            to_feats = self.aggregator.forward(to_neighs, rel, keep_prob, max_keep)
            to_feats = to_feats.t()
            neigh_feats.append(to_feats)
        
        neigh_feats.append(self_feat)
        combined = torch.cat(neigh_feats, dim=0)
        combined = self.compress_params[mode].mm(combined)
        if self.layer_norm:
            combined = self.lns[mode](combined.t()).t()
        combined = F.relu(combined)
        return combined
    
class LayerNorm(nn.Module):
    """
    Simple layer norm object optionally used with the convolutional encoder.
    """

    def __init__(self, feature_dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones((feature_dim,)))
        self.beta = nn.Parameter(torch.zeros((feature_dim,)))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    

class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, features, cuda=False): 
        """
        Initializes the aggregator for a specific graph.

        features         -- function mapping (node_list, features, offset) to feature values
                            see torch.nn.EmbeddingBag and forward function below docs for offset meaning.
        """

        super(MeanAggregator, self).__init__()

        self.features = features
        self.cuda = cuda
        
    def forward(self, to_neighs, rel, keep_prob=0.5, max_keep=10):
        """
        Aggregates embeddings for a batch of nodes.
        keep_prob and max_keep are the parameters for edge/neighbour dropout.

        to_neighs -- list of neighbors of nodes
        keep_prob -- probability of keeping a neighbor
        max_keep  -- maximum number of neighbors kept per node
        """

        # Local pointers to functions (speed hack)
        _int = int
        _set = set
        _min = min
        _len = len
        _ceil = math.ceil
        _sample = random.sample
        samp_neighs = [_set(_sample(to_neigh, 
                        _min(_int(_ceil(_len(to_neigh)*keep_prob)), max_keep)
                        )) for to_neigh in to_neighs]
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]   
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
        embed_matrix = self.features(unique_nodes_list, rel[-1])
        if len(embed_matrix.size()) == 1:
            embed_matrix = embed_matrix.unsqueeze(dim=0)
        to_feats = mask.mm(embed_matrix)
        return to_feats

class FastMeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, features, cuda=False): 
        """
        Initializes the aggregator for a specific graph.

        features         -- function mapping (node_list, features, offset) to feature values
                            see torch.nn.EmbeddingBag and forward function below docs for offset meaning.
        """

        super(FastMeanAggregator, self).__init__()

        self.features = features
        self.cuda = cuda
        
    def forward(self, to_neighs, rel, keep_prob=None, max_keep=25):
        """
        Aggregates embeddings for a batch of nodes.
        keep_prob and max_keep are the parameters for edge/neighbour dropout.

        to_neighs -- list of neighbors of nodes
        keep_prob -- probability of keeping a neighbor
        max_keep  -- maximum number of neighbors kept per node
        """
        _random = random.random
        _int = int 
        _len = len
        samp_neighs = [to_neigh[_int(_random()*_len(to_neigh))] for i in itertools.repeat(None, max_keep) 
                for to_neigh in to_neighs]
        embed_matrix = self.features(samp_neighs, rel[-1])
        to_feats = embed_matrix.view(max_keep, len(to_neighs), embed_matrix.size()[1])
        return to_feats.mean(dim=0)

class PoolAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean pooling of neighbors' embeddings
    """
    def __init__(self, features, feature_dims, cuda=False): 
        """
        Initializes the aggregator for a specific graph.

        features         -- function mapping (node_list, features, offset) to feature values
                            see torch.nn.EmbeddingBag and forward function below docs for offset meaning.
        """

        super(PoolAggregator, self).__init__()

        self.features = features
        self.feat_dims = feature_dims
        self.pool_matrix = {}
        for mode, feat_dim in self.feat_dims.items():
            self.pool_matrix[mode] = nn.Parameter(torch.FloatTensor(feat_dim, feat_dim))
            init.xavier_uniform(self.pool_matrix[mode])
            self.register_parameter(mode+"_pool", self.pool_matrix[mode])
        self.cuda = cuda
        
    def forward(self, to_neighs, rel, keep_prob=0.5, max_keep=10):
        """
        Aggregates embeddings for a batch of nodes.
        keep_prob and max_keep are the parameters for edge/neighbour dropout.

        to_neighs -- list of neighbors of nodes
        keep_prob -- probability of keeping a neighbor
        max_keep  -- maximum number of neighbors kept per node
        """
        _int = int
        _set = set
        _min = min
        _len = len
        _ceil = math.ceil
        _sample = random.sample
        samp_neighs = [_set(_sample(to_neigh, 
                        _min(_int(_ceil(_len(to_neigh)*keep_prob)), max_keep)
                        )) for to_neigh in to_neighs]
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]   
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        mode = rel[0]
        if self.cuda:
            mask = mask.cuda()
        embed_matrix = self.features(unique_nodes, rel[-1]).mm(self.pool_matrix[mode])
        to_feats = F.relu(mask.mm(embed_matrix))
        return to_feats

class FastPoolAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean pooling of neighbors' embeddings
    """
    def __init__(self, features, feature_dims,
            cuda=False): 
        """
        Initializes the aggregator for a specific graph.

        features         -- function mapping (node_list, features, offset) to feature values
                            see torch.nn.EmbeddingBag and forward function below docs for offset meaning.
        """

        super(FastPoolAggregator, self).__init__()

        self.features = features
        self.feat_dims = feature_dims
        self.pool_matrix = {}
        for mode, feat_dim in self.feat_dims.items():
            self.pool_matrix[mode] = nn.Parameter(torch.FloatTensor(feat_dim, feat_dim))
            init.xavier_uniform(self.pool_matrix[mode])
            self.register_parameter(mode+"_pool", self.pool_matrix[mode])
        self.cuda = cuda
        
    def forward(self, to_neighs, rel, keep_prob=0.5, max_keep=10):
        """
        Aggregates embeddings for a batch of nodes.
        keep_prob and max_keep are the parameters for edge/neighbour dropout.

        to_neighs -- list of neighbors of nodes
        keep_prob -- probability of keeping a neighbor
        max_keep  -- maximum number of neighbors kept per node
        """
        _random = random.random
        _int = int 
        _len = len
        samp_neighs = [to_neigh[_int(_random()*_len(to_neigh))] for i in itertools.repeat(None, max_keep) 
                for to_neigh in to_neighs]
        mode = rel[0]
        embed_matrix = self.features(samp_neighs, rel[-1]).mm(self.pool_matrix[mode])
        to_feats = embed_matrix.view(max_keep, len(to_neighs), embed_matrix.size()[1])
        return to_feats.mean(dim=0)

def cudify(feature_modules, node_maps=None):
   if node_maps is None:
       features = lambda nodes, mode : feature_modules[mode](
            torch.autograd.Variable(torch.LongTensor(nodes)+1).cuda())
   else:
    #    features = lambda nodes, mode : feature_modules[mode](node_maps[nodes].cuda())
       features = lambda nodes, mode : feature_modules[mode](node_maps[nodes].to(feature_modules[mode].weight.device))
   return features

def _get_perc_scores(scores, lengths):
    perc_scores = []
    cum_sum = 0
    neg_scores = scores[len(lengths):]
    for i, length in enumerate(lengths):
        perc_scores.append(stats.percentileofscore(neg_scores[cum_sum:cum_sum+length], scores[i]))
        cum_sum += length
    return perc_scores

def eval_auc_queries(test_queries, enc_dec, batch_size=128, hard_negatives=False, seed=0):
    predictions = []
    labels = []
    formula_aucs = {}
    random.seed(seed)
    for formula in test_queries:
        formula_labels = []
        formula_predictions = []
        formula_queries = test_queries[formula]
        offset = 0
        while offset < len(formula_queries):
            max_index = min(offset+batch_size, len(formula_queries))
            batch_queries = formula_queries[offset:max_index]
            if hard_negatives:
                lengths = [1 for j in range(offset, max_index)]
                negatives = [random.choice(formula_queries[j].hard_neg_samples) for j in range(offset, max_index)]
            else:
                lengths = [1 for j in range(offset, max_index)]
                negatives = [random.choice(formula_queries[j].neg_samples) for j in range(offset, max_index)]
            offset += batch_size

            formula_labels.extend([1 for _ in range(len(lengths))])
            formula_labels.extend([0 for _ in range(len(negatives))])

            targets = [q.target_node for q in batch_queries]
            batch_scores = enc_dec.forward_query(formula, batch_queries, targets,
                                           neg_nodes=negatives,
                                           neg_lengths=lengths)

            batch_scores = batch_scores.data.tolist()
            formula_predictions.extend(batch_scores)
        formula_aucs[formula] = roc_auc_score(formula_labels, np.nan_to_num(formula_predictions))
        labels.extend(formula_labels)
        predictions.extend(formula_predictions)
    overall_auc = roc_auc_score(labels, np.nan_to_num(predictions))
    return overall_auc, formula_aucs

    
def eval_perc_queries(test_queries, enc_dec, batch_size=128, hard_negatives=False):
    perc_scores = []
    for formula in test_queries:
        formula_queries = test_queries[formula]
        offset = 0
        while offset < len(formula_queries):
            max_index = min(offset+batch_size, len(formula_queries))
            batch_queries = formula_queries[offset:max_index]
            if hard_negatives:
                lengths = [len(formula_queries[j].hard_neg_samples) for j in range(offset, max_index)]
                negatives = [n for j in range(offset, max_index) for n in formula_queries[j].hard_neg_samples]
            else:
                lengths = [len(formula_queries[j].neg_samples) for j in range(offset, max_index)]
                negatives = [n for j in range(offset, max_index) for n in formula_queries[j].neg_samples]
            offset += batch_size

            targets = [q.target_node for q in batch_queries]
            batch_scores = enc_dec.forward(formula, batch_queries, targets,
                                           neg_nodes=negatives,
                                           neg_lengths=lengths)

            batch_scores = batch_scores.data.tolist()
            perc_scores.extend(_get_perc_scores(batch_scores, lengths))
    return np.mean(perc_scores)

def get_encoder(depth, graph, out_dims, feature_modules, cuda): 
    if depth < 0 or depth > 3:
        raise Exception("Depth must be between 0 and 3 (inclusive)")

    if depth == 0:
         enc = DirectEncoder(graph.features, feature_modules)
    else:
        aggregator1 = MeanAggregator(graph.features)
        enc1 = Encoder(graph.features, 
                graph.feature_dims, 
                out_dims, 
                graph.relations, 
                graph.adj_lists, feature_modules=feature_modules, 
                cuda=cuda, aggregator=aggregator1)
        enc = enc1
        if depth >= 2:
            aggregator2 = MeanAggregator(lambda nodes, mode : enc1(nodes, mode).t().squeeze())
            enc2 = Encoder(lambda nodes, mode : enc1(nodes, mode).t().squeeze(),
                    enc1.out_dims, 
                    out_dims, 
                    graph.relations, 
                    graph.adj_lists, base_model=enc1,
                    cuda=cuda, aggregator=aggregator2)
            enc = enc2
            if depth >= 3:
                aggregator3 = MeanAggregator(lambda nodes, mode : enc2(nodes, mode).t().squeeze())
                enc3 = Encoder(lambda nodes, mode : enc1(nodes, mode).t().squeeze(),
                        enc2.out_dims, 
                        out_dims, 
                        graph.relations, 
                        graph.adj_lists, base_model=enc2,
                        cuda=cuda, aggregator=aggregator3)
                enc = enc3
    return enc


def check_conv(vals, window=2, tol=1e-6):
    if len(vals) < 2 * window:
        return False
    conv = np.mean(vals[-window:]) - np.mean(vals[-2*window:-window]) 
    return conv < tol


def update_loss(loss, losses, ema_loss, ema_alpha=0.01):
    losses.append(loss)
    if ema_loss is None:
        ema_loss = loss
    else:
        ema_loss = (1-ema_alpha)*ema_loss + ema_alpha*loss
    return losses, ema_loss


def run_batch(train_queries, enc_dec, iter_count, batch_size, hard_negatives=False):
    num_queries = [float(len(queries)) for queries in list(train_queries.values())]
    denom = float(sum(num_queries))
    formula_index = np.argmax(np.random.multinomial(1, 
            np.array(num_queries)/denom))
    formula = list(train_queries.keys())[formula_index]
    n = len(train_queries[formula])
    start = (iter_count * batch_size) % n
    end = min(((iter_count+1) * batch_size) % n, n)
    end = n if end <= start else end
    queries = train_queries[formula][start:end]
    loss = enc_dec.margin_loss(formula, queries, hard_negatives=hard_negatives)
    return loss


def run_batch_v2(queries_iterator, enc_dec, hard_negatives=False):
    enc_dec.train()

    batch = next(queries_iterator)
    loss = enc_dec.margin_loss(*batch, hard_negatives=hard_negatives)
    return loss

def setup_logging(log_file, console=True):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=log_file,
                        filemode='w')
    if console:
        console = logging.StreamHandler()
        # optional, set the logging level
        console.setLevel(logging.INFO)
        # set a format which is the same for console use
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)
    return logging
