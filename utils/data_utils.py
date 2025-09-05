from collections import defaultdict, OrderedDict
import pickle as pickle
from multiprocessing import Process
from collections import Counter
from multiprocessing import cpu_count
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch_geometric.data import Data, Batch
import torch
import os.path as osp
import random
import os


class Formula():

    def __init__(self, query_type, rels):
        self.query_type = query_type
        self.target_mode = rels[0][0]
        self.rels = rels
        if query_type == "1-chain" or query_type == "2-chain" or query_type == "3-chain":
            self.anchor_modes = (rels[-1][-1],)
        elif query_type == "2-inter" or query_type == "3-inter":
            self.anchor_modes = tuple([rel[-1] for rel in rels])
        elif query_type == "3-inter_chain":
            self.anchor_modes = (rels[0][-1], rels[1][-1][-1])
        elif query_type == "3-chain_inter":
            self.anchor_modes = (rels[1][0][-1], rels[1][1][-1])

    @staticmethod
    def flatten(S):
        if len(S) == 0:
            return S
        if isinstance(S[0], tuple):
            return Formula.flatten(S[0]) + Formula.flatten(S[1:])
        return S[:1] + Formula.flatten(S[1:])

    def get_rels(self):
        flat_rels = Formula.flatten(self.rels)
        rels = []
        for i in range(0, len(flat_rels), 3):
            rels.append(tuple(flat_rels[i:i+3]))
        return rels

    def get_nodes(self):
        flat_rels = Formula.flatten(self.rels)
        variables = []
        for i in range(0, len(flat_rels), 3):
            variables.extend([flat_rels[i], flat_rels[i+2]])
        return variables

    def __hash__(self):
         return hash((self.query_type, self.rels))

    def __eq__(self, other):
        return ((self.query_type, self.rels)) == ((other.query_type, other.rels))

    def __neq__(self, other):
        return ((self.query_type, self.rels)) != ((other.query_type, other.rels))

    def __str__(self):
        return self.query_type + ": " + str(self.rels)

class Query():

    def __init__(self, query_graph, neg_samples, hard_neg_samples, neg_sample_max=100, keep_graph=False):
        query_type = query_graph[0]
        if query_type == "1-chain" or query_type == "2-chain" or query_type == "3-chain":
            self.formula = Formula(query_type, tuple([query_graph[i][1] for i in range(1, len(query_graph))]))
            self.anchor_nodes = (query_graph[-1][-1],)
        elif query_type == "2-inter" or query_type == "3-inter":
            self.formula = Formula(query_type, tuple([query_graph[i][1] for i in range(1, len(query_graph))]))
            self.anchor_nodes = tuple([query_graph[i][-1] for i in range(1, len(query_graph))])
        elif query_type == "3-inter_chain":
            self.formula = Formula(query_type, (query_graph[1][1], (query_graph[2][0][1], query_graph[2][1][1])))
            self.anchor_nodes = (query_graph[1][-1], query_graph[2][-1][-1])
        elif query_type == "3-chain_inter":
            self.formula = Formula(query_type, (query_graph[1][1], (query_graph[2][0][1], query_graph[2][1][1])))
            self.anchor_nodes = (query_graph[2][0][-1], query_graph[2][1][-1])

        self.target_node = query_graph[1][0]
        if keep_graph:
            self.query_graph = query_graph
        else:
            self.query_graph = None
        if not neg_samples is None:
            self.neg_samples = list(neg_samples) if len(neg_samples) < neg_sample_max else random.sample(neg_samples, neg_sample_max)
        else:
            self.neg_samples = None
        if not hard_neg_samples is None:
            self.hard_neg_samples = list(hard_neg_samples) if len(hard_neg_samples) <= neg_sample_max else random.sample(hard_neg_samples, neg_sample_max)
        else:
            self.hard_neg_samples =  None

    def contains_edge(self, edge):
        if self.query_graph is None:
            raise Exception("Can only test edge contain if graph is kept. Reinit with keep_graph=True")
        edges =  self.query_graph[1:]
        if "inter_chain" in self.query_graph[0] or "chain_inter" in self.query_graph[0]:
            edges = (edges[0], edges[1][0], edges[1][1])
        return edge in edges or (edge[1], _reverse_relation(edge[1]), edge[0]) in edges

    def get_edges(self):
        if self.query_graph is None:
            raise Exception("Can only test edge contain if graph is kept. Reinit with keep_graph=True")
        edges =  self.query_graph[1:]
        if "inter_chain" in self.query_graph[0] or "chain_inter" in self.query_graph[0]:
            edges = (edges[0], edges[1][0], edges[1][1])
        return set(edges).union(set([(e[-1], _reverse_relation(e[1]), e[0]) for e in edges]))

    def __hash__(self):
         return hash((self.formula, self.target_node, self.anchor_nodes))

    def __eq__(self, other):
        return (self.formula, self.target_node, self.anchor_nodes) == (other.formula, other.target_node, other.anchor_nodes)

    def __neq__(self, other):
        return self.__hash__() != other.__hash__()

    def serialize(self):
        if self.query_graph is None:
            raise Exception("Cannot serialize query loaded with query graph!")
        return (self.query_graph, self.neg_samples, self.hard_neg_samples)

    @staticmethod
    def deserialize(serial_info, keep_graph=False):
        return Query(serial_info[0], serial_info[1], serial_info[2], None if serial_info[1] is None else len(serial_info[1]), keep_graph=keep_graph)



class Graph():
    """
    Simple container for heteregeneous graph data.
    """
    def __init__(self, features, feature_dims, relations, adj_lists):
        self.features = features
        self.feature_dims = feature_dims
        self.relations = relations
        self.adj_lists = adj_lists
        self.full_sets = defaultdict(set)
        self.full_lists = {}
        self.meta_neighs = defaultdict(dict)
        for rel, adjs in self.adj_lists.items():
            full_set = set(self.adj_lists[rel].keys())
            self.full_sets[rel[0]] = self.full_sets[rel[0]].union(full_set)
        for mode, full_set in self.full_sets.items():
            self.full_lists[mode] = list(full_set)
        self._cache_edge_counts()
        self._make_flat_adj_lists()

    def _make_flat_adj_lists(self):
        self.flat_adj_lists = defaultdict(lambda : defaultdict(list))
        for rel, adjs in self.adj_lists.items():
            for node, neighs in adjs.items():
                self.flat_adj_lists[rel[0]][node].extend([(rel, neigh) for neigh in neighs])

    def _cache_edge_counts(self):
        self.edges = 0.
        self.rel_edges = {}
        for r1 in self.relations:
            for r2 in self.relations[r1]:
                rel = (r1,r2[1], r2[0])
                self.rel_edges[rel] = 0.
                for adj_list in list(self.adj_lists[rel].values()):
                    self.rel_edges[rel] += len(adj_list)
                    self.edges += 1.
        self.rel_weights = OrderedDict()
        self.mode_edges = defaultdict(float)
        self.mode_weights = OrderedDict()
        for rel, edge_count in self.rel_edges.items():
            self.rel_weights[rel] = edge_count / self.edges
            self.mode_edges[rel[0]] += edge_count
        for mode, edge_count in self.mode_edges.items():
            self.mode_weights[mode] = edge_count / self.edges

    def remove_edges(self, edge_list):
        for edge in edge_list:
            try:
                self.adj_lists[edge[1]][edge[0]].remove(edge[-1])
            except Exception:
                continue

            try:
                self.adj_lists[_reverse_relation(edge[1])][edge[-1]].remove(edge[0])
            except Exception:
                continue
        self.meta_neighs = defaultdict(dict)
        self._cache_edge_counts()
        self._make_flat_adj_lists()

    def get_all_edges(self, seed=0, exclude_rels=set([])):
        """
        Returns all edges in the form (node1, relation, node2)
        """
        edges = []
        random.seed(seed)
        for rel, adjs in self.adj_lists.items():
            if rel in exclude_rels:
                continue
            for node, neighs in adjs.items():
                edges.extend([(node, rel, neigh) for neigh in neighs if neigh != -1])
        random.shuffle(edges)
        return edges

    def get_all_edges_byrel(self, seed=0, 
            exclude_rels=set([])):
        random.seed(seed)
        edges = defaultdict(list)
        for rel, adjs in self.adj_lists.items():
            if rel in exclude_rels:
                continue
            for node, neighs in adjs.items():
                edges[(rel,)].extend([(node, neigh) for neigh in neighs if neigh != -1])

    def get_negative_edge_samples(self, edge, num, rejection_sample=True):
        if rejection_sample:
            neg_nodes = set([])
            counter = 0
            while len(neg_nodes) < num:
                neg_node = random.choice(self.full_lists[edge[1][0]])
                if not neg_node in self.adj_lists[_reverse_relation(edge[1])][edge[2]]:
                    neg_nodes.add(neg_node)
                counter += 1
                if counter > 100*num:
                    return self.get_negative_edge_samples(edge, num, rejection_sample=False)
        else:
            neg_nodes = self.full_sets[edge[1][0]] - self.adj_lists[_reverse_relation(edge[1])][edge[2]]
        neg_nodes = list(neg_nodes) if len(neg_nodes) <= num else random.sample(list(neg_nodes), num)
        return neg_nodes

    def sample_test_queries(self, train_graph, q_types, samples_per_type, neg_sample_max, verbose=True):
        queries = []
        for q_type in q_types:
            sampled = 0
            while sampled < samples_per_type:
                q = self.sample_query_subgraph_bytype(q_type)
                if q is None or not train_graph._is_negative(q, q[1][0], False):
                    continue
                negs, hard_negs = self.get_negative_samples(q)
                if negs is None or ("inter" in q[0] and hard_negs is None):
                    continue
                query = Query(q, negs, hard_negs, neg_sample_max=neg_sample_max, keep_graph=True)
                queries.append(query)
                sampled += 1
                if sampled % 1000 == 0 and verbose:
                    print("Sampled", sampled)
        return queries

    def sample_queries(self, arity, num_samples, neg_sample_max, verbose=True):
        sampled = 0
        queries = []
        while sampled < num_samples:
            q = self.sample_query_subgraph(arity)
            if q is None:
                continue
            negs, hard_negs = self.get_negative_samples(q)
            if negs is None or ("inter" in q[0] and hard_negs is None):
                continue
            query = Query(q, negs, hard_negs, neg_sample_max=neg_sample_max, keep_graph=True)
            queries.append(query)
            sampled += 1
            if sampled % 1000 == 0 and verbose:
                print("Sampled", sampled)
        return queries


    def get_negative_samples(self, query):
        if query[0] == "3-chain" or query[0] == "2-chain":
            edges = query[1:]
            rels = [_reverse_relation(edge[1]) for edge in edges[::-1]]
            meta_neighs = self.get_metapath_neighs(query[-1][-1], tuple(rels))
            negative_samples = self.full_sets[query[1][1][0]] - meta_neighs
            if len(negative_samples) == 0:
                return None, None
            else:
                return negative_samples, None
        elif query[0] == "2-inter" or query[0] == "3-inter":
            rel_1 = _reverse_relation(query[1][1])
            union_neighs = self.adj_lists[rel_1][query[1][-1]]
            inter_neighs = self.adj_lists[rel_1][query[1][-1]]
            for i in range(2,len(query)):
                rel = _reverse_relation(query[i][1])
                union_neighs = union_neighs.union(self.adj_lists[rel][query[i][-1]])
                inter_neighs = inter_neighs.intersection(self.adj_lists[rel][query[i][-1]])
            neg_samples = self.full_sets[query[1][1][0]] - inter_neighs
            hard_neg_samples = union_neighs - inter_neighs
            if len(neg_samples) == 0 or len(hard_neg_samples) == 0:
                return None, None
            return neg_samples, hard_neg_samples
        elif query[0] == "3-inter_chain":
            rel_1 = _reverse_relation(query[1][1])
            union_neighs = self.adj_lists[rel_1][query[1][-1]]
            inter_neighs = self.adj_lists[rel_1][query[1][-1]]
            chain_rels = [_reverse_relation(edge[1]) for edge in query[2][::-1]]
            chain_neighs = self.get_metapath_neighs(query[2][-1][-1], tuple(chain_rels))
            union_neighs = union_neighs.union(chain_neighs)
            inter_neighs = inter_neighs.intersection(chain_neighs)
            neg_samples = self.full_sets[query[1][1][0]] - inter_neighs
            hard_neg_samples = union_neighs - inter_neighs
            if len(neg_samples) == 0 or len(hard_neg_samples) == 0:
                return None, None
            return neg_samples, hard_neg_samples
        elif query[0] == "3-chain_inter":
            inter_rel_1 = _reverse_relation(query[-1][0][1])
            inter_neighs_1 = self.adj_lists[inter_rel_1][query[-1][0][-1]]
            inter_rel_2 = _reverse_relation(query[-1][1][1])
            inter_neighs_2 = self.adj_lists[inter_rel_2][query[-1][1][-1]]
            
            inter_neighs = inter_neighs_1.intersection(inter_neighs_2)
            union_neighs = inter_neighs_1.union(inter_neighs_2)
            rel = _reverse_relation(query[1][1])
            pos_nodes = set([n for neigh in inter_neighs for n in self.adj_lists[rel][neigh]]) 
            union_pos_nodes = set([n for neigh in union_neighs for n in self.adj_lists[rel][neigh]]) 
            neg_samples = self.full_sets[query[1][1][0]] - pos_nodes
            hard_neg_samples = union_pos_nodes - pos_nodes
            if len(neg_samples) == 0 or len(hard_neg_samples) == 0:
                return None, None
            return neg_samples, hard_neg_samples

    def sample_edge(self, node, mode):
        rel, neigh = random.choice(self.flat_adj_lists[mode][node])
        edge = (node, rel, neigh)
        return edge

    def sample_query_subgraph_bytype(self, q_type, start_node=None):
        if start_node is None:
            start_rel = random.choice(list(self.adj_lists.keys()))
            node = random.choice(list(self.adj_lists[start_rel].keys()))
            mode = start_rel[0]
        else:
            node, mode = start_node

        if q_type[0] == "3":
            if q_type == "3-chain" or q_type == "3-chain_inter":
                num_edges = 1
            elif q_type == "3-inter_chain":
                num_edges = 2
            elif q_type == "3-inter":
                num_edges = 3
            if num_edges > len(self.flat_adj_lists[mode][node]):
                return None
            if num_edges == 1:
                rel, neigh = random.choice(self.flat_adj_lists[mode][node])
                edge = (node, rel, neigh)
                next_query = self.sample_query_subgraph_bytype(
                    "2-chain" if q_type == "3-chain" else "2-inter", start_node=(neigh, rel[-1]))
                if next_query is None:
                    return None
                if next_query[0] == "2-chain":
                    return ("3-chain", edge, next_query[1], next_query[2])
                else:
                    return ("3-chain_inter", edge, (next_query[1], next_query[2]))
            elif num_edges == 2:
                rel_1, neigh_1 = random.choice(self.flat_adj_lists[mode][node])
                edge_1 = (node, rel_1, neigh_1)
                neigh_2 = neigh_1
                rel_2 = rel_1
                while (neigh_1, rel_1) == (neigh_2, rel_2):
                    rel_2, neigh_2 = random.choice(self.flat_adj_lists[mode][node])
                edge_2 = (node, rel_2, neigh_2)
                return ("3-inter_chain", edge_1, (edge_2, self.sample_edge(neigh_2, rel_2[-1])))
            elif num_edges == 3:
                rel_1, neigh_1 = random.choice(self.flat_adj_lists[mode][node])
                edge_1 = (node, rel_1, neigh_1)
                neigh_2 = neigh_1
                rel_2 = rel_1
                while (rel_1, neigh_1) == (rel_2, neigh_2):
                    rel_2, neigh_2 = random.choice(self.flat_adj_lists[mode][node])
                edge_2 = (node, rel_2, neigh_2)
                neigh_3 = neigh_1
                rel_3 = rel_1
                while ((rel_1, neigh_1) == (rel_3, neigh_3)) or ((rel_2, neigh_2) == (rel_3, neigh_3)):
                    rel_3, neigh_3 = random.choice(self.flat_adj_lists[mode][node])
                edge_3 = (node, rel_3, neigh_3)
                return ("3-inter", edge_1, edge_2, edge_3)

        if q_type[0] == "2":
            num_edges = 1 if q_type == "2-chain" else 2
            if num_edges > len(self.flat_adj_lists[mode][node]):
                return None
            if num_edges == 1:
                rel, neigh = random.choice(self.flat_adj_lists[mode][node])
                edge = (node, rel, neigh)
                return ("2-chain", edge, self.sample_edge(neigh, rel[-1]))
            elif num_edges == 2:
                rel_1, neigh_1 = random.choice(self.flat_adj_lists[mode][node])
                edge_1 = (node, rel_1, neigh_1)
                neigh_2 = neigh_1
                rel_2 = rel_1
                while (neigh_1, rel_1) == (neigh_2, rel_2):
                    rel_2, neigh_2 = random.choice(self.flat_adj_lists[mode][node])
                edge_2 = (node, rel_2, neigh_2)
                return ("2-inter", edge_1, edge_2)


    def sample_query_subgraph(self, arity, start_node=None):
        if start_node is None:
            start_rel = random.choice(list(self.adj_lists.keys()))
            node = random.choice(list(self.adj_lists[start_rel].keys()))
            mode = start_rel[0]
        else:
            node, mode = start_node
        if arity > 3 or arity < 2:
            raise Exception("Only arity of at most 3 is supported for queries")

        if arity == 3:
            # 1/2 prob of 1 edge, 1/4 prob of 2, 1/4 prob of 3
            num_edges = random.choice([1,1,2,3])
            if num_edges > len(self.flat_adj_lists[mode][node]):
                return None
            if num_edges == 1:
                rel, neigh = random.choice(self.flat_adj_lists[mode][node])
                edge = (node, rel, neigh)
                next_query = self.sample_query_subgraph(2, start_node=(neigh, rel[-1]))
                if next_query is None:
                    return None
                if next_query[0] == "2-chain":
                    return ("3-chain", edge, next_query[1], next_query[2])
                else:
                    return ("3-chain_inter", edge, (next_query[1], next_query[2]))
            elif num_edges == 2:
                rel_1, neigh_1 = random.choice(self.flat_adj_lists[mode][node])
                edge_1 = (node, rel_1, neigh_1)
                neigh_2 = neigh_1
                rel_2 = rel_1
                while (neigh_1, rel_1) == (neigh_2, rel_2):
                    rel_2, neigh_2 = random.choice(self.flat_adj_lists[mode][node])
                edge_2 = (node, rel_2, neigh_2)
                return ("3-inter_chain", edge_1, (edge_2, self.sample_edge(neigh_2, rel_2[-1])))
            elif num_edges == 3:
                rel_1, neigh_1 = random.choice(self.flat_adj_lists[mode][node])
                edge_1 = (node, rel_1, neigh_1)
                neigh_2 = neigh_1
                rel_2 = rel_1
                while (rel_1, neigh_1) == (rel_2, neigh_2):
                    rel_2, neigh_2 = random.choice(self.flat_adj_lists[mode][node])
                edge_2 = (node, rel_2, neigh_2)
                neigh_3 = neigh_1
                rel_3 = rel_1
                while ((rel_1, neigh_1) == (rel_3, neigh_3)) or ((rel_2, neigh_2) == (rel_3, neigh_3)):
                    rel_3, neigh_3 = random.choice(self.flat_adj_lists[mode][node])
                edge_3 = (node, rel_3, neigh_3)
                return ("3-inter", edge_1, edge_2, edge_3)

        if arity == 2:
            num_edges = random.choice([1,2])
            if num_edges > len(self.flat_adj_lists[mode][node]):
                return None
            if num_edges == 1:
                rel, neigh = random.choice(self.flat_adj_lists[mode][node])
                edge = (node, rel, neigh)
                return ("2-chain", edge, self.sample_edge(neigh, rel[-1]))
            elif num_edges == 2:
                rel_1, neigh_1 = random.choice(self.flat_adj_lists[mode][node])
                edge_1 = (node, rel_1, neigh_1)
                neigh_2 = neigh_1
                rel_2 = rel_1
                while (neigh_1, rel_1) == (neigh_2, rel_2):
                    rel_2, neigh_2 = random.choice(self.flat_adj_lists[mode][node])
                edge_2 = (node, rel_2, neigh_2)
                return ("2-inter", edge_1, edge_2)

    def get_metapath_neighs(self, node, rels):
        if node in self.meta_neighs[rels]:
            return self.meta_neighs[rels][node]
        current_set = [node]
        for rel in rels:
            new_set = set()
            for n in current_set:
                # For small graphs, n might not have such neighbors
                if n in self.adj_lists[rel]:
                    new_set = new_set.union(self.adj_lists[rel][n])

            current_set = new_set

        self.meta_neighs[rels][node] = current_set
        return current_set

    ## TESTING CODE

    def _check_edge(self, query, i):
        return query[i][-1] in self.adj_lists[query[i][1]][query[i][0]]

    def _is_subgraph(self, query, verbose):
        if query[0] == "3-chain":
            for i in range(3):
                if not self._check_edge(query, i+1):
                    raise Exception(str(query))
            if not (query[1][-1] == query[2][0] and query[2][-1] == query[3][0]):
                raise Exception(str(query))
        if query[0] == "2-chain":
            for i in range(2):
                if not self._check_edge(query, i+1):
                    raise Exception(str(query))
            if not query[1][-1] == query[2][0]:
                raise Exception(str(query))
        if query[0] == "2-inter":
            for i in range(2):
                if not self._check_edge(query, i+1):
                    raise Exception(str(query))
            if not query[1][0] == query[2][0]:
                raise Exception(str(query))
        if query[0] == "3-inter":
            for i in range(3):
                if not self._check_edge(query, i+1):
                    raise Exception(str(query))
            if not (query[1][0] == query[2][0] and query[2][0] == query[3][0]):
                raise Exception(str(query))
        if query[0] == "3-inter_chain":
            if not (self._check_edge(query, 1) and self._check_edge(query[2], 0) and self._check_edge(query[2], 1)):
                raise Exception(str(query))
            if not (query[1][0] == query[2][0][0] and query[2][0][-1] == query[2][1][0]):
                raise Exception(str(query))
        if query[0] == "3-chain_inter":
            if not (self._check_edge(query, 1) and self._check_edge(query[2], 0) and self._check_edge(query[2], 1)):
                raise Exception(str(query))
            if not (query[1][-1] == query[2][0][0] and query[2][0][0] == query[2][1][0]):
                raise Exception(str(query))
        return True

    def _is_negative(self, query, neg_node, is_hard):
        if query[0] == "2-chain":
            query = (query[0], (neg_node, query[1][1], query[1][2]), query[2])
            if query[2][-1] in self.get_metapath_neighs(query[1][0], (query[1][1], query[2][1])):
                return False
        if query[0] == "3-chain":
            query = (query[0], (neg_node, query[1][1], query[1][2]), query[2], query[3])
            if query[3][-1] in self.get_metapath_neighs(query[1][0], (query[1][1], query[2][1], query[3][1])):
                return False
        if query[0] == "2-inter":
            query = (query[0], (neg_node, query[1][1], query[1][2]), (neg_node, query[2][1], query[2][2]))
            if not is_hard:
                if self._check_edge(query, 1) and self._check_edge(query, 2):
                    return False
            else:
                if (self._check_edge(query, 1) and self._check_edge(query, 2)) or not (self._check_edge(query, 1) or self._check_edge(query, 2)):
                    return False
        if query[0] == "3-inter":
            query = (query[0], (neg_node, query[1][1], query[1][2]), (neg_node, query[2][1], query[2][2]), (neg_node, query[3][1], query[3][2]))
            if not is_hard:
                if self._check_edge(query, 1) and self._check_edge(query, 2) and self._check_edge(query, 3):
                    return False
            else:
                if (self._check_edge(query, 1) and self._check_edge(query, 2) and self._check_edge(query, 3))\
                        or not (self._check_edge(query, 1) or self._check_edge(query, 2) or self._check_edge(query, 3)):
                    return False
        if query[0] == "3-inter_chain":
            query = (query[0], (neg_node, query[1][1], query[1][2]), ((neg_node, query[2][0][1], query[2][0][2]), query[2][1]))
            meta_check = lambda : query[2][-1][-1] in self.get_metapath_neighs(query[1][0], (query[2][0][1], query[2][1][1]))
            neigh_check = lambda : self._check_edge(query, 1)
            if not is_hard:
                if meta_check() and neigh_check():
                    return False
            else:
                if (meta_check() and neigh_check()) or not (meta_check() or neigh_check()):
                    return False
        if query[0] == "3-chain_inter":
            query = (query[0], (neg_node, query[1][1], query[1][2]), query[2])
            target_neigh = self.adj_lists[query[1][1]][neg_node]
            neigh_1 = self.adj_lists[_reverse_relation(query[2][0][1])][query[2][0][-1]]
            neigh_2 = self.adj_lists[_reverse_relation(query[2][1][1])][query[2][1][-1]]
            if not is_hard:
                if target_neigh in neigh_1.intersection(neigh_2):
                    return False
            else:
                if target_neigh in neigh_1.intersection(neigh_2) and not target_neigh in neigh_1.union(neigh_2):
                    return False
        return True

            

    def _run_test(self, num_samples=1000):
        for i in range(num_samples):
            q = self.sample_query_subgraph(2)
            if q is None:
                continue
            self._is_subgraph(q, True)
            negs, hard_negs = self.get_negative_samples(q)
            if not negs is None:
                for n in negs:
                    self._is_negative(q, n, False)
            if not hard_negs is None:
                for n in hard_negs:
                    self._is_negative(q, n, True)
            q = self.sample_query_subgraph(3)
            if q is None:
                continue
            self._is_subgraph(q, True)
            negs, hard_negs = self.get_negative_samples(q)
            if not negs is None:
                for n in negs:
                    self._is_negative(q, n, False)
            if not hard_negs is None:
                for n in hard_negs:
                    self._is_negative(q, n, True)
        return True


def _reverse_relation(relation):
    return (relation[-1], relation[1], relation[0])

def _reverse_edge(edge):
    return (edge[-1], _reverse_relation(edge[1]), edge[0])


def load_graph(data_dir, embed_dim):
    rels, adj_lists, node_maps = pickle.load(open(data_dir+"/graph_data.pkl", "rb"))
    node_mode_counts = {mode: len(node_maps[mode]) for mode in node_maps}
    num_nodes = sum(node_mode_counts.values())

    new_node_maps = torch.ones(num_nodes + 1, dtype=torch.long).fill_(-1)
    for m, id_list in node_maps.items():
        for i, n in enumerate(id_list):
            assert new_node_maps[n] == -1
            new_node_maps[n] = i

    node_maps = new_node_maps
    feature_dims = {m : embed_dim for m in rels}
    feature_modules = {m : torch.nn.Embedding(node_mode_counts[m] + 1, embed_dim) for m in rels}
    for mode in rels:
        feature_modules[mode].weight.data.normal_(0, 1./embed_dim)

    features = lambda nodes, mode: feature_modules[mode](node_maps[nodes])
    graph = Graph(features, feature_dims, rels, adj_lists)
    return graph, feature_modules, node_maps


def make_test_queries(data_dir):
    graph_loader = lambda : load_graph(data_dir, 10)[0]
    sample_clean_test(graph_loader, data_dir)


def clean_test_queries(data_dir):
    test_edges = pickle.load(open(osp.join(data_dir, 'test_edges.pkl'), "rb"))
    val_edges = pickle.load(open(osp.join(data_dir, 'val_edges.pkl'), "rb"))
    deleted_edges = set([q[0][1] for q in test_edges] + [_reverse_edge(q[0][1]) for q in test_edges] +
                [q[0][1] for q in val_edges] + [_reverse_edge(q[0][1]) for q in val_edges])

    for i in range(2,4):
        for kind in ["val", "test"]:
            if kind == "val":
                to_keep = 1000
            else:
                to_keep = 10000
            test_queries = load_queries_by_type(data_dir+"/{:s}_queries_{:d}.pkl".format(kind, i), keep_graph=True)
            print("Loaded", i, kind)
            for query_type in test_queries:
                test_queries[query_type] = [q for q in test_queries[query_type] if len(q.get_edges().intersection(deleted_edges)) > 0]
                test_queries[query_type] = test_queries[query_type][:to_keep]

            print(f'Done making {i:d}-{kind} queries:')
            for q_type in test_queries:
                print(f'\t{q_type}: {len(test_queries[q_type])}')

            test_queries = [q.serialize() for queries in list(test_queries.values()) for q in queries]
            pickle.dump(test_queries, open(data_dir+"/{:s}_queries_{:d}.pkl".format(kind, i), "wb"), protocol=pickle.HIGHEST_PROTOCOL)


def make_train_test_edge_data(data_dir):
    print("Loading graph...")
    graph, _, _ = load_graph(data_dir, 10)
    print("Getting all edges...")
    edges = graph.get_all_edges()
    split_point = int(0.1*len(edges))
    val_test_edges_all = edges[:split_point]
    val_test_edges = []
    print("Getting test negative samples...")
    val_test_edge_negsamples = []
    non_neg_edges = set()

    for e in val_test_edges_all:
        neg_samples = graph.get_negative_edge_samples(e, 100)
        # In some special cases there might not be any valid negative samples,
        # for instance for the edge (topic, rdf:type, class): since all
        # topics are of type Topic (a class), there are no entities of type
        # topic *not* related to the class Topic through the type relationship.
        if len(neg_samples) > 0:
            val_test_edges.append(e)
            val_test_edge_negsamples.append(neg_samples)
        elif e[1] not in non_neg_edges:
            non_neg_edges.add(e[1])
            print('Omitting edges of type', e[1])

    print("Making and storing test queries.")
    val_test_edge_queries = [Query(("1-chain", val_test_edges[i]), val_test_edge_negsamples[i], None, 100, True) for i in range(len(val_test_edges))]
    val_split_point = int(0.1*len(val_test_edge_queries))
    val_queries = val_test_edge_queries[:val_split_point]
    test_queries = val_test_edge_queries[val_split_point:]
    pickle.dump([q.serialize() for q in val_queries], open(data_dir+"/val_edges.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([q.serialize() for q in test_queries], open(data_dir+"/test_edges.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    print("Removing test edges...")
    graph.remove_edges(val_test_edges)
    print("Making and storing train queries.")
    train_edges = graph.get_all_edges()
    train_queries = [Query(("1-chain", e), None, None, keep_graph=True) for e in train_edges]
    pickle.dump([q.serialize() for q in train_queries], open(data_dir+"/train_edges.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)


def _discard_negatives(file_name, prob=0.9):
    """Discard all but one negative sample for each query, with probability
    prob"""
    queries = pickle.load(open(file_name, "rb"))
    queries = [q if random.random() > prob else (q[0], [random.choice(list(q[1]))], None if q[2] is None else [random.choice(list(q[2]))]) for q in queries]
    pickle.dump(queries, open(file_name, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    print("Finished", file_name)


def discard_negatives(data_dir):
    _discard_negatives(data_dir + "/val_edges.pkl")
    _discard_negatives(data_dir + "/test_edges.pkl")
    for i in range(2, 4):
        _discard_negatives(data_dir + "/val_queries_{:d}.pkl".format(i))
        _discard_negatives(data_dir + "/test_queries_{:d}.pkl".format(i))


def print_query_stats(queries):
    counts = Counter()
    for q in queries:
        q_type = q.formula.query_type
        counts[q_type] += 1

    for q_type in counts:
        print(f'\t{q_type}: {counts[q_type]}')


def make_train_queries(data_dir):
    print('Making training queries...')
    graph, _, _ = load_graph(data_dir, 10)
    num_samples = 1e6
    num_workers = cpu_count()
    samples_per_worker = num_samples // num_workers
    queries_2, queries_3 = parallel_sample(graph, num_workers, samples_per_worker, data_dir, test=False)

    print('Done making training queries:')
    print_query_stats(queries_2)
    print_query_stats(queries_3)

    pickle.dump([q.serialize() for q in queries_2], open(data_dir + "/train_queries_2.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([q.serialize() for q in queries_3], open(data_dir + "/train_queries_3.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)


def load_queries(data_file, keep_graph=False):
    raw_info = pickle.load(open(data_file, "rb"))
    return [Query.deserialize(info, keep_graph=keep_graph) for info in raw_info]


def load_queries_by_formula(data_file):
    raw_info = pickle.load(open(data_file, "rb"))
    queries = defaultdict(lambda : defaultdict(list))
    for raw_query in raw_info:
        query = Query.deserialize(raw_query)
        queries[query.formula.query_type][query.formula].append(query)
    return queries


def load_queries_by_type(data_file, keep_graph=True):
    raw_info = pickle.load(open(data_file, "rb"))
    queries = defaultdict(list)
    for raw_query in raw_info:
        query = Query.deserialize(raw_query, keep_graph=keep_graph)
        queries[query.formula.query_type].append(query)
    return queries


def load_test_queries_by_formula(data_file):
    raw_info = pickle.load(open(data_file, "rb"))
    queries = {"full_neg" : defaultdict(lambda : defaultdict(list)), 
            "one_neg" : defaultdict(lambda : defaultdict(list))}
    for raw_query in raw_info:
        neg_type = "full_neg" if len(raw_query[1]) > 1 else "one_neg"
        query = Query.deserialize(raw_query)
        queries[neg_type][query.formula.query_type][query.formula].append(query)
    return queries


def sample_clean_test(graph_loader, data_dir):
    num_val = 1_000
    num_test = 10_000

    train_graph = graph_loader()
    test_graph = graph_loader()
    test_edges = load_queries(data_dir + "/test_edges.pkl")
    val_edges = load_queries(data_dir + "/val_edges.pkl")
    train_graph.remove_edges([(q.target_node, q.formula.rels[0], q.anchor_nodes[0]) for q in test_edges+val_edges])
    
    print('Sampling test 2-queries')
    test_queries_2 = test_graph.sample_test_queries(train_graph, ["2-chain", "2-inter"], 0.9 * num_test, 1)
    test_queries_2.extend(test_graph.sample_test_queries(train_graph, ["2-chain", "2-inter"], 0.1 * num_test, 1000))

    print('Sampling val 2-queries')
    val_queries_2 = test_graph.sample_test_queries(train_graph, ["2-chain", "2-inter"], 0.9 * num_val, 1)
    val_queries_2.extend(test_graph.sample_test_queries(train_graph, ["2-chain", "2-inter"], 0.1 * num_val, 1000))

    val_queries_2 = list(set(val_queries_2)-set(test_queries_2))
    print(len(val_queries_2))
    
    print('Sampling test 3-queries')
    test_queries_3 = test_graph.sample_test_queries(train_graph, ["3-chain", "3-inter", "3-inter_chain", "3-chain_inter"], 0.9 * num_test, 1)
    test_queries_3.extend(test_graph.sample_test_queries(train_graph, ["3-chain", "3-inter", "3-inter_chain", "3-chain_inter"], 0.1 * num_test, 1000))

    print('Sampling val 3-queries')
    val_queries_3 = test_graph.sample_test_queries(train_graph, ["3-chain", "3-inter", "3-inter_chain", "3-chain_inter"], 0.9 * num_val, 1)
    val_queries_3.extend(test_graph.sample_test_queries(train_graph, ["3-chain", "3-inter", "3-inter_chain", "3-chain_inter"], 0.1 * num_val, 1000))

    val_queries_3 = list(set(val_queries_3)-set(test_queries_3))
    print(len(val_queries_3))

    pickle.dump([q.serialize() for q in test_queries_2], open(data_dir + "/test_queries_2.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([q.serialize() for q in test_queries_3], open(data_dir + "/test_queries_3.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([q.serialize() for q in val_queries_2], open(data_dir + "/val_queries_2.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([q.serialize() for q in val_queries_3], open(data_dir + "/val_queries_3.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

        
def parallel_sample_worker(pid, num_samples, graph, data_dir, is_test, test_edges):
    if not is_test:
        graph.remove_edges([(q.target_node, q.formula.rels[0], q.anchor_nodes[0]) for q in test_edges])
    print("Running worker", pid)
    queries_2 = graph.sample_queries(2, num_samples, 100 if is_test else 1, verbose=True)
    queries_3 = graph.sample_queries(3, num_samples, 100 if is_test else 1, verbose=True)
    print("Done running worker, now saving data", pid)
    pickle.dump([q.serialize() for q in queries_2], open(data_dir + "/queries_2-{:d}.pkl".format(pid), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([q.serialize() for q in queries_3], open(data_dir + "/queries_3-{:d}.pkl".format(pid), "wb"), protocol=pickle.HIGHEST_PROTOCOL)


def parallel_sample(graph, num_workers, samples_per_worker, data_dir, test=False, start_ind=None):
    if not test:
        print("Loading test/val data..")
        test_edges = load_queries(data_dir + "/test_edges.pkl")
        val_edges = load_queries(data_dir + "/val_edges.pkl")
    else:
        test_edges = []
        val_edges = []
    proc_range = list(range(num_workers)) if start_ind is None else list(range(start_ind, num_workers+start_ind))
    procs = [Process(target=parallel_sample_worker, args=[i, samples_per_worker, graph, data_dir, test, val_edges+test_edges]) for i in proc_range]
    for p in procs:
        p.start()
    for p in procs:
        p.join() 
    queries_2 = []
    queries_3 = []
    for i in range(num_workers):
        queries_2_file = osp.join(data_dir, "queries_2-{:d}.pkl".format(i))
        new_queries_2 = load_queries(queries_2_file, keep_graph=True)
        os.remove(queries_2_file)
        queries_2.extend(new_queries_2)

        queries_3_file = osp.join(data_dir, "queries_3-{:d}.pkl".format(i))
        new_queries_3 = load_queries(queries_3_file, keep_graph=True)
        os.remove(queries_3_file)
        queries_3.extend(new_queries_3)

    return queries_2, queries_3


class QueryDataset(Dataset):
    """A dataset for queries of a specific type, e.g. 1-chain.
    The dataset contains queries for formulas of different types, e.g.
    200 queries of type (('protein', '0', 'protein')),
    500 queries of type (('protein', '0', 'function')).
    (note that these queries are of type 1-chain).

    Args:
        queries (dict): maps formulas (graph.Formula) to query instances
            (list of graph.Query?)
    """
    def __init__(self, queries, *args, **kwargs):
        self.queries = queries
        self.num_formula_queries = OrderedDict()
        for form, form_queries in queries.items():
            self.num_formula_queries[form] = len(form_queries)
        self.num_queries = sum(self.num_formula_queries.values())
        self.max_num_queries = max(self.num_formula_queries.values())

    def __len__(self):
        return self.max_num_queries

    def __getitem__(self, index):
        return index

    def collate_fn(self, idx_list):
        # Select a formula type (e.g. ('protein', '0', 'protein'))
        # with probability proportional to the number of queries of that
        # formula type
        counts = np.array(list(self.num_formula_queries.values()))
        probs = counts / float(self.num_queries)
        formula_index = np.argmax(np.random.multinomial(1, probs))
        formula = list(self.num_formula_queries.keys())[formula_index]

        n = self.num_formula_queries[formula]
        # Assume sorted idx_list
        min_idx, max_idx = idx_list[0], idx_list[-1]

        start = min_idx % n
        end = min((max_idx + 1) % n, n)
        end = n if end <= start else end
        queries = self.queries[formula][start:end]

        return formula, queries


class RGCNQueryDataset(QueryDataset):
    """A dataset for queries of a specific type, e.g. 1-chain.
    The dataset contains queries for formulas of different types, e.g.
    200 queries of type (('protein', '0', 'protein')),
    500 queries of type (('protein', '0', 'function')).
    (note that these queries are of type 1-chain).

    Args:
        queries (dict): maps formulas (graph.Formula) to query instances
            (list of graph.Query?)
    """
    query_edge_indices = {'1-chain': [[0],
                                      [1]],
                          '2-chain': [[0, 2],
                                      [2, 1]],
                          '3-chain': [[0, 3, 2],
                                      [3, 2, 1]],
                          '2-inter': [[0, 1],
                                      [2, 2]],
                          '3-inter': [[0, 1, 2],
                                      [3, 3, 3]],
                          '3-inter_chain': [[0, 1, 3],
                                            [2, 3, 2]],
                          '3-chain_inter': [[0, 1, 3],
                                            [3, 3, 2]]}

    query_diameters = {'1-chain': 1,
                       '2-chain': 2,
                       '3-chain': 3,
                       '2-inter': 1,
                       '3-inter': 1,
                       '3-inter_chain': 2,
                       '3-chain_inter': 2}

    query_edge_label_idx = {'1-chain': [0],
                            '2-chain': [1, 0],
                            '3-chain': [2, 1, 0],
                            '2-inter': [0, 1],
                            '3-inter': [0, 1, 2],
                            '3-inter_chain': [0, 2, 1],
                            '3-chain_inter': [1, 2, 0]}

    variable_node_idx = {'1-chain': [0],
                         '2-chain': [0, 2],
                         '3-chain': [0, 2, 4],
                         '2-inter': [0],
                         '3-inter': [0],
                         '3-chain_inter': [0, 2],
                         '3-inter_chain': [0, 3]}

    def __init__(self, queries, enc_dec):
        super(RGCNQueryDataset, self).__init__(queries)
        self.mode_ids = enc_dec.mode_ids
        self.rel_ids = enc_dec.rel_ids

    def collate_fn(self, idx_list):
        formula, queries = super(RGCNQueryDataset, self).collate_fn(idx_list)
        graph_data = RGCNQueryDataset.get_query_graph(formula, queries,
                                                      self.rel_ids,
                                                      self.mode_ids)
        anchor_ids, var_ids, graph = graph_data
        return formula, queries, anchor_ids, var_ids, graph

    @staticmethod
    def get_query_graph(formula, queries, rel_ids, mode_ids):
        batch_size = len(queries)
        n_anchors = len(formula.anchor_modes)

        anchor_ids = np.zeros([batch_size, n_anchors]).astype(np.int64)
        # First rows of x contain embeddings of all anchor nodes
        for i, anchor_mode in enumerate(formula.anchor_modes):
            anchors = [q.anchor_nodes[i] for q in queries]
            anchor_ids[:, i] = anchors

        # The rest of the rows contain generic mode embeddings for variables
        all_nodes = formula.get_nodes()
        var_idx = RGCNQueryDataset.variable_node_idx[formula.query_type]
        var_ids = np.array([mode_ids[all_nodes[i]] for i in var_idx],
                            dtype=np.int64)

        edge_index = RGCNQueryDataset.query_edge_indices[formula.query_type]
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        rels = formula.get_rels()
        rel_idx = RGCNQueryDataset.query_edge_label_idx[formula.query_type]
        edge_type = [rel_ids[_reverse_relation(rels[i])] for i in rel_idx]
        edge_type = torch.tensor(edge_type, dtype=torch.long)

        edge_data = Data(edge_index=edge_index)
        edge_data.edge_type = edge_type
        edge_data.num_nodes = n_anchors + len(var_idx)
        graph = Batch.from_data_list([edge_data for i in range(batch_size)])

        return (torch.tensor(anchor_ids, dtype=torch.long),
                torch.tensor(var_ids, dtype=torch.long),
                graph)


def make_data_iterator(data_loader):
    iterator = iter(data_loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(data_loader)
            continue


def get_queries_iterator(queries, batch_size, enc_dec=None):
    dataset = RGCNQueryDataset(queries, enc_dec)
    loader = DataLoader(dataset, batch_size, shuffle=False,
                        collate_fn=dataset.collate_fn)
    return make_data_iterator(loader)


if __name__ == '__main__':
    queries = {('protein','0','protein'): ['a' + str(i) for i in range(10)],
               ('protein', '0', 'function'): ['b' + str(i) for i in range(20)],
               ('function', '0', 'function'): ['c' + str(i) for i in range(30)]}

    iterator = get_queries_iterator(queries, batch_size=4)

    for i in range(50):
        batch = next(iterator)
        print(batch)
