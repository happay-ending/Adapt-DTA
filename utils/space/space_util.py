#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""==============================================
# @Project : ENAS-pytorch-master
# @File    : space_util.py
# @IDE     : PyCharm
# @Author  : Austin
# @Time    : 2023/6/23 19:38
================================================"""

import torch
import torch.nn.functional as F
import torch_geometric
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, ChebConv, SAGEConv, GatedGraphConv, ARMAConv, SGConv, SumAggregation, \
    MeanAggregation, MaxAggregation, AttentionalAggregation, Set2Set, global_add_pool, GINConv


def act_map(act):
    if act == "linear":
        return lambda x: x
    elif act == "elu":
        return F.elu
    elif act == "sigmoid":
        return F.sigmoid
    elif act == "tanh":
        return F.tanh
    elif act == "relu":
        return F.relu
    elif act == "relu6":
        return F.relu6
    elif act == "softplus":
        return F.softplus
    elif act == "leaky_relu":
        return F.leaky_relu
    else:
        raise Exception("wrong activate function")


class LinearConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(LinearConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = torch.nn.Linear(in_channels, out_channels, bias)

    def forward(self, x, edge_index, edge_weight=None):
        return self.linear(x)

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )


class ZeroConv(nn.Module):
    def forward(self, x, edge_index, edge_weight=None):
        out = torch.zeros_like(x)
        out.requires_grad = True
        return out

    def __repr__(self):
        return "ZeroConv()"


class Identity(nn.Module):
    def forward(self, x, edge_index, edge_weight=None):
        return x

    def __repr__(self):
        return "Identity()"


def gnn_map(gnn_aggr_name, in_dim, out_dim, concat=False, bias=True) -> nn.Module:
    """
    :param gnn_aggr_name:
    :param in_dim:
    :param out_dim:
    :param concat: for gat, concat multi-head output or not
    :return: GNN model
    """
    gnn_name, aggr_name = gnn_aggr_name.split("_")
    if gnn_name == "gat":
        return GATConv(in_dim, out_dim, aggr=aggr_name, concat=concat, bias=bias)
    elif gnn_name == "gcn":
        return GCNConv(in_dim, out_dim, aggr=aggr_name)
    elif gnn_name == "cheb":
        return ChebConv(in_dim, out_dim, aggr=aggr_name, K=2, bias=bias)
    elif gnn_name == "gin":
        return GINConv(nn=nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)), eps=0, aggr=aggr_name)
    elif gnn_name == "sage":
        return SAGEConv(in_dim, out_dim, aggr=aggr_name, bias=bias)
    elif gnn_name == "gated":
        return GatedGraphConv(in_dim, out_dim, aggr=aggr_name, bias=bias)
    elif gnn_name == "arma":
        return ARMAConv(in_dim, out_dim, aggr=aggr_name, bias=bias)
    elif gnn_name == "sg":
        return SGConv(in_dim, out_dim, aggr=aggr_name, bias=bias)
    elif gnn_name == "linear":
        return LinearConv(in_dim, out_dim, bias=bias)
    elif gnn_name == "zero":
        return ZeroConv()
    elif gnn_name == "identity":
        return Identity()
    elif hasattr(torch_geometric.nn, gnn_name):
        cls = getattr(torch_geometric.nn, gnn_name)
        assert isinstance(cls, type), "Only support modules, get %s" % (gnn_name)
        kwargs = {
            "in_channels": in_dim,
            "out_channels": out_dim,
            "concat": concat,
            "bias": bias,
        }
        kwargs = {
            key: kwargs[key]
            for key in cls.__init__.__code__.co_varnames
            if key in kwargs
        }
        return cls(**kwargs)
    raise KeyError("Cannot parse key %s" % (gnn_name))


def pooling_map(graph_pooling, emb_dim, processing_steps=2) -> nn.Module:
    # Pooling function to generate whole-graph embeddings
    if graph_pooling == "sum":
        # return global_add_pool
        return SumAggregation()
    elif graph_pooling == "mean":
        return MeanAggregation()
        # return global_mean_pool
    elif graph_pooling == "max":
        return MaxAggregation()
        # return global_max_pool
    elif graph_pooling == "attention":
        return AttentionalAggregation(gate_nn=nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(), nn.Linear(emb_dim, 1)))
    elif graph_pooling == "set2set":
        return Set2Set(emb_dim, processing_steps)
    else:
        raise ValueError("Invalid graph pooling type.")
