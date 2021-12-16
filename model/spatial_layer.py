from typing import List

import dgl
import torch
from dgl import init as g_init
from dgl.nn.pytorch import GraphConv
from torch import nn


class GCN(nn.Module):
    def __init__(self, in_features: int, hidden_sizes: List[int], out_features: int):
        super(GCN, self).__init__()
        gcns, relus, bns = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for idx, hidden_size in enumerate(hidden_sizes):
            if idx == 0:
                gcns.append(GraphConv(in_features, hidden_size))
                relus.append(nn.ReLU())
                bns.append(nn.BatchNorm1d(hidden_size))
            else:
                gcns.append(GraphConv(hidden_sizes[idx - 1], hidden_size))
                relus.append(nn.ReLU())
                bns.append(nn.BatchNorm1d(hidden_size))
        relus.append(nn.ReLU())
        bns.append(nn.BatchNorm1d(out_features))
        gcns.append(GraphConv(hidden_sizes[-1], out_features))
        self.gcns, self.relus, self.bns = gcns, relus, bns

    def forward(self, g: dgl.DGLGraph, node_features: torch.Tensor):
        """
        :param g: a graph
        :param node_features: shape (n_nodes, n_features)
        :return:
        """
        g.set_n_initializer(g_init.zero_initializer)
        g.set_e_initializer(g_init.zero_initializer)
        h = node_features
        for gcn, relu, bn in zip(self.gcns, self.relus, self.bns):
            h = gcn(g, h)
            if len(h.shape) > 2:
                h = bn(h.transpose(1, -1)).transpose(1, -1)
            else:
                h = bn(h)
            h = relu(h)
        return h


class StackedSBlocks(nn.ModuleList):
    def __init__(self, *args, **kwargs):
        super(StackedSBlocks, self).__init__(*args, **kwargs)

    def forward(self, *input):
        g, h = input
        for module in self[:-1]:
            h = h + module(g, h)
        h = self[-1](g, h)
        return h
