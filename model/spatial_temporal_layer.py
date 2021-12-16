import torch
import dgl
from torch import nn

from model.spatial_layer import GCN


class STBlock(nn.Module):
    def __init__(self, f_in: int, f_out: int):
        """
        :param f_in: the number of dynamic features each node before
        :param f_out: the number of dynamic features each node after
        """
        super(STBlock, self).__init__()
        # stack four middle layers to transform features from f_in to f_out
        self.spatial_embedding = GCN(f_in, [(f_in * (4 - i) + f_out * i) // 4 for i in (1, 4)], f_out)
        self.temporal_embedding = nn.Conv1d(f_out, f_out, 3, padding=1)

    def forward(self, g: dgl.DGLGraph, temporal_features: torch.Tensor):
        """
        :param g: batched graphs,
             with the total number of nodes is `node_num`,
             including `batch_size` disconnected subgraphs
        :param temporal_features: shape [node_num, f_in, t_in]
        :return: hidden features after temporal and spatial embedding, shape [node_num, f_out, t_in]
        """
        return self.temporal_embedding(self.spatial_embedding(g, temporal_features.transpose(-2, -1)).transpose(-2, -1))


class StackedSTBlocks(nn.ModuleList):
    def __init__(self, *args, **kwargs):
        super(StackedSTBlocks, self).__init__(*args, **kwargs)

    def forward(self, *input):
        g, h = input
        for module in self:
            h = torch.cat((h, module(g, h)), dim=1)

        return h
