import torch
import dgl
from torch import nn

from model.fully_connected import fully_connected_layer
from model.spatial_layer import GCN, StackedSBlocks

from utils.load_config import get_attribute


class STBlock(nn.Module):
    def __init__(self, f_in: int, f_out: int):
        """
        :param t_in: the number of time steps before
        :param f_in: the number of dynamic features each node before
        :param f_out: the number of dynamic features each node after
        """
        super(STBlock, self).__init__()
        # stack four middle layers to reduce features from f_in to f_out
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


class DSTGCN(nn.Module):
    def __init__(self, f_1: int, f_2: int, f_3: int):
        """
        :param f_1: the number of static features each node, default 22
        :param f_2: the number of dynamic features each node, default 1
        :param f_3: the number of features overall
        """
        super(DSTGCN, self).__init__()

        if get_attribute("use_SBlock"):
            self.spatial_embedding = fully_connected_layer(f_1, [20], 15)
            self.spatial_gcn = StackedSBlocks([GCN(15, [15, 15, 15], 15),
                                               GCN(15, [15, 15, 15], 15),
                                               GCN(15, [14, 13, 12, 11], 10)])
        else:
            # replace spatial layer
            self.replace_spatial_gcn = fully_connected_layer(f_1, [20, 15], 10)

        if get_attribute("use_STBlock"):
            self.temporal_embedding = StackedSTBlocks([STBlock(f_2, 4), STBlock(5, 5), STBlock(10, 10)])
        else:
            # replace spatial-temporal layer
            self.replace_temporal_layer = fully_connected_layer(f_2, [5, 10], 20)

        self.temporal_agg = nn.AvgPool1d(24)

        self.external_embedding = fully_connected_layer(f_3, [(f_3 * (4 - i) + 10 * i) // 4 for i in (1, 4)], 10)

        if get_attribute("use_Embedding"):
            self.output_layer = nn.Sequential(nn.ReLU(),
                                              nn.Linear(40, 1),
                                              nn.Sigmoid())
        else:
            self.output_layer = nn.Sequential(nn.ReLU(),
                                              nn.Linear(73, 1),
                                              nn.Sigmoid())

    def forward(self,
                bg: dgl.DGLGraph,
                spatial_features: torch.Tensor,
                temporal_features: torch.Tensor,
                external_features: torch.Tensor):
        """
        get predictions
        :param bg: batched graphs,
             with the total number of nodes is `node_num`,
             including `batch_size` disconnected subgraphs
        :param spatial_features: shape [node_num, F_1]
        :param temporal_features: shape [node_num, F_2, T]
        :param external_features: shape [batch_size, F_3]
        :return: a tensor, shape [batch_size], with the prediction results for each graphs
        """

        if get_attribute("use_SBlock"):
            s_out = self.spatial_gcn(bg, self.spatial_embedding(spatial_features))

        else:
            # remove spatial layer
            s_out = self.replace_spatial_gcn(spatial_features)

        if get_attribute("use_STBlock"):
            # temporal_embeddings of shape [node_num, 20, T_in]
            temporal_embeddings = self.temporal_embedding(bg, temporal_features)
        else:
            # remove temporal layer
            temporal_embeddings = torch.transpose(
                self.replace_temporal_layer(torch.transpose(temporal_features, -1, -2)), -1, -2)

        # t_out of shape [1, node_num, 10]
        # _, (t_out, _) = self.temporal_agg(torch.transpose(temporal_embeddings, -1, -2))
        t_out = self.temporal_agg(temporal_embeddings)
        t_out.squeeze_()

        if get_attribute("use_Embedding"):
            e_out = self.external_embedding(external_features)
        else:
            # remove external embedding layer
            e_out = external_features

        nums_nodes, id = bg.batch_num_nodes, 0
        s_features, t_features = list(), list()
        for num_nodes in nums_nodes:
            s_features.append(s_out[id])
            t_features.append(t_out[id])
            id += num_nodes

        s_features = torch.stack(s_features)
        t_features = torch.stack(t_features)

        return self.output_layer(torch.cat((s_features, t_features, e_out), -1))
