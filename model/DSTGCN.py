import torch
import dgl
from torch import nn

from model.fully_connected import fully_connected_layer
from model.spatial_layer import GCN, StackedSBlocks
from model.spatial_temporal_layer import STBlock, StackedSTBlocks

from utils.load_config import get_attribute


class DSTGCN(nn.Module):
    def __init__(self, f_1: int, f_2: int, f_3: int):
        """
        :param f_1: the number of spatial features each node, default 22
        :param f_2: the number of dynamic features each node, default 1
        :param f_3: the number of features overall (external)
        """
        super(DSTGCN, self).__init__()

        self.spatial_embedding = fully_connected_layer(f_1, [20], 15)
        self.spatial_gcn = StackedSBlocks([GCN(15, [15, 15, 15], 15),
                                           GCN(15, [15, 15, 15], 15),
                                           GCN(15, [14, 13, 12, 11], 10)])
        self.temporal_embedding = StackedSTBlocks([STBlock(f_2, 4), STBlock(5, 5), STBlock(10, 10)])

        self.temporal_agg = nn.AvgPool1d(24)

        self.external_embedding = fully_connected_layer(f_3, [(f_3 * (4 - i) + 10 * i) // 4 for i in (1, 4)], 10)

        self.output_layer = nn.Sequential(nn.ReLU(),
                                          nn.Linear(10 + 20 + 10, 1),
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

        s_out = self.spatial_gcn(bg, self.spatial_embedding(spatial_features))

        temporal_embeddings = self.temporal_embedding(bg, temporal_features)

        # t_out of shape [1, node_num, 10]
        t_out = self.temporal_agg(temporal_embeddings)
        t_out.squeeze_()

        e_out = self.external_embedding(external_features)

        nums_nodes, id = bg.batch_num_nodes(), 0
        s_features, t_features = list(), list()
        for num_nodes in nums_nodes:
            s_features.append(s_out[id])
            t_features.append(t_out[id])
            id += num_nodes

        s_features = torch.stack(s_features)
        t_features = torch.stack(t_features)

        output_features = torch.cat((s_features, t_features, e_out), -1)

        return self.output_layer(output_features)
