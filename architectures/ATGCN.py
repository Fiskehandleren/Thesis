import argparse
from torch import nn
from torch_geometric_temporal.nn.recurrent import A3TGCN2
from architectures.GraphTemporalBase import GraphTemporalBaseClass


class ATGCN(GraphTemporalBaseClass):
    def __init__(self, **args):
        super().__init__(**args)
        # We add improved self-loops for each node, to make sure that the nodes are weighing themselves
        # more than their neighbors. `improved=True` means that A_hat = A + 2I, so the diagonal is 3.
        self.atgcn = A3TGCN2(in_channels=self.node_features, 
                             out_channels=self.hidden_dim,
                             periods=self.sequence_length,
                             add_self_loops=not self.no_self_loops,
                             improved=True,
                             batch_size=self.batch_size)
        self.linear = nn.Linear(self.hidden_dim, self.forecast_horizon)

    def forward(self, x, edge_index, edge_weight):
        h = self.atgcn(x, edge_index, edge_weight)
        y = self.linear(h)
        return y.exp(), h
