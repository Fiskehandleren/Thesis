import torch.optim
from torch_geometric_temporal.nn.recurrent import TGCN2
from architectures.GraphTemporalBase import GraphTemporalBaseClass

class TGCN(GraphTemporalBaseClass):
    def __init__(self, **args):
        super().__init__(**args)
        # We add improved self-loops for each node, to make sure that the nodes are weighing themselves
        # more than their neighbors. `improved=True` means that A_hat = A + 2I, so the diagonal is 3.
        self.tgcn_cell = TGCN2(self.node_features, self.hidden_dim, add_self_loops=True, improved=not self.no_self_loops, batch_size=self.batch_size)
        self.linear = torch.nn.Linear(self.hidden_dim, self.forecast_horizon)


    def forward(self, x, edge_index, edge_weight):
        # X is shape (Batch Size, Nodes, Features, Sequence Length)
        h = None # Maybe initialize randomly?
        # Go over each 
        for i in range(self.sequence_length):
            # Each X_t is of shape (Batch Size, Nodes, Features)
            h = self.tgcn_cell(x[:,:,:,i], edge_index, edge_weight, h)

        y = self.linear(h)
        return y.exp(), h
