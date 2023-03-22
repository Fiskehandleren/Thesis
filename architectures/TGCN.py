import torch
from torch_geometric_temporal.nn.recurrent import TGCN2
import torch.nn.functional as F
import argparse

class TemporalGCN(torch.nn.Module):
    def __init__(self, node_features, hidden_dim, time_steps, batch_size=32):
        super(TemporalGCN, self).__init__()
        self.time_steps = time_steps
        self._hidden_dim = hidden_dim
        # We add improved self-loops for each node, to make sure that the nodes are weighing themselves
        # more than their neighbors. `improved=True` means that A_hat = A + 2I, so the diagonal is 3.
        self.recurrent = TGCN2(node_features, self._hidden_dim, add_self_loops=True, improved=True, batch_size=batch_size)
        self.linear = torch.nn.Linear(hidden_dim, 1)


    def forward(self, x, edge_index, edge_weight):
        h = None # Maybe initialize randomly?
        # Go over each 
        for i in range(self.time_steps):
            # Each X_t is of shape (Batch Size, Nodes, Features)
            h = self.recurrent(x[:,:,:,i], edge_index, edge_weight, h)
        #h = self.recurrent(x, edge_index, edge_weight)
        y = F.relu(h)
        y = self.linear(h)
        return y.exp(), h
    
    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=64)
        return parser
    
    #@property
    #def hyperparameters(self):
    #    return {"hidden_dim": self._hidden_dim}