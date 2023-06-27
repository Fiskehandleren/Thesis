import torch.optim
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from architectures.GraphTemporalBase import GraphTemporalBaseClass


class TGCN2(torch.nn.Module):
    r"""An implementation THAT SUPPORTS BATCHES of the Temporal Graph Convolutional Gated Recurrent Cell.
    For details see this paper: `"T-GCN: A Temporal Graph ConvolutionalNetwork for
    Traffic Prediction." <https://arxiv.org/abs/1811.05320>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        batch_size (int): Size of the batch.
        improved (bool): Stronger self loops. Default is False.
        cached (bool): Caching the message weights. Default is False.
        add_self_loops (bool): Adding self-loops for smoothing. Default is True.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        batch_size: int,  # this entry is unnecessary, kept only for backward compatibility
        edge_weight: torch.FloatTensor,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True,
        train_edge_weight: bool = False,
    ):
        super(TGCN2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.batch_size = batch_size  # not needed
        self._create_parameters_and_layers()

        if train_edge_weight:
            self.edge_weight = torch.nn.Parameter(edge_weight)
        else:
            self.edge_weight = torch.nn.Parameter(edge_weight, requires_grad=False)

    def _create_update_gate_parameters_and_layers(self):
        self.conv_z = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.in_channels * 2,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )
        self.conv_z_2 = GCNConv(
            in_channels=self.in_channels * 2,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )
        self.linear_z = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_reset_gate_parameters_and_layers(self):
        self.conv_r = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.in_channels * 2,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )
        self.conv_r_2 = GCNConv(
            in_channels=self.in_channels * 2,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )
        self.linear_r = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_candidate_state_parameters_and_layers(self):
        self.conv_h = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.in_channels * 2,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )
        self.conv_h_2 = GCNConv(
            in_channels=self.in_channels * 2,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )
        self.linear_h = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            # can infer batch_size from X.shape, because X is [B, N, F]
            H = torch.zeros(X.shape[0], X.shape[1], self.out_channels).to(X.device)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H):
        f_a_x = self.conv_z_2(
            self.conv_z(X, edge_index, edge_weight).relu(), edge_index, edge_weight
        ).sigmoid()
        Z = torch.cat([f_a_x, H], axis=2)
        Z = self.linear_z(Z)  # (b, 207, 32)
        Z = torch.sigmoid(Z)

        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_weight, H):
        f_a_x = self.conv_r_2(
            self.conv_r(X, edge_index, edge_weight).relu(), edge_index, edge_weight
        ).sigmoid()
        R = torch.cat([f_a_x, H], axis=2)
        R = self.linear_r(R)  # (batch, nodes, outputs)
        R = torch.sigmoid(R)

        return R

    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R):
        f_a_x = self.conv_h_2(
            self.conv_h(X, edge_index, edge_weight).relu(), edge_index, edge_weight
        ).sigmoid()
        H_tilde = torch.cat([f_a_x, H * R], axis=2)
        H_tilde = self.linear_h(H_tilde)
        H_tilde = torch.tanh(H_tilde)

        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        # edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.

        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, edge_index, self.edge_weight, H)
        R = self._calculate_reset_gate(X, edge_index, self.edge_weight, H)
        H_tilde = self._calculate_candidate_state(X, edge_index, self.edge_weight, H, R)
        H = self._calculate_hidden_state(Z, H, H_tilde)  # (b, 207, 32)
        return H


class TGCN(GraphTemporalBaseClass):
    def __init__(self, **args):
        super().__init__(**args)
        # We add improved self-loops for each node, to make sure that the nodes are weighing themselves
        # more than their neighbors. `improved=True` means that A_hat = A + 2I, so the diagonal is 3.
        self.tgcn_cell = TGCN2(
            self.node_features,
            self.hidden_dim,
            edge_weight=self.edge_weight,
            add_self_loops=False,  # We already do this in the dataloader
            improved=not self.no_self_loops,
            batch_size=self.batch_size,
            train_edge_weight=self.train_edge_weight,
        )
        self.dropout = torch.nn.Dropout(p=0.2)
        self.linear = torch.nn.Linear(self.hidden_dim, self.forecast_horizon)

    def forward(self, x, edge_index):
        # X is shape (Batch Size, Nodes, Features, Sequence Length)
        h = None  # Maybe initialize randomly?
        # Go over each
        for i in range(self.sequence_length):
            # Each X_t is of shape (Batch Size, Nodes, Features)
            h = self.tgcn_cell(x[:, :, :, i], edge_index, h)
        if self.use_dropout:
            h = self.dropout(h)
        y = self.linear(h)
        return y.exp(), h
