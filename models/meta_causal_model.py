import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphLearner(nn.Module):
    def __init__(self, embed_dim, hidden_dim, seq_len):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=2, dim_feedforward=hidden_dim),
            num_layers=2
        )
        self.proj = nn.Sequential(
            nn.Linear(seq_len * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim * embed_dim)
        )
        self.embed_dim = embed_dim

    def forward(self, x):  # x: [B, T, embed_dim]
        x = self.encoder(x.permute(1, 0, 2)).permute(1, 0, 2)
        x_flat = x.reshape(x.shape[0], -1)
        adj_logits = self.proj(x_flat)
        return adj_logits.view(-1, self.embed_dim, self.embed_dim)


class GNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.message = nn.Linear(input_dim, hidden_dim)
        self.update = nn.GRUCell(hidden_dim, input_dim)

    def forward(self, x, adj):  # x: [B, D], adj: [B, D, D]
        msgs = torch.bmm(adj, x.unsqueeze(2)).squeeze(2)  # sum over neighbors
        msgs = F.relu(self.message(msgs))
        x_next = self.update(msgs, x)
        return x_next

class GraphTemporalForecaster(nn.Module):
    """
    Graph-based temporal forecasting model that learns temporal dependencies
    through learned graph structures.
    
    This model combines:
    1. GraphLearner: Learns graph structure from time series
    2. GNNCell: Processes temporal dependencies via graph neural network
    3. Forecasting: Predicts future values based on learned patterns
    """
    def __init__(self, input_dim=1, hidden_dim=64, seq_len=30, embed_dim=8):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.graph_learner = GraphLearner(embed_dim, hidden_dim, seq_len)
        self.gnn_cell = GNNCell(embed_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_seq):  # x_seq: [B, T, D]
        x_embed = self.embedding(x_seq)        # Embed scalar to [B, T, embed_dim]
        x_last = x_embed[:, -1, :]             # Last step for GNN [B, embed_dim]
        adj_logits = self.graph_learner(x_embed)  # Pass embedded sequence to GraphLearner
        adj = torch.sigmoid(adj_logits)
        adj = adj * (1 - torch.eye(adj.shape[1], device=adj.device).unsqueeze(0))  # Remove self-loop
        x_next = self.gnn_cell(x_last, adj)
        y_pred = self.decoder(x_next)
        return y_pred.squeeze(1), adj

# Keep old name for backward compatibility
MetaCausalForecaster = GraphTemporalForecaster 