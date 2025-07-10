import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import lightgbm as lgb

# Generate challenging synthetic time series data
np.random.seed(42)
T = 100
time = np.arange(T)

# Regime flips every 20 steps
regime = ((time // 20) % 2).astype(float)

# Time-varying chaotic pulse
pulse = np.sin(0.2 * time) * np.cos(0.05 * time ** 1.5)

# Piecewise trend jump
trend = np.piecewise(time, [time < 30, (time >= 30) & (time < 60), time >= 60],
                     [lambda t: 0.1 * t,
                      lambda t: -0.05 * (t - 30) + 3,
                      lambda t: 0.08 * (t - 60)])

# Time-varying frequency seasonality
seasonality = 3 * np.sin(2 * np.pi * time / (6 + regime * 6))

# Noise and shock bursts
noise = np.random.normal(0, 0.8, T)
shock = (np.random.rand(T) < 0.05) * np.random.normal(0, 8, T)

# Combined signal
y = 10 + trend + seasonality + pulse + regime * 0.2 * time + noise + shock



# Train-test split
train_size = 80
y_train, y_test = y[:train_size], y[train_size:]
time_train, time_test = time[:train_size], time[train_size:]

# 1. ARIMA
arima_model = ARIMA(y_train, order=(3,1,2)).fit()
arima_forecast = arima_model.forecast(steps=len(y_test))

# 2. LightGBM with lag features
df = pd.DataFrame({'y': y, 'time': time, 'month': time % 12})
for lag in [1, 2, 3, 12]:
    df[f'lag_{lag}'] = df['y'].shift(lag)
df.dropna(inplace=True)
X = df.drop(columns='y')
y_lgb = df['y']
X_train, X_test = X.iloc[:train_size - 12], X.iloc[train_size - 12:]
y_train_lgb, y_test_lgb = y_lgb.iloc[:train_size - 12], y_lgb.iloc[train_size - 12:]
lgb_model = lgb.LGBMRegressor()
lgb_model.fit(X_train, y_train_lgb)
lgb_forecast = lgb_model.predict(X_test)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(np.arange(train_size), y_train, label='Train', color='black')
plt.plot(np.arange(train_size, T), y_test, label='Test', color='blue')
plt.plot(np.arange(train_size, T), arima_forecast, label='ARIMA Forecast', linestyle='--', color='red')
plt.plot(np.arange(train_size, T), lgb_forecast, label='LightGBM Forecast', linestyle='--', color='green')
plt.title("Forecasting Comparison: Train, Test, ARIMA, LightGBM")
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

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

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# ---- Your synthetic time series ----
# y: shape [T]
seq_len = 30
forecast_horizon = 1
B = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Preprocess
def build_sequences(y, seq_len):
    X, Y = [], []
    for i in range(len(y) - seq_len - forecast_horizon + 1):
        X.append(y[i:i + seq_len])
        Y.append(y[i + seq_len: i + seq_len + forecast_horizon])
    X = torch.tensor(np.stack(X), dtype=torch.float32).unsqueeze(2)
    Y = torch.tensor(np.stack(Y), dtype=torch.float32)
    return X, Y

X, Y = build_sequences(y[:80], seq_len)
test_X, test_Y = build_sequences(y[80 - seq_len:], seq_len)

train_ds = TensorDataset(X, Y)
train_loader = DataLoader(train_ds, batch_size=B, shuffle=True)

# ---- Model ----
model = GraphTemporalForecaster(input_dim=1, hidden_dim=64, seq_len=30, embed_dim=8).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# ---- Training ----
for epoch in range(500):
    model.train()
    losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred, _ = model(xb)
        loss = loss_fn(pred, yb.squeeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(f"Epoch {epoch}: {np.mean(losses):.4f}")

# ---- Forecast ----
model.eval()
with torch.no_grad():
    test_X = test_X.to(device)
    preds, _ = model(test_X)
    preds = preds.cpu().numpy().squeeze()
import matplotlib.pyplot as plt

test_idx = np.arange(80, 100)

plt.figure(figsize=(12, 6))
plt.plot(np.arange(80), y[:80], label='Train', color='black')
plt.plot(test_idx, y[80:], label='Test', color='blue')
plt.plot(test_idx, arima_forecast, label='ARIMA', linestyle='--', color='red')
plt.plot(test_idx, lgb_forecast, label='LightGBM', linestyle='--', color='green')
plt.plot(test_idx, preds, label='GraphTemporalForecaster', linestyle='--', color='purple')
plt.title("Forecast Comparison: ARIMA, LightGBM, GraphTemporalForecaster")
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()