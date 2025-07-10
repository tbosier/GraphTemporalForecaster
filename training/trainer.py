import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yfinance as yf

def build_sequences(y, seq_len, forecast_horizon=1):
    """Build sequences for time series forecasting."""
    X, Y = [], []
    for i in range(len(y) - seq_len - forecast_horizon + 1):
        X.append(y[i:i + seq_len])
        Y.append(y[i + seq_len: i + seq_len + forecast_horizon])
    X = torch.tensor(np.stack(X), dtype=torch.float32).unsqueeze(2)
    Y = torch.tensor(np.stack(Y), dtype=torch.float32)
    return X, Y

class GraphTemporalTrainer:
    def __init__(self, model, device='auto', lr=1e-3, batch_size=32):
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.batch_size = batch_size
        self.train_losses = []
        self.val_losses = []
        
    def prepare_data(self, y, seq_len=30, train_ratio=0.8):
        """Prepare training and validation data."""
        X, Y = build_sequences(y, seq_len)
        
        # Split data
        train_size = int(len(X) * train_ratio)
        X_train, X_val = X[:train_size], X[train_size:]
        Y_train, Y_val = Y[:train_size], Y[train_size:]
        
        # Create data loaders
        train_ds = TensorDataset(X_train, Y_train)
        val_ds = TensorDataset(X_val, Y_val)
        
        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        
        return X_train, Y_train, X_val, Y_val
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        losses = []
        
        for xb, yb in self.train_loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            
            self.optimizer.zero_grad()
            pred, _ = self.model(xb)
            loss = self.loss_fn(pred, yb.squeeze(1))
            loss.backward()
            self.optimizer.step()
            
            losses.append(loss.item())
        
        return np.mean(losses)
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        losses = []
        
        with torch.no_grad():
            for xb, yb in self.val_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred, _ = self.model(xb)
                loss = self.loss_fn(pred, yb.squeeze(1))
                losses.append(loss.item())
        
        return np.mean(losses)
    
    def train(self, epochs=500, verbose=True):
        """Train the model for specified number of epochs."""
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            if verbose and epoch % 50 == 0:
                print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    def forecast(self, X_test):
        """Generate forecasts."""
        self.model.eval()
        with torch.no_grad():
            X_test = X_test.to(self.device)
            preds, adj = self.model(X_test)
            return preds.cpu().numpy().squeeze(), adj.cpu().numpy()
    
    def evaluate(self, y_true, y_pred):
        """Evaluate forecast performance."""
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse
        }
    
    def plot_training(self):
        """Plot training and validation losses."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training and Validation Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

def get_stock_data(symbol, start_date, end_date):
    """Download stock data using yfinance."""
    stock = yf.Ticker(symbol)
    data = stock.history(start=start_date, end=end_date)
    return data['Close'].values

def simulate_trading(initial_capital, stock_prices, predictions, buy_threshold=0.02, sell_threshold=-0.02):
    """
    Simulate trading based on predictions.
    
    Parameters:
    -----------
    initial_capital: Starting cash
    stock_prices: Actual stock prices for test period
    predictions: Model predictions for next day prices
    buy_threshold: Minimum expected return to buy (e.g., 0.02 = 2%)
    sell_threshold: Maximum expected return to sell (e.g., -0.02 = -2%)
    
    Returns:
    --------
    dict: Trading results
    """
    capital = initial_capital
    shares = 0
    trades = []
    
    # We need predictions for each day, but we only have predictions for next-day prices
    # So we need to align predictions with current prices
    for i in range(len(stock_prices)):
        current_price = stock_prices[i]
        
        # If we have a prediction for tomorrow
        if i < len(predictions):
            predicted_next_price = predictions[i]
            expected_return = (predicted_next_price - current_price) / current_price
            
            # Trading logic
            if expected_return > buy_threshold and capital > 0:
                # Buy: Spend all cash
                shares_to_buy = capital / current_price
                shares += shares_to_buy
                capital = 0
                trades.append(('BUY', current_price, shares_to_buy, expected_return))
                print(f"Day {i}: BUY {shares_to_buy:.2f} shares at ${current_price:.2f} (Expected return: {expected_return:.2%})")
                
            elif expected_return < sell_threshold and shares > 0:
                # Sell: Sell all shares
                capital += shares * current_price
                trades.append(('SELL', current_price, shares, expected_return))
                print(f"Day {i}: SELL {shares:.2f} shares at ${current_price:.2f} (Expected return: {expected_return:.2%})")
                shares = 0
    
    # Final portfolio value
    final_value = capital + (shares * stock_prices[-1])
    total_return = (final_value - initial_capital) / initial_capital
    
    return {
        'final_value': final_value,
        'total_return': total_return,
        'trades': trades,
        'final_shares': shares,
        'final_cash': capital,
        'num_trades': len(trades)
    } 