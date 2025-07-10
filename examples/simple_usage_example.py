#!/usr/bin/env python3
"""
Simple usage example for GraphTemporalForecaster package.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.meta_causal_model import GraphTemporalForecaster
from training.trainer import MetaCausalTrainer, build_sequences

def generate_simple_trend(n=500):
    """Generate a simple time series with trend and noise."""
    t = np.linspace(0, 10, n)
    trend = 2 * t
    seasonal = 5 * np.sin(2 * np.pi * t)
    noise = 0.5 * np.random.randn(n)
    return trend + seasonal + noise

def main():
    print("GraphTemporalForecaster - Simple Usage Example")
    print("=" * 50)
    
    # 1. Generate sample data
    print("\n1. Generating sample time series...")
    y = generate_simple_trend(500)
    print(f"Generated {len(y)} data points")
    
    # 2. Prepare data
    print("\n2. Preparing data...")
    seq_len = 30
    X, Y = build_sequences(y, seq_len)
    
    # Split into train/validation
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    Y_train, Y_val = Y[:train_size], Y[train_size:]
    
    print(f"Training sequences: {len(X_train)}")
    print(f"Validation sequences: {len(X_val)}")
    
    # 3. Create and train model
    print("\n3. Creating and training GraphTemporalForecaster...")
    model = GraphTemporalForecaster(
        input_dim=1,
        hidden_dim=32,
        seq_len=seq_len,
        embed_dim=8
    )
    
    trainer = MetaCausalTrainer(model, lr=1e-3, batch_size=16)
    
    # Prepare data loaders
    import torch
    train_ds = torch.utils.data.TensorDataset(X_train, Y_train)
    val_ds = torch.utils.data.TensorDataset(X_val, Y_val)
    trainer.train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
    trainer.val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)
    
    # Train
    trainer.train(epochs=50, verbose=True)
    
    # 4. Make predictions
    print("\n4. Making predictions...")
    predictions, adjacency = trainer.forecast(X_val)
    
    # 5. Evaluate
    print("\n5. Evaluating performance...")
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    mse = mean_squared_error(Y_val.squeeze(), predictions)
    mae = mean_absolute_error(Y_val.squeeze(), predictions)
    
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # 6. Plot results
    print("\n6. Plotting results...")
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Time series and predictions
    plt.subplot(2, 2, 1)
    plt.plot(y, label='Original Data', color='black', linewidth=2)
    
    # Plot predictions
    val_start = len(y) - len(predictions)
    plt.plot(range(val_start, len(y)), predictions, 
            label='GraphTemporalForecaster', color='red', linestyle='--', linewidth=2)
    
    plt.title('Time Series Forecasting')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Training curves
    plt.subplot(2, 2, 2)
    plt.plot(trainer.train_losses, label='Training Loss', color='blue')
    plt.plot(trainer.val_losses, label='Validation Loss', color='red')
    plt.title('Training Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Adjacency matrix (learned graph structure)
    plt.subplot(2, 2, 3)
    plt.imshow(adjacency[0], cmap='viridis', aspect='auto')
    plt.colorbar(label='Edge Weight')
    plt.title('Learned Graph Structure')
    plt.xlabel('Node')
    plt.ylabel('Node')
    
    # Plot 4: Prediction vs Actual
    plt.subplot(2, 2, 4)
    plt.scatter(Y_val.squeeze(), predictions, alpha=0.6, color='blue')
    plt.plot([Y_val.min(), Y_val.max()], [Y_val.min(), Y_val.max()], 
            'r--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Prediction vs Actual')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nExample completed successfully!")
    print("The model learned temporal dependencies through graph structures.")

if __name__ == "__main__":
    main() 