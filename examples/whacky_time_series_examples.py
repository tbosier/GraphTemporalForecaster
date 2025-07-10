#!/usr/bin/env python3
"""
Examples testing GraphTemporalForecaster against LightGBM and ARIMA on wild time series.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Import our model
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.meta_causal_model import GraphTemporalForecaster
from training.trainer import MetaCausalTrainer, build_sequences

# Import comparison models
try:
    import lightgbm as lgb
    from statsmodels.tsa.arima.model import ARIMA
    LIGHTGBM_AVAILABLE = True
    ARIMA_AVAILABLE = True
except ImportError:
    print("Warning: LightGBM or statsmodels not available. Install with: pip install lightgbm statsmodels")
    LIGHTGBM_AVAILABLE = False
    ARIMA_AVAILABLE = False

def generate_chaotic_time_series(n=1000):
    """Generate chaotic Lorenz attractor time series."""
    dt = 0.01
    x, y, z = 1.0, 1.0, 1.0
    xs, ys, zs = [], [], []
    
    for i in range(n):
        dx = 10 * (y - x)
        dy = x * (28 - z) - y
        dz = x * y - (8/3) * z
        
        x += dx * dt
        y += dy * dt
        z += dz * dt
        
        xs.append(x)
        ys.append(y)
        zs.append(z)
    
    # Use x component as our time series
    return np.array(xs)

def generate_fractal_noise(n=1000, octaves=6):
    """Generate fractal noise (like Perlin noise)."""
    noise = np.zeros(n)
    amplitude = 1.0
    frequency = 1.0
    
    for _ in range(octaves):
        # Generate random noise at current frequency
        phase = np.random.uniform(0, 2*np.pi, int(n * frequency))
        wave = np.sin(np.linspace(0, 2*np.pi*n, n) + phase)
        noise += amplitude * wave
        amplitude *= 0.5
        frequency *= 2
    
    return noise

def generate_stock_crash_simulation(n=1000):
    """Generate a time series that simulates a stock market crash."""
    t = np.linspace(0, 10, n)
    
    # Normal growth phase
    growth = 100 * np.exp(0.1 * t)
    
    # Add some volatility
    volatility = 10 * np.sin(2 * np.pi * t) + 5 * np.sin(2 * np.pi * 3 * t)
    
    # Add crash at 70% of the way through
    crash_point = int(0.7 * n)
    crash = np.zeros(n)
    crash[crash_point:] = -50 * np.exp(-0.5 * (t[crash_point:] - t[crash_point]))
    
    # Add recovery
    recovery = np.zeros(n)
    recovery[crash_point:] = 20 * (1 - np.exp(-0.3 * (t[crash_point:] - t[crash_point])))
    
    return growth + volatility + crash + recovery

def generate_multimodal_oscillator(n=1000):
    """Generate time series with multiple oscillating modes."""
    t = np.linspace(0, 20, n)
    
    # Multiple frequencies
    y = (np.sin(2 * np.pi * 0.5 * t) + 
         0.5 * np.sin(2 * np.pi * 2 * t) + 
         0.25 * np.sin(2 * np.pi * 8 * t))
    
    # Add trend
    trend = 0.1 * t
    
    # Add some chaos
    chaos = 0.1 * np.random.randn(n)
    
    return y + trend + chaos

def generate_spike_and_drift(n=1000):
    """Generate time series with sudden spikes and gradual drift."""
    t = np.linspace(0, 10, n)
    
    # Base trend
    base = 10 + 2 * t
    
    # Random spikes
    spikes = np.zeros(n)
    spike_indices = np.random.choice(n, size=20, replace=False)
    spikes[spike_indices] = np.random.uniform(5, 15, len(spike_indices))
    
    # Gradual drift
    drift = 0.5 * np.sin(0.5 * t)
    
    # Noise
    noise = 0.5 * np.random.randn(n)
    
    return base + spikes + drift + noise

def prepare_data_for_models(y, seq_len=30, train_ratio=0.8):
    """Prepare data for all models."""
    # Create sequences
    X, Y = build_sequences(y, seq_len)
    
    # Split data
    train_size = int(len(X) * train_ratio)
    X_train, X_val = X[:train_size], X[train_size:]
    Y_train, Y_val = Y[:train_size], Y[train_size:]
    
    return X_train, Y_train, X_val, Y_val, y

def train_graph_forecaster(X_train, Y_train, X_val, Y_val):
    """Train GraphTemporalForecaster."""
    model = GraphTemporalForecaster(
        input_dim=1, 
        hidden_dim=64, 
        seq_len=30, 
        embed_dim=8
    )
    
    trainer = MetaCausalTrainer(model, lr=1e-3, batch_size=32)
    trainer.train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, Y_train), 
        batch_size=32, shuffle=True
    )
    trainer.val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val, Y_val), 
        batch_size=32, shuffle=False
    )
    
    trainer.train(epochs=100, verbose=False)
    return trainer

def train_lightgbm(X_train, Y_train):
    """Train LightGBM model."""
    if not LIGHTGBM_AVAILABLE:
        return None
    
    # Reshape data for LightGBM
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    Y_train_flat = Y_train.squeeze()
    
    model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train_flat, Y_train_flat)
    return model

def train_arima(y_train):
    """Train ARIMA model."""
    if not ARIMA_AVAILABLE:
        return None
    
    try:
        # Try different ARIMA parameters
        model = ARIMA(y_train, order=(5, 1, 2))
        fitted_model = model.fit()
        return fitted_model
    except:
        # Fallback to simpler model
        try:
            model = ARIMA(y_train, order=(2, 1, 1))
            fitted_model = model.fit()
            return fitted_model
        except:
            return None

def evaluate_models(trainer, lgb_model, arima_model, X_val, Y_val, y_val, y_train=None):
    """Evaluate all models."""
    results = {}
    
    # GraphTemporalForecaster
    if trainer is not None:
        with torch.no_grad():
            preds, _ = trainer.forecast(X_val)
        results['GraphTemporalForecaster'] = {
            'predictions': preds,
            'mse': mean_squared_error(Y_val.squeeze(), preds),
            'mae': mean_absolute_error(Y_val.squeeze(), preds)
        }
    
    # LightGBM
    if lgb_model is not None:
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        preds = lgb_model.predict(X_val_flat)
        results['LightGBM'] = {
            'predictions': preds,
            'mse': mean_squared_error(Y_val.squeeze(), preds),
            'mae': mean_absolute_error(Y_val.squeeze(), preds)
        }
    
    # ARIMA
    if arima_model is not None:
        try:
            preds = arima_model.forecast(steps=len(y_val))
            results['ARIMA'] = {
                'predictions': preds,
                'mse': mean_squared_error(y_val, preds),
                'mae': mean_absolute_error(y_val, preds)
            }
        except:
            results['ARIMA'] = {'predictions': None, 'mse': float('inf'), 'mae': float('inf')}
    
    return results

def plot_comparison(y, results, title):
    """Plot comparison of all models."""
    plt.figure(figsize=(15, 10))
    
    # Plot original data
    plt.subplot(2, 1, 1)
    plt.plot(y, label='Original Data', linewidth=2, color='black')
    
    # Plot predictions
    colors = ['red', 'blue', 'green', 'orange']
    for i, (model_name, result) in enumerate(results.items()):
        if result['predictions'] is not None:
            if model_name == 'ARIMA':
                # ARIMA predictions are for the validation period
                val_start = len(y) - len(result['predictions'])
                plt.plot(range(val_start, len(y)), result['predictions'], 
                        label=f'{model_name}', color=colors[i], linestyle='--', linewidth=2)
            else:
                # Other models predict next values for each sequence
                val_start = len(y) - len(result['predictions'])
                plt.plot(range(val_start, len(y)), result['predictions'], 
                        label=f'{model_name}', color=colors[i], linestyle='--', linewidth=2)
    
    plt.title(f'{title} - Time Series and Predictions')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot metrics
    plt.subplot(2, 1, 2)
    models = list(results.keys())
    mse_scores = [results[model]['mse'] for model in models]
    mae_scores = [results[model]['mae'] for model in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, mse_scores, width, label='MSE', color='skyblue')
    plt.bar(x + width/2, mae_scores, width, label='MAE', color='lightcoral')
    
    plt.xlabel('Models')
    plt.ylabel('Error')
    plt.title('Model Performance Comparison')
    plt.xticks(x, models)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print(f"\n{title} Results:")
    print("-" * 50)
    for model_name, result in results.items():
        if result['predictions'] is not None:
            print(f"{model_name}:")
            print(f"  MSE: {result['mse']:.4f}")
            print(f"  MAE: {result['mae']:.4f}")
        else:
            print(f"{model_name}: Failed to train/predict")

def run_experiment(data_generator, title):
    """Run a complete experiment on a time series."""
    print(f"\n{'='*60}")
    print(f"Testing on: {title}")
    print(f"{'='*60}")
    
    # Generate data
    y = data_generator()
    print(f"Generated {len(y)} data points")
    print(f"Data range: {y.min():.2f} to {y.max():.2f}")
    
    # Prepare data
    X_train, Y_train, X_val, Y_val, y_full = prepare_data_for_models(y)
    print(f"Training on {len(X_train)} sequences, validating on {len(X_val)} sequences")
    
    # Train models
    print("\nTraining models...")
    
    # GraphTemporalForecaster
    trainer = train_graph_forecaster(X_train, Y_train, X_val, Y_val)
    print("✓ GraphTemporalForecaster trained")
    
    # LightGBM
    lgb_model = train_lightgbm(X_train, Y_train) if LIGHTGBM_AVAILABLE else None
    if lgb_model:
        print("✓ LightGBM trained")
    else:
        print("✗ LightGBM not available")
    
    # ARIMA
    arima_model = train_arima(y_full[:len(y_full)-len(X_val)]) if ARIMA_AVAILABLE else None
    if arima_model:
        print("✓ ARIMA trained")
    else:
        print("✗ ARIMA not available")
    
    # Evaluate
    results = evaluate_models(trainer, lgb_model, arima_model, X_val, Y_val, y_full[-len(X_val):])
    
    # Plot and print results
    plot_comparison(y_full, results, title)
    
    return results

def main():
    """Run all experiments."""
    print("GraphTemporalForecaster vs LightGBM vs ARIMA on Wild Time Series")
    print("=" * 70)
    
    # Define experiments
    experiments = [
        (generate_chaotic_time_series, "Chaotic Lorenz Attractor"),
        (generate_fractal_noise, "Fractal Noise"),
        (generate_stock_crash_simulation, "Stock Market Crash Simulation"),
        (generate_multimodal_oscillator, "Multimodal Oscillator"),
        (generate_spike_and_drift, "Spike and Drift"),
    ]
    
    all_results = {}
    
    for data_gen, title in experiments:
        try:
            results = run_experiment(data_gen, title)
            all_results[title] = results
        except Exception as e:
            print(f"Error in {title}: {e}")
            continue
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY OF ALL EXPERIMENTS")
    print(f"{'='*70}")
    
    for title, results in all_results.items():
        print(f"\n{title}:")
        for model_name, result in results.items():
            if result['predictions'] is not None:
                print(f"  {model_name}: MSE={result['mse']:.4f}, MAE={result['mae']:.4f}")
            else:
                print(f"  {model_name}: Failed")

if __name__ == "__main__":
    main() 