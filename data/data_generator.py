import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import yfinance as yf

def generate_challenging_synthetic_data(T=100, seed=42):
    """
    Generate challenging synthetic time series data with multiple regimes,
    trends, seasonality, and shocks.
    """
    np.random.seed(seed)
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
    
    return y, time

def generate_stock_like_data(T=252, seed=42):
    """
    Generate synthetic data that mimics stock price behavior.
    """
    np.random.seed(seed)
    
    # Random walk with drift
    returns = np.random.normal(0.0005, 0.02, T)  # Daily returns
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Add some mean reversion
    for i in range(1, T):
        if prices[i] > 120:
            returns[i] -= 0.01
        elif prices[i] < 80:
            returns[i] += 0.01
    
    # Add volatility clustering
    volatility = np.zeros(T)
    volatility[0] = 0.02
    for i in range(1, T):
        volatility[i] = 0.95 * volatility[i-1] + 0.05 * abs(returns[i-1])
        returns[i] = np.random.normal(0.0005, volatility[i])
    
    prices = 100 * np.exp(np.cumsum(returns))
    
    return prices

def get_real_stock_data(symbol, start_date, end_date):
    """
    Download real stock data using yfinance.
    """
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(start=start_date, end=end_date)
        
        if data.empty:
            raise ValueError(f"No data found for {symbol}")
        
        return data
    except Exception as e:
        print(f"Error downloading data for {symbol}: {e}")
        return None

def compare_forecasting_methods(y, train_size=0.8):
    """
    Compare different forecasting methods on the data.
    """
    train_size = int(len(y) * train_size)
    y_train, y_test = y[:train_size], y[train_size:]
    
    # ARIMA
    try:
        arima_model = ARIMA(y_train, order=(3,1,2)).fit()
        arima_forecast = arima_model.forecast(steps=len(y_test))
    except:
        arima_forecast = np.full(len(y_test), np.mean(y_train))
    
    # Simple moving average
    ma_forecast = np.full(len(y_test), np.mean(y_train[-20:]))
    
    # Linear trend
    x_train = np.arange(len(y_train))
    x_test = np.arange(len(y_train), len(y_train) + len(y_test))
    trend_coef = np.polyfit(x_train, y_train, 1)
    trend_forecast = np.polyval(trend_coef, x_test)
    
    return {
        'arima': arima_forecast,
        'moving_average': ma_forecast,
        'linear_trend': trend_forecast,
        'y_test': y_test
    }

def plot_forecast_comparison(y, forecasts, title="Forecast Comparison"):
    """
    Plot comparison of different forecasting methods.
    """
    train_size = len(y) - len(forecasts['y_test'])
    
    plt.figure(figsize=(15, 8))
    plt.plot(np.arange(train_size), y[:train_size], label='Train', color='black', linewidth=2)
    plt.plot(np.arange(train_size, len(y)), y[train_size:], label='Test', color='blue', linewidth=2)
    
    colors = ['red', 'green', 'orange', 'purple', 'brown']
    for i, (method, forecast) in enumerate(forecasts.items()):
        if method != 'y_test':
            plt.plot(np.arange(train_size, len(y)), forecast, 
                    label=f'{method.replace("_", " ").title()}', 
                    linestyle='--', color=colors[i % len(colors)])
    
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def calculate_metrics(y_true, y_pred):
    """
    Calculate forecasting metrics.
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    
    # Directional accuracy
    direction_correct = np.sum(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred)))
    directional_accuracy = direction_correct / (len(y_true) - 1)
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'Directional_Accuracy': directional_accuracy
    } 