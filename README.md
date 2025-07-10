# GraphTemporalForecaster

A graph-based temporal forecasting model that learns temporal dependencies through learned graph structures. This model combines transformer-based graph learning with graph neural networks for time series forecasting.

## Features

- **Graph Learning**: Learns temporal dependencies through learned graph structures
- **Transformer Integration**: Uses transformer encoders for sequence processing
- **GNN Processing**: Graph neural networks for temporal pattern recognition
- **Flexible Architecture**: Configurable embedding dimensions and sequence lengths
- **Comprehensive Examples**: Wild time series testing against LightGBM and ARIMA

## Installation

```bash
pip install -r requirements.txt
```

Or install the package:

```bash
pip install -e .
```

## Quick Start

```python
from models.meta_causal_model import GraphTemporalForecaster
from training.trainer import GraphTemporalTrainer, build_sequences
import numpy as np

# Generate sample data
y = np.random.randn(1000)  # Your time series data

# Prepare sequences
X, Y = build_sequences(y, seq_len=30)

# Create and train model
model = GraphTemporalForecaster(
    input_dim=1,
    hidden_dim=64,
    seq_len=30,
    embed_dim=8
)

trainer = GraphTemporalTrainer(model, lr=1e-3, batch_size=32)
# ... train the model

# Make predictions
predictions, adjacency = trainer.forecast(X_test)
```

## Examples

### 1. Simple Usage Example

Run the basic example to see how the model works:

```bash
python examples/simple_usage_example.py
```

This demonstrates:
- Basic model training
- Prediction generation
- Performance evaluation
- Visualization of results

### 2. Wild Time Series Comparison

Test the model against LightGBM and ARIMA on challenging time series:

```bash
python examples/whacky_time_series_examples.py
```

This tests the model on:
- **Chaotic Lorenz Attractor**: Complex dynamical system
- **Fractal Noise**: Multi-scale patterns
- **Stock Market Crash Simulation**: Sudden regime changes
- **Multimodal Oscillator**: Multiple frequency components
- **Spike and Drift**: Sudden spikes with gradual trends

## Model Architecture

The GraphTemporalForecaster consists of three main components:

### 1. GraphLearner
- Transformer encoder for sequence processing
- Learns graph structure from temporal patterns
- Outputs adjacency matrix representing temporal dependencies

### 2. GNNCell
- Graph neural network for processing temporal relationships
- Uses learned adjacency matrix for message passing
- GRU-based update mechanism

### 3. Forecasting Head
- Decoder network for final predictions
- Combines learned temporal patterns with graph structure

## Performance Comparison

The model is tested against:
- **LightGBM**: Gradient boosting for tabular data
- **ARIMA**: Traditional time series model

Results show how the graph-based approach handles different types of temporal patterns.

## Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy 1.21+
- Pandas 1.3+
- Matplotlib 3.4+
- Scikit-learn 1.0+
- LightGBM 3.3+ (for comparisons)
- Statsmodels 0.13+ (for ARIMA comparisons)

## Project Structure

```
graph-temporal-forecaster/
├── models/
│   └── meta_causal_model.py      # Main model implementation
├── training/
│   └── trainer.py                 # Training utilities
├── data/
│   └── data_generator.py         # Data generation utilities
├── examples/
│   ├── simple_usage_example.py   # Basic usage example
│   └── whacky_time_series_examples.py  # Comparison examples
├── setup.py                      # Package setup
├── requirements.txt               # Dependencies
└── README.md                     # This file
```

## Key Insights

1. **Graph Learning**: The model learns temporal dependencies as graph structures, capturing complex relationships between time steps.

2. **Adaptive Architecture**: The transformer-based graph learner can adapt to different types of temporal patterns.

3. **Interpretability**: The learned adjacency matrix provides insights into temporal dependencies.

4. **Performance**: Shows competitive performance against traditional methods on complex time series.

## Contributing

Feel free to submit issues and enhancement requests!

## License

MIT License - see LICENSE file for details. 