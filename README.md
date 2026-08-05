# GNN-Power-Grid-Fault-Detection
Spatio-temporal Graph Neural Network (ST-GNN) project designed for fault detection and localization in electrical transmission networks with benchmarking against CNN and LSTM.

This repository contains a modular Deep Learning architecture developed for a Master's Thesis. It focuses on fault detection and localization in critical infrastructures (power grids) by comparing Spatio-Temporal Graph Neural Networks (ST-GNN) against traditional CNN and LSTM approaches.

## Project Structure

The project is organized into modular packages for scalability and clean code practices:

- `dataset/`: Contains data processing logic and topology construction (`dataset.py`).
- `models/`: Contains the PyTorch neural network architectures (`base_cnn.py`), including the proposed ST-GNN with Graph Attention (GAT) alongside the baseline comparisons.
- `data_visualization/`: Includes tools to render the network's spatial topology and comparative performance metrics (`data_visualization.py`).
- `main/`: The entry point for the pipeline (`main.py`), orchestrating the data flow, model inference, and rendering.

## Setup & Installation

It is recommended to run this project within a virtual environment.

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Execute the main pipeline from the root directory:
   ```bash
   python main/main.py
   ```

Note: The `main.py` script automatically handles the `sys.path` resolution, meaning imports will work seamlessly even when executed from your local macOS terminal.
