# FaultLab

FaultLab is a Master's Thesis project for comparing graph-based and conventional deep-learning approaches for fault analysis in electrical transmission networks.

The project uses an IEEE 14-bus fault dataset and compares:

- **ST-GAT** — topology-aware spatio-temporal Graph Attention Network
- **CNN** — temporal convolutional baseline
- **LSTM** — recurrent temporal baseline

The main tasks are:

- Fault-type classification
- Faulted-line localization
- Fault-position estimation
- Dynamic-security classification

## Dataset

The main dataset contains **21,120 IEEE 14-bus fault scenarios** with:

- 10 fault clearing times
- 4 fault types
- 16 faulted lines
- 3 fault positions
- 11 operating conditions
- Pre-fault and fault-on measurements from 5 generator buses

Raw datasets are not included in the repository because of file size.

Expected local path:

```text
data/raw/ieee14_fault/IEEE 14 bus test system data.xlsx
```

The project also keeps an IEEE 39-bus transient-stability dataset as secondary material, but the IEEE 14-bus dataset is the main dataset used for fault localization experiments.

## Project structure

```text
.
├── configs/
├── data/
│   ├── raw/
│   └── processed_ieee14/
├── reports/
│   └── baseline/
├── scripts/
│   ├── inspect_data.py
│   ├── prepare_data.py
│   ├── train_all.py
│   └── run_repeated.py
├── src/
│   └── powergrid_faults/
├── dashboard.py
├── pyproject.toml
└── requirements.txt
```

Model checkpoints and large datasets are intentionally excluded from Git.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Run the pipeline

Inspect the dataset:

```bash
python scripts/inspect_data.py
```

Prepare tensors and splits:

```bash
python scripts/prepare_data.py
```

Train the three main models:

```bash
python scripts/train_all.py
```

Run the Streamlit dashboard:

```bash
python -m streamlit run dashboard.py
```

Run repeated experiments with five seeds:

```bash
python scripts/run_repeated.py
```

## Current baseline results

Five-seed repeated experiments show that all three models perform similarly overall.

| Model | Fault-type Macro-F1 | Line Accuracy | Position MAE | Security Macro-F1 | Instability AUROC |
|---|---:|---:|---:|---:|---:|
| CNN | 0.947 ± 0.003 | **0.688 ± 0.020** | 12.24 ± 0.43 | 0.679 ± 0.031 | **0.987 ± 0.001** |
| LSTM | **0.948 ± 0.001** | 0.656 ± 0.010 | **12.05 ± 0.36** | **0.684 ± 0.013** | 0.987 ± 0.001 |
| ST-GAT | 0.945 ± 0.003 | 0.667 ± 0.006 | 12.37 ± 0.26 | 0.679 ± 0.003 | 0.986 ± 0.001 |

No statistically significant superiority of ST-GAT over CNN or LSTM was observed with five runs.

ST-GAT does, however, show lower run-to-run variability in several metrics.

Detailed outputs are stored in:

```text
reports/baseline/
```

## Next experiment

The next experiment is a **topology ablation**:

```text
ST-GAT + real IEEE-14 topology
vs
ST-GAT + shuffled topology
```

The purpose is to determine whether the physical grid structure itself contributes useful information to the graph-based model.

After that, only two robustness checks are planned:

- Missing PMUs
- Gaussian measurement noise

This keeps the experimental scope focused while still providing a defensible comparison for the thesis.

## Dashboard

The Streamlit dashboard provides:

- IEEE-14 topology visualization
- Model selection
- Scenario inspection
- Fault-type predictions
- Fault-line localization
- Fault-position estimates
- Dynamic-security predictions
- Model comparison
- Robustness results

## Thesis focus

The project evaluates whether explicitly incorporating electrical-grid topology through spatio-temporal graph neural networks improves fault localization and dynamic-security assessment compared with CNN and LSTM baselines.

The goal is comparative evaluation, not to assume in advance that the graph model must outperform the baselines.