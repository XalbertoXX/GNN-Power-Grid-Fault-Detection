# FaultLab

FaultLab is a Master's Thesis project for evaluating deep-learning approaches for fault analysis in electrical transmission networks.

The project compares three architectures on an IEEE 14-bus fault dataset:

- **ST-GAT** — topology-aware spatio-temporal Graph Attention Network
- **CNN** — temporal convolutional baseline
- **LSTM** — recurrent temporal baseline

The evaluated tasks are fault-type classification, faulted-line localization, fault-position estimation, and dynamic-security classification.

## Dataset

The main dataset contains **21,120 IEEE 14-bus fault scenarios** generated from:

- 10 fault clearing times
- 4 fault types
- 16 faulted lines
- 3 fault positions
- 11 operating conditions

It includes pre-fault and fault-on measurements from five generator buses.

Raw datasets are not stored in Git because of file size. The expected local location is:

```text
data/raw/ieee14_fault/IEEE 14 bus test system data.xlsx
```

## Project structure

```text
tfm_powergrid_gnn_ieee14/
├── data/
│   └── raw/
├── figures/
├── reports/
│   └── baseline/
├── scripts/
│   ├── download_data.py
│   ├── generate_figures.py
│   ├── inspect_data.py
│   ├── prepare_data.py
│   ├── run_repeated.py
│   ├── run_topology_ablation.py
│   └── train_all.py
├── src/
│   └── powergrid_faults/
│       ├── __init__.py
│       ├── data.py
│       ├── metrics.py
│       ├── models.py
│       ├── topology.py
│       ├── trainlib.py
│       ├── utils.py
│       └── viz.py
├── tests/
├── .gitignore
├── dashboard.py
├── pyproject.toml
├── README.md
├── requirements.txt
├── TFM_EXPERIMENT_PLAN.md
└── LICENSE
```

Generated model checkpoints and large datasets are intentionally excluded from version control.

## Setup

```bash
# cd to folder tfm_powergrid_gnn_ieee14
python -m venv .venv
source .venv/bin/activate

python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Main workflow

Inspect the dataset:

```bash
python scripts/inspect_data.py
```

Prepare the processed tensors and train/validation/test splits:

```bash
python scripts/prepare_data.py
```

Train CNN, LSTM and ST-GAT:

```bash
python scripts/train_all.py
```

Run the interactive dashboard:

```bash
python -m streamlit run dashboard.py
```

Run the five-seed repeated evaluation:

```bash
python scripts/run_repeated.py
```

Run the topology ablation:

```bash
python scripts/run_topology_ablation.py
```

Generate the final thesis figures:

```bash
python scripts/generate_figures.py
```

## Experimental results

Five-seed repeated evaluation produced the following baseline results:

| Model | Fault-type Macro-F1 | Line Accuracy | Position MAE | Security Macro-F1 | Instability AUROC |
|---|---:|---:|---:|---:|---:|
| CNN | 0.947 ± 0.003 | **0.688 ± 0.020** | 12.24 ± 0.43 | 0.679 ± 0.031 | **0.987 ± 0.001** |
| LSTM | **0.948 ± 0.001** | 0.656 ± 0.010 | **12.05 ± 0.36** | **0.684 ± 0.013** | 0.987 ± 0.001 |
| ST-GAT | 0.945 ± 0.003 | 0.667 ± 0.006 | 12.37 ± 0.26 | 0.679 ± 0.003 | 0.986 ± 0.001 |

No statistically significant superiority of ST-GAT over the CNN or LSTM baselines was observed in the standard evaluation.

The topology ablation also showed no performance advantage from the physical IEEE-14 connectivity over a degree-preserving shuffled graph.

However, the robustness evaluation showed that **ST-GAT degrades less strongly when PMU observations are removed**, particularly for fault classification and line localization. This suggests that the graph-based architecture may be most useful under reduced grid observability rather than under clean, fully available measurements.

Detailed experiment outputs are stored under `reports/`, while publication-ready figures are generated under `figures/`.

## Dashboard

`dashboard.py` provides an interactive view of the IEEE-14 experiment, including:

- network topology
- scenario selection
- model predictions
- fault-line localization
- fault-position estimation
- dynamic-security prediction
- model comparison
- robustness results

## Thesis objective

The project evaluates whether explicitly incorporating electrical-grid structure through a spatio-temporal graph neural network improves fault analysis compared with conventional temporal CNN and LSTM architectures.

The experiments are designed as a comparative study; the graph model is not assumed to outperform the baselines in advance.