# FaultLab — IEEE-14 ST-GAT vs CNN/LSTM

FaultLab is a reproducible deep-learning project for **fault classification, fault-line localization, fault-position estimation, and post-contingency dynamic-security assessment in power transmission networks**.

The main experiment compares a topology-aware **Spatio-Temporal Graph Attention Network (ST-GAT)** against two topology-agnostic baselines, a **Temporal CNN** and an **LSTM**, under the same dataset split, input information, training protocol, and evaluation metrics.

The project is designed as the experimental core of a Master's Thesis on whether graph-based deep learning can improve fault localization in critical power-grid infrastructure by explicitly exploiting the physical network topology.

---

## Current project status

The main IEEE-14 pipeline is operational end to end:

- IEEE-14 dataset inspection and validation
- leakage-safe preprocessing
- operating-condition-aware train/validation/test split
- IEEE-14 graph construction
- ST-GAT, CNN, and LSTM implementations
- multi-task training
- fault localization and dynamic-security metrics
- robustness analysis under noise and missing PMUs
- trained-model checkpoints for the interactive dashboard
- Streamlit + Plotly interactive visualization
- repeated-run and statistical-significance pipeline implemented

The repeated five-seed experiment is implemented but should be executed separately with:

```bash
python scripts/run_repeated.py
```

Repeated runs are stored under `artifacts_repeated/` and **do not overwrite** the checkpoints used by the dashboard in `artifacts/`.

---

## Research question

The main experimental question is:

> Does explicitly incorporating the electrical-grid topology through graph attention and message passing improve fault localization and robustness compared with conventional temporal CNN and LSTM architectures?

The project does not assume that ST-GAT must win. The purpose of the experimental design is to measure when the graph representation provides an advantage, particularly for fault-line localization and degraded observability.

---

## Main dataset

**IEEE 14 bus test systems row data**  
Figshare DOI: `10.6084/m9.figshare.30590399`

Place the downloaded Excel file at:

```text
data/raw/ieee14_fault/IEEE 14 bus test system data.xlsx
```

FaultLab uses only the Excel sheet:

```text
row data
```

The provided normalized sheet is intentionally ignored because preprocessing and scaling are fitted exclusively on the training partition to prevent data leakage.

### Dataset dimensions

The inspected release contains:

```text
21,120 scenarios
230 columns
```

The experimental design is perfectly factorized as:

```text
10 fault-clearing times
× 4 fault types
× 16 fault-line IDs
× 3 fault positions
× 11 operating conditions
= 21,120 scenarios
```

### Main metadata columns

| Column | Meaning |
|---|---|
| `FCT` | Fault clearing time, expressed in cycles |
| `FaultType` | Fault-type ID, values 1–4 |
| `lname` | Fault-line ID, values 1–16 |
| `POL` | Position on line, values 21.4%, 67.2%, 91.7% |
| `output` | Five-class post-contingency dynamic-security state |

### Dynamic-security labels

The `output` values are interpreted as:

```text
0 → Unstable
1 → Urgent
2 → Strong
3 → Alarm
4 → Normal
```

Observed class counts in the Excel release:

```text
Unstable: 5,944
Urgent:     647
Strong:   2,920
Alarm:    6,681
Normal:   4,928
```

Because the `Urgent` class is substantially smaller than the others, the training pipeline supports balanced class weighting for the security objective.

---

## Important dataset limitation

All 21,120 rows represent **fault scenarios**.

Therefore, this dataset supports:

- fault-type classification
- fault-line localization
- fault-position estimation
- dynamic-security classification
- instability vs non-instability evaluation

It does **not** contain a clean normal-operation / no-fault class. Consequently, the binary metric reported by FaultLab is:

```text
unstable vs non-unstable
```

and must not be presented as:

```text
fault vs no-fault
```

This distinction is intentionally preserved in the code, dashboard, and thesis methodology.

---

## Input representation

The raw Excel contains:

```text
4 metadata columns
15 pre-fault measurements
210 dynamic measurements
1 output column
= 230 columns
```

The 15 pre-fault measurements correspond to:

```text
P × 5 generator buses
Q × 5 generator buses
V × 5 generator buses
```

The 210 dynamic features are reorganized into:

```text
14 dynamic blocks × 5 generators × 3 electrical variables
```

so the original dynamic representation becomes:

```text
[samples, 14, 5, 3]
```

The five observed generator buses are:

```text
1, 2, 3, 6, 8
```

These measurements are projected onto the full IEEE-14 graph. Unobserved buses receive zero-valued measurement channels together with an explicit observation mask.

The final model tensor is:

```text
X.shape = [samples, 14 dynamic blocks, 14 buses, 7 features]
```

Each bus contains:

```text
[
  dynamic_P,
  dynamic_Q,
  dynamic_V,
  prefault_P,
  prefault_Q,
  prefault_V,
  observed
]
```

If the optional FCT ablation is enabled, one additional feature is appended.

---

## Leakage prevention

The following columns are targets or fault metadata and are not allowed into the main model input:

```text
FaultType
lname
POL
output
```

`FCT` is also excluded from the main experiment by default:

```yaml
dataset:
  include_fct_as_feature: false
```

This creates a stricter online-style experiment. FCT can later be enabled as a controlled ablation.

Scaling follows the correct order:

```text
split operating conditions
        ↓
fit scalers on TRAIN only
        ↓
transform train
transform validation
transform test
```

---

## Operating-condition split

The 1,920 unique combinations of:

```text
FCT × FaultType × lname × POL
```

appear exactly 11 times, corresponding to the 11 operating conditions in the release.

The preprocessing pipeline attempts to recover operating-condition groups from the pre-fault P/Q/V signature. The main split keeps complete operating conditions separate:

```text
8 groups → training
1 group  → validation
2 groups → test
```

This is intentionally more demanding than a random row split because it evaluates generalization to operating conditions that were not used to fit the model.

The detected grouping method and selected groups are written to:

```text
data/processed_ieee14/metadata.json
reports/ieee14_schema_report.json
```

---

## IEEE-14 topology

The graph contains:

```text
14 buses
16 faultable line objects
5 transformer objects
```

The reference PowerFactory-style fault-line set currently used by the visualization is:

```text
1-2 circuit 1
1-2 circuit 2
1-5
2-3
2-4
2-5
3-4
4-5
6-11
6-12
6-13
9-10
9-14
10-11
12-13
13-14
```

The five transformer connections are also included in message passing.

### Line-ID mapping caveat

The Excel provides:

```text
lname = 1..16
```

but does not include an explicit `lname → physical endpoints` lookup table.

`topology.py` therefore uses a reference PowerFactory ordering for visualization and graph construction. Until the exact ID-to-endpoint mapping is independently verified from author metadata, the thesis should treat **line-ID metrics** as the primary localization results and avoid making unsupported endpoint-specific claims.

The default configuration therefore keeps:

```yaml
model:
  edge_localization_head: false
```

An edge-based localization head is already implemented and can be enabled after the mapping is verified.

---

## Multi-task learning objectives

All three architectures solve the same four tasks.

### 1. Fault-type classification

```text
4 classes
```

Target:

```text
FaultType
```

### 2. Fault-line localization

```text
16 classes
```

Target:

```text
lname
```

### 3. Fault-position localization

Three position classes:

```text
21.4%
67.2%
91.7%
```

The network predicts a probability distribution over the three positions. An expected position is also computed for continuous error metrics.

### 4. Dynamic-security classification

```text
Unstable
Urgent
Strong
Alarm
Normal
```

A derived binary instability score is also evaluated:

```text
Unstable vs all other states
```

---

## Models

### ST-GAT

The proposed model combines graph attention and temporal recurrence.

For each dynamic block:

```text
bus measurements
      ↓
GATv2
      ↓
message passing through IEEE-14
      ↓
GATv2
      ↓
node embeddings
```

The sequence of graph embeddings is then processed by a GRU:

```text
graph embeddings over 14 dynamic blocks
                    ↓
                   GRU
                    ↓
             multi-task heads
```

The graph enables the model to explicitly incorporate electrical connectivity while learning the temporal evolution of the disturbance.

### Temporal CNN

A 1-D temporal convolutional baseline using the same underlying measurements but without access to `edge_index`.

Its role is to test whether local temporal patterns alone can match the graph-aware approach.

### LSTM

A recurrent temporal baseline that receives the flattened system state over the same sequence of dynamic blocks.

It can learn long-range temporal dependencies but does not explicitly receive the physical grid connectivity.

---

## Multi-task loss

Training optimizes a weighted combination of:

```text
fault-type cross entropy
fault-line cross entropy
position cross entropy
dynamic-security cross entropy
```

Default weights are configured in `configs/default.yaml`:

```yaml
training:
  lambda_fault_type: 1.0
  lambda_line: 1.5
  lambda_position: 1.0
  lambda_security: 0.75
```

Fault-line localization receives a slightly larger weight because it is the central task of the thesis.

---

## Evaluation metrics

### Fault type

- Macro F1
- Accuracy where applicable

### Fault-line localization

- Top-1 accuracy
- Macro F1
- Top-3 accuracy
- Corridor accuracy

`line_corridor_accuracy` merges the two parallel 1–2 circuits in the reference topology when evaluating the physical corridor.

### Fault-position localization

- Expected-position MAE
- Expected-position RMSE
- Median absolute error where available
- Accuracy within ±5 percentage points
- Accuracy within ±10 percentage points

### Dynamic security

- Macro F1
- Balanced accuracy
- Ordinal MAE

### Instability

- AUROC
- F1
- Precision / recall where generated

### Computational metrics

- Number of trainable parameters
- Inference latency per sample

---

## Robustness experiments

The project evaluates the three architectures under degraded measurements.

### Gaussian measurement noise

Default standard deviations:

```yaml
robustness:
  noise_std: [0.0, 0.02, 0.05, 0.10]
```

### Missing PMUs

Default fractions:

```yaml
robustness:
  missing_pmu_fraction: [0.0, 0.20, 0.40, 0.60]
```

This experiment is particularly relevant to the graph-learning hypothesis: if message passing is useful under partial observability, ST-GAT may degrade more gracefully than topology-agnostic baselines.

The hypothesis must be tested from the results rather than assumed in advance.

---

## Repeated experiments and statistical testing

For thesis-quality comparison, a single random initialization is not sufficient.

Run:

```bash
python scripts/run_repeated.py
```

The current repeated protocol uses:

```text
seeds = [11, 22, 33, 44, 55]
```

for:

```text
CNN
LSTM
ST-GAT
```

This produces 15 independent training runs.

Repeated checkpoints are written to:

```text
artifacts_repeated/
├── seed_11/
│   ├── cnn/
│   ├── lstm/
│   └── stgat/
├── seed_22/
├── seed_33/
├── seed_44/
└── seed_55/
```

The primary dashboard checkpoints under `artifacts/` are left untouched.

Generated statistical outputs:

```text
reports/repeated_runs.csv
reports/repeated_summary.csv
reports/significance.csv
```

The significance table includes paired comparisons of ST-GAT against CNN and LSTM using:

- paired t-test
- Wilcoxon signed-rank test
- Cohen's dz effect size

The thesis should report mean ± standard deviation across seeds and should only claim statistically significant superiority when supported by the repeated-run results.

---

## Recommended ablation studies

The most useful additional experiments are:

### Real vs shuffled topology

```text
ST-GAT with IEEE-14 edges
vs
ST-GAT with randomized edges
```

This is one of the strongest tests of whether the physical graph itself contributes predictive value.

### GATv2 vs GCN

Tests whether learned attention improves over a simpler graph convolution.

### With vs without temporal recurrence

```text
GATv2 + GRU
vs
GATv2 without GRU
```

Tests whether the temporal component adds measurable value.

### With vs without FCT

```text
signals only
vs
signals + FCT
```

Quantifies how much the known fault-clearing time contributes.

### Operating-condition split vs random split

The group-wise split is the primary protocol. A random split can be included only as a secondary comparison with less demanding generalization assumptions.

---

## Interactive dashboard

Run:

```bash
streamlit run dashboard.py
```

The dashboard is fully English-language and uses **FaultLab** consistently as the application name.

It allows the user to:

- switch between ST-GAT, Temporal CNN, and LSTM
- select train, validation, or test scenarios
- inspect individual operating conditions and FCT values
- compare predicted vs actual fault type
- compare predicted vs actual fault-line ID
- inspect predicted fault position
- inspect dynamic-security predictions
- visualize P(unstable)
- view the IEEE-14 network topology
- inspect top candidate fault lines
- inspect fault-position probability distributions
- inspect dynamic P/Q/V measurements at observed generator buses
- compare model-level benchmark metrics
- inspect robustness curves

Dataset limitations and the unverified physical line-ID mapping are kept in a compact sidebar note rather than dominating the main visualization.

---

## Installation

Recommended Python version:

```text
Python 3.11–3.13
```

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
pip install -e .
```

Verify the package:

```bash
python -c "import powergrid_faults; print('POWERGRID IMPORT OK')"
```

---

## End-to-end workflow

### 1. Place the dataset

```text
data/raw/ieee14_fault/IEEE 14 bus test system data.xlsx
```

### 2. Verify dataset availability

```bash
python scripts/download_data.py
```

For the current Figshare workflow this script checks the expected local file rather than downloading the inaccessible Kaggle dataset used during early prototyping.

### 3. Inspect the dataset

```bash
python scripts/inspect_data.py
```

### 4. Prepare tensors and splits

```bash
python scripts/prepare_data.py
```

Expected main tensor shape:

```text
[21120, 14, 14, 7]
```

### 5. Train all primary models

```bash
python scripts/train_all.py
```

This trains:

```text
CNN
LSTM
ST-GAT
```

and generates the main benchmark and robustness reports.

### 6. Launch the dashboard

```bash
streamlit run dashboard.py
```

### 7. Run repeated experiments

```bash
python scripts/run_repeated.py
```

Run this after the main pipeline is stable because it performs 15 complete training runs.

---

## Main repository structure

```text
.
├── configs/
│   └── default.yaml
│
├── data/
│   ├── raw/
│   │   └── ieee14_fault/
│   │       └── IEEE 14 bus test system data.xlsx
│   └── processed_ieee14/
│       ├── dataset.npz
│       ├── metadata.json
│       └── scalers.joblib
│
├── artifacts/
│   ├── cnn/
│   ├── lstm/
│   └── stgat/
│
├── artifacts_repeated/
│   └── seed_<seed>/...
│
├── reports/
│   ├── ieee14_schema_report.json
│   ├── comparison.csv
│   ├── robustness.csv
│   ├── repeated_runs.csv
│   ├── repeated_summary.csv
│   └── significance.csv
│
├── scripts/
│   ├── download_data.py
│   ├── inspect_data.py
│   ├── prepare_data.py
│   ├── train_all.py
│   └── run_repeated.py
│
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
│
├── tests/
│   └── test_shapes.py
│
├── dashboard.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Generated artifacts

Primary checkpoints:

```text
artifacts/cnn/best.pt
artifacts/lstm/best.pt
artifacts/stgat/best.pt
```

Main reports:

```text
reports/comparison.csv
reports/robustness.csv
reports/ieee14_schema_report.json
```

Repeated-run reports:

```text
reports/repeated_runs.csv
reports/repeated_summary.csv
reports/significance.csv
```

---

## Repository and data policy

Large raw datasets, processed tensors, virtual environments, and model checkpoints should normally remain outside Git history.

Recommended `.gitignore` entries include:

```gitignore
.venv/
__pycache__/
.DS_Store

data/raw/
data/processed/
data/processed_ieee14/

*.xlsx
*.xls
*.pkl
*.npz

artifacts/
artifacts_repeated/
*.pt
*.pth
*.ckpt
```

Small reproducibility outputs such as benchmark CSV files can be committed if desired.

The README should remain sufficient for another researcher to download the public dataset, reconstruct the processed tensors, retrain the models, and reproduce the reported analyses.

---

## Thesis interpretation

The central comparison is not simply which neural network obtains the highest accuracy.

The intended scientific contribution is to test whether explicit representation of power-grid structure provides measurable value for:

1. fault-line localization,
2. fault-position estimation,
3. generalization to unseen operating conditions,
4. robustness under noisy measurements,
5. robustness under partial PMU observability.

A convincing final conclusion should combine:

- mean ± standard deviation across repeated runs,
- statistical significance tests,
- effect sizes,
- localization metrics,
- robustness results,
- topology ablations,
- latency / model-size trade-offs,
- qualitative examples from the interactive network visualization.

If ST-GAT only improves specific tasks or degraded-observability conditions, that is still a valid and potentially more interesting conclusion than claiming universal superiority.

---

## Suggested final thesis experiments

Before freezing the experimental chapter, complete at least:

1. five-seed ST-GAT vs CNN vs LSTM comparison,
2. statistical significance analysis,
3. real-topology vs shuffled-topology ablation,
4. missing-PMU robustness comparison,
5. measurement-noise robustness comparison,
6. with-FCT vs without-FCT ablation,
7. final dashboard screenshots / qualitative examples,
8. final results tables and figures for the thesis.

---

## Application name

The project and dashboard use the single name:

# FaultLab

`IEEE-14` is the current experimental network, not part of the product name itself. This keeps the interface consistent and leaves room for future IEEE-39 or other-network experiments under the same application.