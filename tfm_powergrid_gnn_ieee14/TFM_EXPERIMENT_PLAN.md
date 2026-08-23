# TFM Experimental Plan

## 1. Project title

**FaultLab: Comparative Fault Localization in Power Transmission Networks Using Spatio-Temporal Graph Neural Networks, CNNs and LSTMs**

## 2. Main objective

The objective of this Master's Thesis is to evaluate whether explicitly incorporating electrical-grid structure through a spatio-temporal Graph Neural Network provides advantages over conventional deep-learning architectures for fault analysis in power transmission systems.

The study compares:

- **ST-GAT** — graph-attention + temporal recurrent modeling
- **CNN** — temporal convolutional baseline
- **LSTM** — recurrent temporal baseline

The comparison focuses on predictive performance, robustness, stability across repeated runs, and the contribution of physical grid topology.

## 3. Research questions

The experiments are designed to answer the following questions:

1. Does ST-GAT outperform CNN and LSTM for fault classification and localization under standard conditions?
2. Does the physical IEEE-14 topology provide measurable value to the graph-based model?
3. Which architecture is more robust to degraded observability and measurement noise?
4. Are observed differences consistent across repeated runs?

The study is comparative and does not assume in advance that ST-GAT must outperform the baselines.

## 4. Dataset

The main dataset is based on an **IEEE 14-bus transmission system** and contains **21,120 fault scenarios**.

The scenario space is composed of:

- 10 fault clearing times
- 4 fault types
- 16 faulted lines
- 3 fault positions
- 11 operating conditions

The dataset includes:

- fault metadata
- pre-fault active power, reactive power and voltage
- fault-on dynamic measurements
- measurements from five generator buses
- a five-class dynamic-security output

Main targets:

- `FaultType` — fault-type classification
- `lname` — faulted-line classification
- `POL` — fault-position estimation
- `output` — dynamic-security classification

The dataset contains fault scenarios rather than an explicit no-fault class, so the project does not claim a fault-vs-no-fault detection benchmark.

## 5. Input representation

Dynamic measurements are reconstructed into a temporal representation and mapped to the IEEE-14 network.

The model input contains:

- dynamic active power
- dynamic reactive power
- dynamic voltage
- pre-fault active power
- pre-fault reactive power
- pre-fault voltage
- observation mask

Only the five generator buses contain directly observed dynamic measurements. Unobserved buses are represented through masking and graph propagation.

Target variables and metadata used as labels are excluded from the model inputs to prevent target leakage.

## 6. Data splitting and normalization

The experimental pipeline separates training, validation and test data before normalization.

Normalization parameters are fitted on the training set only and then applied to validation and test data.

Where possible, operating-condition grouping is used so that evaluation is not based only on random row-level mixing.

## 7. Models

### 7.1 CNN baseline

The CNN processes temporal measurements without explicit access to the electrical topology.

Its role is to provide a strong convolutional baseline for detecting local temporal patterns in the measurements.

### 7.2 LSTM baseline

The LSTM models temporal dependencies through recurrent hidden states.

It receives the same measurement information as the other models but does not explicitly use the power-grid graph.

### 7.3 ST-GAT

The proposed graph-based architecture combines:

- **GATv2** for spatial message passing
- **GRU-based temporal modeling**
- shared latent representations
- multi-task prediction heads

The graph structure is based on the IEEE-14 electrical network.

## 8. Prediction tasks

The three models are evaluated on the same tasks:

### Fault-type classification
Primary metric:
- Macro-F1

### Faulted-line localization
Primary metrics:
- Accuracy
- Macro-F1

Additional metric:
- Top-3 accuracy

### Fault-position estimation
Primary metric:
- Mean Absolute Error in percentage points

Additional metrics:
- RMSE
- within ±5 percentage points
- within ±10 percentage points

### Dynamic-security classification
Primary metrics:
- Macro-F1
- balanced accuracy

### Instability discrimination
Derived binary evaluation:
- AUROC
- F1
- precision
- recall

## 9. Standard comparative experiment

CNN, LSTM and ST-GAT are trained and evaluated under the same data splits and task definitions.

The main comparison is repeated using five independent random seeds:

```text
11, 22, 33, 44, 55
```

The repeated experiment reports:

- mean
- standard deviation
- paired t-test
- Wilcoxon signed-rank test
- Cohen's dz

### Current repeated-run results

| Model | Fault-type Macro-F1 | Line Accuracy | Position MAE | Security Macro-F1 | Instability AUROC |
|---|---:|---:|---:|---:|---:|
| CNN | 0.947 ± 0.003 | **0.688 ± 0.020** | 12.24 ± 0.43 | 0.679 ± 0.031 | **0.987 ± 0.001** |
| LSTM | **0.948 ± 0.001** | 0.656 ± 0.010 | **12.05 ± 0.36** | **0.684 ± 0.013** | 0.987 ± 0.001 |
| ST-GAT | 0.945 ± 0.003 | 0.667 ± 0.006 | 12.37 ± 0.26 | 0.679 ± 0.003 | 0.986 ± 0.001 |

### Current interpretation

The standard evaluation does not show statistically significant superiority of ST-GAT over the CNN or LSTM baselines.

ST-GAT does, however, show relatively low run-to-run variability in several metrics.

## 10. Topology ablation

A dedicated topology ablation evaluates whether the physical IEEE-14 connectivity itself improves the ST-GAT model.

The comparison is:

```text
ST-GAT + physical IEEE-14 topology
vs
ST-GAT + degree-preserving shuffled topology
```

The shuffled graph preserves:

- number of buses
- number of edges
- node degree sequence
- graph connectivity

Only the wiring is changed.

The same five training seeds are used.

### Current ablation result

The physical topology did not outperform the shuffled topology.

The shuffled graph achieved slightly better average results across the evaluated tasks, including line localization, position estimation, security classification and instability AUROC.

This indicates that the current ST-GAT implementation does not derive a measurable performance advantage from the physical IEEE-14 connectivity alone.

## 11. Robustness evaluation

Two stress tests are used.

### 11.1 Gaussian measurement noise

Noise levels:

```text
0.00
0.02
0.05
0.10
```

The aim is to measure performance degradation as measurement quality decreases.

### 11.2 Missing PMU observations

Missing-observation levels:

```text
0%
20%
40%
60%
```

The aim is to evaluate resilience under reduced grid observability.

### Current robustness finding

The strongest result appears under missing PMU observations.

At high levels of missing measurements, ST-GAT degrades less strongly than CNN and LSTM, particularly for:

- fault-type classification
- faulted-line localization
- position estimation
- security classification
- instability discrimination

This suggests that the graph-based model may be more useful under degraded observability than under clean measurement conditions.

The robustness evaluation is treated as a stress test rather than as a repeated statistical comparison across multiple seeds for every perturbation level.

## 12. Visualization and reporting

The project includes a Streamlit dashboard for interactive exploration of:

- IEEE-14 topology
- scenario selection
- model predictions
- faulted-line localization
- fault-position estimation
- dynamic-security classification
- model comparison
- robustness results

Final thesis figures are generated reproducibly with:

```bash
python scripts/generate_figures.py
```

The figure set includes:

1. IEEE-14 topology and observed generator buses
2. Baseline fault-line accuracy
3. Baseline position MAE
4. Physical vs shuffled topology ablation
5. Missing-PMU robustness for line localization
6. Missing-PMU robustness for fault-type classification
7. Gaussian-noise robustness for line localization

## 13. Main conclusions supported by the experiments

The completed experimental work currently supports the following conclusions:

1. **ST-GAT is competitive with CNN and LSTM but does not outperform them consistently under standard conditions.**
2. **The physical IEEE-14 topology does not provide a measurable advantage in the current ST-GAT implementation.**
3. **ST-GAT shows a meaningful robustness advantage when PMU observations are removed.**
4. **Temporal information alone is highly informative for several tasks in this dataset.**
5. **Graph-based modeling appears more promising for degraded-observability scenarios than for clean-data benchmark performance.**

## 14. Limitations

The main limitations are:

- only five generator buses provide directly observed dynamic measurements
- the dataset does not contain an explicit no-fault class
- topology ablation is performed on the current ST-GAT architecture rather than on multiple graph-model variants
- robustness tests are not repeated across multiple random seeds at every perturbation level
- the fault-line label mapping is dataset-specific
- results are based on a simulated IEEE-14 system rather than real field measurements

These limitations should be discussed explicitly rather than hidden.