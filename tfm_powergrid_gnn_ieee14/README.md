# PowerGrid FaultLab — IEEE-14 ST-GAT vs CNN/LSTM

Código reproducible para un TFM de **clasificación y localización espacio-temporal de faltas en redes eléctricas**, comparando una Graph Attention Network con message passing frente a CNN y LSTM bajo el mismo protocolo experimental.

## Dataset principal

**IEEE 14 bus test systems row data** — Figshare DOI `10.6084/m9.figshare.30590399`.

Coloca el Excel exactamente en:

```text
data/raw/ieee14_fault/IEEE 14 bus test system data.xlsx
```

El release inspeccionado contiene 21.120 filas y 230 columnas. Se trabaja únicamente con la hoja `row data`; la hoja ya normalizada no se usa para evitar leakage y para ajustar los scalers exclusivamente con entrenamiento.

Estructura relevante:

- `FCT`: fault clearing time, expresado en ciclos en este dataset.
- `FaultType`: 4 clases de fallo (IDs 1..4; no se inventan nombres físicos si el release no los documenta).
- `lname`: 16 IDs de líneas faulted.
- `POL`: posición del fallo, {21.4, 67.2, 91.7}%.
- 15 medidas pre-fallo: P/Q/V en cinco buses generadores.
- 210 features dinámicas = 14 bloques × 15 medidas (P/Q/V × 5 generadores).
- `output`: cinco estados de seguridad. Por coincidencia exacta con los conteos publicados en el artículo: 0=unstable, 1=urgent, 2=strong, 3=alarm, 4=normal.

**Limitación importante:** las 21.120 observaciones son escenarios de fallo. Este dataset permite clasificar el tipo y localizar línea/posición, y permite detectar *inestabilidad* (`unstable` vs resto), pero **no valida una tarea fault-vs-no-fault** porque no contiene una clase normal sin contingencia.

## Topología

El grafo utiliza la representación PowerFactory del IEEE-14: 14 buses, 16 objetos de línea faultables y 5 transformadores. Los buses generadores observados son 1, 2, 3, 6 y 8; los otros buses tienen features dinámicas a cero y una máscara `observed=0`.

El Excel publica `lname=1..16`, pero no incluye en sus columnas una tabla ID→extremos. `topology.py` incorpora el orden de referencia PowerFactory para visualización y message passing. Hasta verificar el mapping exacto con metadata del autor, los resultados académicos principales deben reportarse por **ID de línea**. El proyecto también reporta `line_corridor_accuracy`, que fusiona los dos circuitos paralelos 1–2 de la representación de referencia.

## Tensor de entrada

El preprocesado produce:

```text
X.shape = [samples, 14 dynamic blocks, 14 buses, 7 features]
```

por nodo:

```text
[dynamic_P, dynamic_Q, dynamic_V,
 prefault_P, prefault_Q, prefault_V,
 observed]
```

`FCT`, `FaultType`, `lname`, `POL` y `output` no se filtran accidentalmente como input. `FCT` está excluido por defecto para un experimento online más estricto; puede activarse como ablación en `configs/default.yaml`.

## Objetivos multi-task

Los tres modelos resuelven exactamente las mismas cuatro tareas:

1. `FaultType`: 4-way classification.
2. `lname`: 16-way fault-line localization.
3. `POL`: 3-way location classification + expected-position MAE en %.
4. `output`: 5-way dynamic-security classification.

Además se deriva `unstable vs non-unstable` para AUROC/F1, explícitamente etiquetado como **instability detection**, no fault detection.

## Modelos

### ST-GAT

Por cada bloque dinámico se aplican dos capas `GATv2Conv` sobre la topología física. Después, un GRU procesa la secuencia de embeddings por bus. Por defecto `lname` se clasifica desde el embedding global del grafo porque el release no documenta explícitamente el mapping ID→extremos. El proyecto incluye un **edge head** opcional que combina embeddings de los extremos de cada línea; actívalo solo cuando ese mapping haya sido verificado (`model.edge_localization_head: true`).

### CNN

CNN 1-D sobre la secuencia de 14 bloques, usando exactamente las mismas features pero sin `edge_index`.

### LSTM

LSTM temporal sobre el estado aplanado del sistema, sin conocimiento explícito de la topología.

## Split y anti-leakage

Las combinaciones `FCT × FaultType × lname × POL` son 1.920 y cada una aparece 11 veces, consistente con 11 condiciones de operación. El preprocesador intenta recuperar las 11 OCs por la firma P/Q/V pre-fallo. Si no encuentra exactamente 11 firmas, usa el rango dentro de cada combinación y lo registra en `metadata.json` para que sea auditable.

El split principal deja condiciones de operación completas fuera de entrenamiento (`8 train / 1 validation / 2 test` por defecto). Los scalers se ajustan **solo con train**. Esto es más exigente que un split aleatorio por filas y evita que el mismo operating point aparezca en train y test.

## Instalación

Python 3.11–3.13. En macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
pip install -e .
```

Comprueba el paquete:

```bash
python -c "import powergrid_faults; print('OK')"
```

## Flujo completo

```bash
python scripts/download_data.py
python scripts/inspect_data.py
python scripts/prepare_data.py
python scripts/train_all.py
streamlit run dashboard.py
```

`download_data.py` ya no intenta Kaggle: solo comprueba que el Excel de Figshare esté en la ruta esperada.

`prepare_data.py` debería mostrar algo parecido a:

```text
samples: 21120
shape: [samples, time, buses, features] = [21120, 14, 14, 7]
group detection: ... n_groups=11 ...
```

## Resultados generados

```text
artifacts/cnn/best.pt
artifacts/lstm/best.pt
artifacts/stgat/best.pt
reports/ieee14_schema_report.json
reports/comparison.csv
reports/robustness.csv
```

Para el estudio final con cinco semillas:

```bash
python scripts/run_repeated.py
```

que genera `reports/repeated_runs.csv` y `reports/significance.csv` con t-test pareado, Wilcoxon y Cohen's dz.

## Métricas principales

Para localización: line accuracy, macro-F1, Top-3, corridor accuracy, POL MAE/RMSE y accuracy within ±5/±10 puntos porcentuales. Para seguridad: macro-F1, balanced accuracy, ordinal MAE e instability AUROC/F1. También se comparan parámetros, latencia y robustez ante ruido y pérdida de PMUs.

## Experimentos que más valor añaden al TFM

- ST-GAT vs CNN vs LSTM con idénticos splits.
- Topología real vs `edge_index` aleatorizado.
- GATv2 vs GCN.
- Con/sin GRU.
- Con/sin `FCT` como feature.
- 0/20/40/60% de PMUs ausentes.
- Ruido gaussiano sobre las medidas.
- Split por OC (principal) vs split aleatorio (solo como réplica del protocolo del artículo).
- Exact line accuracy vs corridor accuracy para el doble circuito 1–2.

La hipótesis debe probarse, no asumirse: una GNN podría ganar especialmente en localización y observabilidad degradada, mientras LSTM puede ser muy competitivo en el objetivo global de seguridad dinámica.
