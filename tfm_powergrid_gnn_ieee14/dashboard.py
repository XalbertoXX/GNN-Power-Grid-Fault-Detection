from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch

from powergrid_faults.data import IEEE14Dataset, POSITION_VALUES
from powergrid_faults.models import build_model
from powergrid_faults.topology import case14_powerfactory_graph
from powergrid_faults.utils import get_device
from powergrid_faults.viz import topology_figure

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data/processed_ieee14"
SECURITY = ["unstable", "urgent", "strong", "alarm", "normal"]

MODEL_INFO = {
    "stgat": {
        "label": "ST-GAT",
        "icon": "🕸️",
        "summary": "Modelo espacio-temporal consciente de la topología eléctrica.",
        "detail": (
            "Aplica GATv2 para intercambiar información entre buses conectados y una GRU para "
            "integrar los 14 bloques dinámicos. Es el modelo principal de la hipótesis del TFM."
        ),
        "topology": "Sí — usa explícitamente el grafo IEEE-14",
        "temporal": "Sí — GRU sobre la evolución dinámica",
        "role": "Modelo propuesto / candidato a demostrar la ventaja del grafo",
    },
    "cnn": {
        "label": "CNN temporal",
        "icon": "📈",
        "summary": "Baseline convolucional que busca patrones locales en la secuencia.",
        "detail": (
            "Procesa las mismas medidas que la ST-GAT mediante convoluciones temporales, pero no conoce "
            "qué buses están conectados físicamente. Sirve para aislar el valor añadido de la topología."
        ),
        "topology": "No — trata las señales sin adjacency matrix",
        "temporal": "Sí — convoluciones 1D",
        "role": "Baseline de deep learning convolucional",
    },
    "lstm": {
        "label": "LSTM",
        "icon": "🔁",
        "summary": "Baseline recurrente centrado en dependencias temporales.",
        "detail": (
            "Modela la evolución de las medidas como una secuencia recurrente. Puede aprender dependencias "
            "de largo alcance, pero tampoco recibe explícitamente la topología IEEE-14."
        ),
        "topology": "No — no usa conexiones físicas entre buses",
        "temporal": "Sí — memoria recurrente LSTM",
        "role": "Baseline secuencial",
    },
}

SPLIT_INFO = {
    "test": "Evaluación final sobre condiciones de operación no usadas para ajustar el modelo.",
    "val": "Validación usada durante el desarrollo/early stopping; útil para diagnóstico.",
    "train": "Datos vistos durante entrenamiento. No deben usarse para afirmar rendimiento final.",
}

FRIENDLY_METRICS = {
    "line_accuracy": "Exactitud de línea",
    "line_top3": "Línea Top-3",
    "line_macro_f1": "F1 macro de línea",
    "fault_type_macro_f1": "F1 macro del tipo de fallo",
    "position_mae_expected_pct": "Error medio de posición (%)",
    "security_macro_f1": "F1 macro de seguridad",
    "instability_auroc": "AUROC de inestabilidad",
    "latency_ms_per_sample": "Latencia (ms/muestra)",
}

st.set_page_config(page_title="PowerGrid FaultLab — IEEE-14", layout="wide", page_icon="⚡")
st.title("⚡ PowerGrid FaultLab — IEEE-14")
st.caption(
    "Comparador ST-GAT vs CNN/LSTM · clasificación de fallo · localización de línea y posición · seguridad dinámica"
)

if not (PROCESSED / "dataset.npz").exists():
    st.error("Dataset procesado no encontrado. Ejecuta: python scripts/prepare_data.py")
    st.stop()

models = [m for m in ["stgat", "cnn", "lstm"] if (ROOT / "artifacts" / m / "best.pt").exists()]
if not models:
    st.error("No hay modelos entrenados. Ejecuta: python scripts/train_all.py")
    st.stop()

meta = json.loads((PROCESSED / "metadata.json").read_text(encoding="utf-8"))
cmp = ROOT / "reports/comparison.csv"
benchmark_df = pd.read_csv(cmp) if cmp.exists() else None

# -----------------------------------------------------------------------------
# Sidebar: model + scenario guide
# -----------------------------------------------------------------------------
st.sidebar.title("⚡ FaultLab")
st.sidebar.caption("Explorador interactivo del experimento IEEE-14")
st.sidebar.divider()

st.sidebar.subheader("1 · Modelo")
model_name = st.sidebar.selectbox(
    "Selecciona arquitectura",
    models,
    format_func=lambda m: f"{MODEL_INFO[m]['icon']} {MODEL_INFO[m]['label']}",
    help="Los tres modelos reciben los mismos escenarios y se evalúan con el mismo protocolo.",
)
info = MODEL_INFO[model_name]
st.sidebar.markdown(f"### {info['icon']} {info['label']}")
st.sidebar.write(info["summary"])
st.sidebar.caption(info["detail"])
st.sidebar.markdown(
    f"**Topología:** {info['topology']}  \n"
    f"**Temporalidad:** {info['temporal']}  \n"
    f"**Rol:** {info['role']}"
)

if benchmark_df is not None and "model" in benchmark_df.columns:
    selected_rows = benchmark_df.loc[benchmark_df["model"].eq(model_name)]
    if not selected_rows.empty:
        row = selected_rows.iloc[0]
        with st.sidebar.expander("📊 Rendimiento global en test", expanded=False):
            for key in ["line_accuracy", "line_top3", "position_mae_expected_pct", "security_macro_f1", "instability_auroc"]:
                if key in row.index and pd.notna(row[key]):
                    value = float(row[key])
                    label = FRIENDLY_METRICS.get(key, key)
                    if key == "position_mae_expected_pct":
                        st.metric(label, f"{value:.2f} pp")
                    elif 0.0 <= value <= 1.0:
                        st.metric(label, f"{value:.1%}")
                    else:
                        st.metric(label, f"{value:.3f}")

st.sidebar.divider()
st.sidebar.subheader("2 · Datos")
split = st.sidebar.selectbox(
    "Partición",
    ["test", "val", "train"],
    format_func=lambda s: {"test": "🧪 Test", "val": "🛠️ Validación", "train": "🎓 Entrenamiento"}[s],
    help="Para presentar resultados finales, utiliza Test.",
)
st.sidebar.caption(SPLIT_INFO[split])

ds = IEEE14Dataset(PROCESSED / "dataset.npz", split)
idx = st.sidebar.slider(
    "Escenario",
    0,
    len(ds) - 1,
    0,
    help="Cambia de escenario para inspeccionar cómo responde el modelo a distintas faltas y condiciones de operación.",
)
item = ds[idx]

with st.sidebar.expander("🔎 Contexto del escenario", expanded=True):
    st.write(f"**Índice:** {idx:,}")
    st.write(f"**Operating condition:** grupo {int(item['oc_group'])}")
    st.write(f"**FCT:** {float(item['fct']):.4f} ciclos")
    st.write(f"**Split:** {split}")

with st.sidebar.expander("🧭 Cómo leer la pantalla", expanded=False):
    st.markdown(
        "- **Mapa de red:** probabilidad asignada a cada línea.\n"
        "- **Position:** localización porcentual dentro de la línea.\n"
        "- **Security:** estado dinámico estimado tras la contingencia.\n"
        "- **P(unstable):** probabilidad de inestabilidad.\n"
        "- **Benchmark:** compara las arquitecturas con la misma métrica.\n"
        "- **Robustez:** muestra cuánto cae el rendimiento con ruido o PMUs ausentes."
    )

with st.sidebar.expander("ℹ️ Alcance del dataset", expanded=False):
    st.caption(
        "El dataset contiene escenarios de fallo. La tarea binaria mostrada es inestabilidad vs no-inestabilidad; "
        "no debe interpretarse como fault vs no-fault. El mapeo lname→extremos físicos debe verificarse antes "
        "de presentar conclusiones por nombre de corredor."
    )

if "verify" in meta.get("line_mapping_status", "").lower():
    st.warning(
        "El Excel publica lname=1..16, pero no documenta explícitamente en el propio fichero el mapeo "
        "ID→extremos de línea. El diagrama usa el orden de referencia PowerFactory. Usa las métricas por ID "
        "como resultado principal hasta verificar ese mapeo con metadata del autor."
    )

device = get_device("auto")
ckpt = torch.load(ROOT / "artifacts" / model_name / "best.pt", map_location=device)
model = build_model(model_name, **ckpt["kwargs"]).to(device)
model.load_state_dict(ckpt["state_dict"])
model.eval()
_, edge_index, graph_meta = case14_powerfactory_graph()
edge_index = edge_index.to(device)

parameter_count = int(sum(p.numel() for p in model.parameters()))
st.sidebar.caption(f"Parámetros del modelo cargado: {parameter_count:,} · dispositivo: {device}")

x = item["x"].unsqueeze(0).to(device)
with torch.no_grad():
    out = model(x, edge_index, return_attention=(model_name == "stgat"))

ptype = torch.softmax(out["fault_type"], -1)[0].cpu().numpy()
pline = torch.softmax(out["line"], -1)[0].cpu().numpy()
pos = torch.softmax(out["position"], -1)[0].cpu().numpy()
psec = torch.softmax(out["security"], -1)[0].cpu().numpy()

pred_type = int(ptype.argmax())
pred_line = int(pline.argmax())
pred_pos_class = int(pos.argmax())
pred_pol = float(pos @ POSITION_VALUES)
pred_security = int(psec.argmax())
true_type = int(item["y_type"])
true_line = int(item["y_line"])
true_pol = float(item["y_pol"])
true_security = int(item["y_security"])

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Fault type", f"Type {pred_type + 1}", delta=f"real Type {true_type + 1}")
c2.metric("Fault line", f"{pred_line + 1}", delta=f"real {true_line + 1}")
c3.metric("Position", f"{pred_pol:.1f}%", delta=f"real {true_pol:.1f}%")
c4.metric("Security", SECURITY[pred_security], delta=f"real {SECURITY[true_security]}")
c5.metric("P(unstable)", f"{psec[0]:.1%}")

left, right = st.columns([1.45, 1.0])
with left:
    st.plotly_chart(
        topology_figure(
            pline,
            true_line=true_line,
            pred_line=pred_line,
            true_pol=true_pol,
            pred_pol=pred_pol,
            attention=out.get("attention"),
        ),
        width="stretch",
    )
with right:
    line_df = pd.DataFrame({
        "line": np.arange(1, 17),
        "reference": graph_meta["line_names"],
        "probability": pline,
    }).nlargest(8, "probability")
    st.plotly_chart(
        px.bar(line_df, x="line", y="probability", hover_data=["reference"], title="Top candidate lines"),
        width="stretch",
    )
    pos_df = pd.DataFrame({"POL": POSITION_VALUES, "probability": pos})
    st.plotly_chart(
        px.bar(pos_df, x="POL", y="probability", title="Fault-position probabilities"),
        width="stretch",
    )

st.subheader("Fault-on dynamic feature blocks")
feature = st.selectbox("Variable", ["P", "Q", "V"])
gen_bus = st.selectbox("Bus generador observado", meta["generator_buses_1based"])
generator_buses = meta["generator_buses_1based"]
bus_idx = int(gen_bus) - 1
fidx = {"P": 0, "Q": 1, "V": 2}[feature]
series = item["x"][:, bus_idx, fidx].numpy()
wave = pd.DataFrame({"dynamic_block": np.arange(1, len(series) + 1), "value_standardized": series})
st.plotly_chart(
    px.line(
        wave,
        x="dynamic_block",
        y="value_standardized",
        markers=True,
        title=f"{feature} · Bus {gen_bus}",
    ),
    width="stretch",
)
st.caption(
    "Los 14 pasos mostrados son los 14 bloques dinámicos/derivativos publicados en el Excel; "
    "no se reinterpretan como muestras uniformes de una señal cruda."
)

p1, p2 = st.columns(2)
with p1:
    tdf = pd.DataFrame({"fault_type": [f"Type {i}" for i in range(1, 5)], "probability": ptype})
    st.plotly_chart(
        px.bar(tdf, x="fault_type", y="probability", title="Fault-type probabilities"),
        width="stretch",
    )
with p2:
    sdf = pd.DataFrame({"security": SECURITY, "probability": psec})
    st.plotly_chart(
        px.bar(sdf, x="security", y="probability", title="Dynamic-security probabilities"),
        width="stretch",
    )

if benchmark_df is not None:
    st.subheader("Benchmark")
    df = benchmark_df
    numeric = [c for c in df.columns if c != "model" and pd.api.types.is_numeric_dtype(df[c])]
    metric = st.selectbox(
        "Métrica",
        numeric,
        index=numeric.index("line_accuracy") if "line_accuracy" in numeric else 0,
        format_func=lambda m: FRIENDLY_METRICS.get(m, m.replace("_", " ").title()),
    )
    st.plotly_chart(px.bar(df, x="model", y=metric, text_auto=".3f", title=FRIENDLY_METRICS.get(metric, metric)), width="stretch")
    st.dataframe(df, width="stretch", hide_index=True)

rob = ROOT / "reports/robustness.csv"
if rob.exists():
    st.subheader("Robustez")
    rdf = pd.read_csv(rob)
    candidates = [
        c
        for c in rdf.columns
        if c not in {"model", "perturbation", "level"} and pd.api.types.is_numeric_dtype(rdf[c])
    ]
    rmetric = st.selectbox(
        "Métrica de robustez",
        candidates,
        index=candidates.index("line_accuracy") if "line_accuracy" in candidates else 0,
        format_func=lambda m: FRIENDLY_METRICS.get(m, m.replace("_", " ").title()),
    )
    st.plotly_chart(
        px.line(
            rdf,
            x="level",
            y=rmetric,
            color="model",
            facet_col="perturbation",
            markers=True,
            title=f"Robustez · {FRIENDLY_METRICS.get(rmetric, rmetric)}",
        ),
        width="stretch",
    )
