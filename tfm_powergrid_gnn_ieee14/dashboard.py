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

st.set_page_config(page_title="PowerGrid FaultLab — IEEE-14", layout="wide", page_icon="⚡")
st.title("⚡ PowerGrid FaultLab — IEEE-14")
st.caption("ST-GAT vs CNN/LSTM · clasificación, localización de línea/posición y seguridad dinámica")

if not (PROCESSED / "dataset.npz").exists():
    st.error("Dataset procesado no encontrado. Ejecuta: python scripts/prepare_data.py")
    st.stop()

models = [m for m in ["stgat", "cnn", "lstm"] if (ROOT / "artifacts" / m / "best.pt").exists()]
if not models:
    st.error("No hay modelos entrenados. Ejecuta: python scripts/train_all.py")
    st.stop()

meta = json.loads((PROCESSED / "metadata.json").read_text(encoding="utf-8"))
model_name = st.sidebar.selectbox("Modelo", models)
split = st.sidebar.selectbox("Split", ["test", "val", "train"])
ds = IEEE14Dataset(PROCESSED / "dataset.npz", split)
idx = st.sidebar.slider("Escenario", 0, len(ds) - 1, 0)
item = ds[idx]

st.sidebar.caption(f"OC group: {int(item['oc_group'])} · FCT: {float(item['fct']):.4f} cycles")

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
            pline, true_line=true_line, pred_line=pred_line,
            true_pol=true_pol, pred_pol=pred_pol,
            attention=out.get("attention"),
        ),
        use_container_width=True,
    )
with right:
    line_df = pd.DataFrame({
        "line": np.arange(1, 17),
        "reference": graph_meta["line_names"],
        "probability": pline,
    }).nlargest(8, "probability")
    st.plotly_chart(
        px.bar(line_df, x="line", y="probability", hover_data=["reference"], title="Top candidate lines"),
        use_container_width=True,
    )
    pos_df = pd.DataFrame({"POL": POSITION_VALUES, "probability": pos})
    st.plotly_chart(px.bar(pos_df, x="POL", y="probability", title="Fault-position probabilities"), use_container_width=True)

st.subheader("Fault-on dynamic feature blocks")
feature = st.selectbox("Variable", ["P", "Q", "V"])
gen_bus = st.selectbox("Bus generador observado", meta["generator_buses_1based"])
generator_buses = meta["generator_buses_1based"]
bus_idx = int(gen_bus) - 1
fidx = {"P": 0, "Q": 1, "V": 2}[feature]
series = item["x"][:, bus_idx, fidx].numpy()
wave = pd.DataFrame({"dynamic_block": np.arange(1, len(series) + 1), "value_standardized": series})
st.plotly_chart(px.line(wave, x="dynamic_block", y="value_standardized", markers=True,
                        title=f"{feature} · Bus {gen_bus}"), use_container_width=True)
st.caption(
    "Los 14 pasos mostrados son los 14 bloques dinámicos/derivativos publicados en el Excel; "
    "no se reinterpretan como muestras uniformes de una señal cruda."
)

p1, p2 = st.columns(2)
with p1:
    tdf = pd.DataFrame({"fault_type": [f"Type {i}" for i in range(1, 5)], "probability": ptype})
    st.plotly_chart(px.bar(tdf, x="fault_type", y="probability", title="Fault-type probabilities"), use_container_width=True)
with p2:
    sdf = pd.DataFrame({"security": SECURITY, "probability": psec})
    st.plotly_chart(px.bar(sdf, x="security", y="probability", title="Dynamic-security probabilities"), use_container_width=True)

cmp = ROOT / "reports/comparison.csv"
if cmp.exists():
    st.subheader("Benchmark")
    df = pd.read_csv(cmp)
    numeric = [c for c in df.columns if c != "model" and pd.api.types.is_numeric_dtype(df[c])]
    metric = st.selectbox("Métrica", numeric, index=numeric.index("line_accuracy") if "line_accuracy" in numeric else 0)
    st.plotly_chart(px.bar(df, x="model", y=metric, text_auto=".3f", title=metric), use_container_width=True)
    st.dataframe(df, use_container_width=True)

rob = ROOT / "reports/robustness.csv"
if rob.exists():
    st.subheader("Robustez")
    rdf = pd.read_csv(rob)
    candidates = [c for c in rdf.columns if c not in {"model", "perturbation", "level"} and pd.api.types.is_numeric_dtype(rdf[c])]
    rmetric = st.selectbox("Métrica de robustez", candidates, index=candidates.index("line_accuracy") if "line_accuracy" in candidates else 0)
    st.plotly_chart(px.line(rdf, x="level", y=rmetric, color="model", facet_col="perturbation", markers=True), use_container_width=True)
