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
SECURITY = ["Unstable", "Urgent", "Strong", "Alarm", "Normal"]

MODEL_INFO = {
    "stgat": {
        "label": "ST-GAT",
        "icon": "🕸️",
        "summary": "Topology-aware spatio-temporal model",
        "detail": (
            "GATv2 exchanges information between electrically connected buses, while a GRU integrates "
            "the 14 dynamic blocks over time. This is the proposed model used to test whether explicit "
            "grid structure improves fault localization."
        ),
    },
    "cnn": {
        "label": "Temporal CNN",
        "icon": "📈",
        "summary": "Convolutional temporal baseline",
        "detail": (
            "Processes the same measurements with 1-D temporal convolutions. It can learn local temporal "
            "patterns, but it does not receive the IEEE-14 adjacency structure explicitly."
        ),
    },
    "lstm": {
        "label": "LSTM",
        "icon": "🔁",
        "summary": "Recurrent temporal baseline",
        "detail": (
            "Models the measurement sequence with recurrent memory. It captures temporal dependencies, "
            "but it does not explicitly know which buses are physically connected."
        ),
    },
}

SPLIT_INFO = {
    "test": "Final evaluation on operating conditions that were not used to fit the model.",
    "val": "Validation data used during model development and early stopping.",
    "train": "Training data seen by the model. Do not use it to report final performance.",
}

FRIENDLY_METRICS = {
    "line_accuracy": "Fault-line accuracy",
    "line_top3": "Fault-line Top-3 accuracy",
    "line_macro_f1": "Fault-line macro F1",
    "line_corridor_accuracy": "Fault-corridor accuracy",
    "fault_type_macro_f1": "Fault-type macro F1",
    "position_mae_expected_pct": "Mean position error (%)",
    "position_rmse_expected_pct": "Position RMSE (%)",
    "position_within_5pct": "Position within ±5 pp",
    "position_within_10pct": "Position within ±10 pp",
    "security_macro_f1": "Dynamic-security macro F1",
    "security_balanced_accuracy": "Dynamic-security balanced accuracy",
    "security_ordinal_mae": "Dynamic-security ordinal MAE",
    "instability_auroc": "Instability AUROC",
    "instability_f1": "Instability F1",
    "latency_ms_per_sample": "Latency (ms/sample)",
    "parameters": "Parameters",
}

DISPLAY_MODEL = {key: value["label"] for key, value in MODEL_INFO.items()}

st.set_page_config(page_title="FaultLab — IEEE-14", layout="wide", page_icon="⚡")
st.title("FaultLab")
st.caption(
    "IEEE-14 power-grid experiment · ST-GAT vs CNN/LSTM · fault classification · "
    "line and position localization · dynamic-security assessment"
)

if not (PROCESSED / "dataset.npz").exists():
    st.error("Processed dataset not found. Run: python scripts/prepare_data.py")
    st.stop()

models = [m for m in ["stgat", "cnn", "lstm"] if (ROOT / "artifacts" / m / "best.pt").exists()]
if not models:
    st.error("No trained models were found. Run: python scripts/train_all.py")
    st.stop()

meta = json.loads((PROCESSED / "metadata.json").read_text(encoding="utf-8"))
cmp = ROOT / "reports/comparison.csv"
benchmark_df = pd.read_csv(cmp) if cmp.exists() else None

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.title("FaultLab")
st.sidebar.caption("IEEE-14 experiment explorer")
st.sidebar.divider()

st.sidebar.subheader("1 · Model")
model_name = st.sidebar.selectbox(
    "Model architecture",
    models,
    format_func=lambda m: f"{MODEL_INFO[m]['icon']} {MODEL_INFO[m]['label']}",
    help="All models receive the same scenarios and are evaluated with the same protocol.",
    label_visibility="collapsed",
)
info = MODEL_INFO[model_name]
st.sidebar.markdown(f"**{info['summary']}**")
st.sidebar.caption(info["detail"])

with st.sidebar.expander("Model comparison context", expanded=False):
    if model_name == "stgat":
        st.write(
            "ST-GAT is the topology-aware model. Its key difference is that it performs message passing "
            "over the IEEE-14 graph before temporal aggregation."
        )
    elif model_name == "cnn":
        st.write(
            "The CNN is a topology-agnostic baseline. It tests how far temporal convolutions can go when "
            "the physical grid graph is not provided explicitly."
        )
    else:
        st.write(
            "The LSTM is a topology-agnostic recurrent baseline. It tests whether recurrent temporal memory "
            "alone can match the graph-aware model."
        )

if benchmark_df is not None and "model" in benchmark_df.columns:
    selected_rows = benchmark_df.loc[benchmark_df["model"].eq(model_name)]
    if not selected_rows.empty:
        row = selected_rows.iloc[0]
        with st.sidebar.expander("Test-set performance", expanded=False):
            for key in [
                "line_accuracy",
                "line_top3",
                "position_mae_expected_pct",
                "security_macro_f1",
                "instability_auroc",
            ]:
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
st.sidebar.subheader("2 · Scenario")
split = st.sidebar.selectbox(
    "Dataset split",
    ["test", "val", "train"],
    format_func=lambda s: {"test": "🧪 Test", "val": "🛠️ Validation", "train": "🎓 Training"}[s],
    help="Use the test split when presenting final model performance.",
)
st.sidebar.caption(SPLIT_INFO[split])

ds = IEEE14Dataset(PROCESSED / "dataset.npz", split)
idx = st.sidebar.slider(
    "Scenario index",
    0,
    len(ds) - 1,
    0,
    help="Move between scenarios to inspect different faults and operating conditions.",
)
item = ds[idx]

with st.sidebar.expander("Scenario context", expanded=True):
    st.write(f"**Index:** {idx:,}")
    st.write(f"**Operating condition:** group {int(item['oc_group'])}")
    st.write(f"**Fault clearing time:** {float(item['fct']):.4f} cycles")
    st.write(f"**Split:** {split.title()}")

with st.sidebar.expander("How to read the dashboard", expanded=False):
    st.markdown(
        "- **Network map:** probability assigned to each candidate fault line.\n"
        "- **Position:** estimated percentage along the faulted line.\n"
        "- **Dynamic security:** predicted post-contingency operating state.\n"
        "- **P(unstable):** predicted probability of transient instability.\n"
        "- **Benchmark:** side-by-side comparison of the three architectures.\n"
        "- **Robustness:** performance degradation under noise or missing PMUs."
    )

with st.sidebar.expander("Dataset notes", expanded=False):
    st.caption(
        "The IEEE-14 dataset contains fault scenarios. Therefore, the binary score shown here is "
        "instability vs non-instability, not fault vs no-fault."
    )
    if "verify" in meta.get("line_mapping_status", "").lower():
        st.caption(
            "The Excel file provides fault-line IDs (lname=1..16) but does not contain an explicit "
            "ID-to-line-endpoint table. The network visualization uses the reference PowerFactory ordering. "
            "Report line-ID metrics as the primary result until this mapping is independently verified."
        )

device = get_device("auto")
ckpt = torch.load(ROOT / "artifacts" / model_name / "best.pt", map_location=device)
model = build_model(model_name, **ckpt["kwargs"]).to(device)
model.load_state_dict(ckpt["state_dict"])
model.eval()
_, edge_index, graph_meta = case14_powerfactory_graph()
edge_index = edge_index.to(device)

parameter_count = int(sum(p.numel() for p in model.parameters()))
st.sidebar.caption(f"Loaded model · {parameter_count:,} parameters · device: {device}")

# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------
x = item["x"].unsqueeze(0).to(device)
with torch.no_grad():
    out = model(x, edge_index, return_attention=(model_name == "stgat"))

ptype = torch.softmax(out["fault_type"], -1)[0].cpu().numpy()
pline = torch.softmax(out["line"], -1)[0].cpu().numpy()
pos = torch.softmax(out["position"], -1)[0].cpu().numpy()
psec = torch.softmax(out["security"], -1)[0].cpu().numpy()

pred_type = int(ptype.argmax())
pred_line = int(pline.argmax())
pred_pol = float(pos @ POSITION_VALUES)
pred_security = int(psec.argmax())
true_type = int(item["y_type"])
true_line = int(item["y_line"])
true_pol = float(item["y_pol"])
true_security = int(item["y_security"])

# -----------------------------------------------------------------------------
# Main summary
# -----------------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Fault type", f"Type {pred_type + 1}", delta=f"actual Type {true_type + 1}")
c2.metric("Fault line", f"{pred_line + 1}", delta=f"actual {true_line + 1}")
c3.metric("Position", f"{pred_pol:.1f}%", delta=f"actual {true_pol:.1f}%")
c4.metric("Dynamic security", SECURITY[pred_security], delta=f"actual {SECURITY[true_security]}")
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
    line_df = pd.DataFrame(
        {
            "Line": np.arange(1, 17),
            "Reference": graph_meta["line_names"],
            "Probability": pline,
        }
    ).nlargest(8, "Probability")
    st.plotly_chart(
        px.bar(
            line_df,
            x="Line",
            y="Probability",
            hover_data=["Reference"],
            title="Top candidate fault lines",
        ),
        width="stretch",
    )

    pos_df = pd.DataFrame({"Position (%)": POSITION_VALUES, "Probability": pos})
    st.plotly_chart(
        px.bar(pos_df, x="Position (%)", y="Probability", title="Fault-position probabilities"),
        width="stretch",
    )

st.subheader("Fault-on dynamic measurements")
feature = st.selectbox("Electrical variable", ["P", "Q", "V"])
gen_bus = st.selectbox("Observed generator bus", meta["generator_buses_1based"])
bus_idx = int(gen_bus) - 1
fidx = {"P": 0, "Q": 1, "V": 2}[feature]
series = item["x"][:, bus_idx, fidx].numpy()
wave = pd.DataFrame({"Dynamic block": np.arange(1, len(series) + 1), "Standardized value": series})
st.plotly_chart(
    px.line(
        wave,
        x="Dynamic block",
        y="Standardized value",
        markers=True,
        title=f"{feature} · Bus {gen_bus}",
    ),
    width="stretch",
)
st.caption(
    "The 14 points correspond to the 14 dynamic/derivative blocks published in the Excel dataset. "
    "They are not reinterpreted as uniformly sampled raw waveform points."
)

p1, p2 = st.columns(2)
with p1:
    tdf = pd.DataFrame({"Fault type": [f"Type {i}" for i in range(1, 5)], "Probability": ptype})
    st.plotly_chart(
        px.bar(tdf, x="Fault type", y="Probability", title="Fault-type probabilities"),
        width="stretch",
    )
with p2:
    sdf = pd.DataFrame({"Security state": SECURITY, "Probability": psec})
    st.plotly_chart(
        px.bar(sdf, x="Security state", y="Probability", title="Dynamic-security probabilities"),
        width="stretch",
    )

# -----------------------------------------------------------------------------
# Global comparison
# -----------------------------------------------------------------------------
if benchmark_df is not None:
    st.subheader("Model benchmark")
    df = benchmark_df.copy()
    numeric = [c for c in df.columns if c != "model" and pd.api.types.is_numeric_dtype(df[c])]
    metric = st.selectbox(
        "Benchmark metric",
        numeric,
        index=numeric.index("line_accuracy") if "line_accuracy" in numeric else 0,
        format_func=lambda m: FRIENDLY_METRICS.get(m, m.replace("_", " ").title()),
    )
    df["Model"] = df["model"].map(DISPLAY_MODEL).fillna(df["model"])
    st.plotly_chart(
        px.bar(
            df,
            x="Model",
            y=metric,
            text_auto=".3f",
            title=FRIENDLY_METRICS.get(metric, metric.replace("_", " ").title()),
        ),
        width="stretch",
    )
    display_df = df.drop(columns=["model"])
    st.dataframe(display_df, width="stretch", hide_index=True)

# -----------------------------------------------------------------------------
# Robustness
# -----------------------------------------------------------------------------
rob = ROOT / "reports/robustness.csv"
if rob.exists():
    st.subheader("Robustness analysis")
    rdf = pd.read_csv(rob)
    candidates = [
        c
        for c in rdf.columns
        if c not in {"model", "perturbation", "level"} and pd.api.types.is_numeric_dtype(rdf[c])
    ]
    rmetric = st.selectbox(
        "Robustness metric",
        candidates,
        index=candidates.index("line_accuracy") if "line_accuracy" in candidates else 0,
        format_func=lambda m: FRIENDLY_METRICS.get(m, m.replace("_", " ").title()),
    )
    rdf = rdf.copy()
    rdf["Model"] = rdf["model"].map(DISPLAY_MODEL).fillna(rdf["model"])
    rdf["Perturbation"] = rdf["perturbation"].astype(str).str.replace("_", " ").str.title()
    st.plotly_chart(
        px.line(
            rdf,
            x="level",
            y=rmetric,
            color="Model",
            facet_col="Perturbation",
            markers=True,
            title=f"Robustness · {FRIENDLY_METRICS.get(rmetric, rmetric.replace('_', ' ').title())}",
        ),
        width="stretch",
    )
