from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from powergrid_faults.topology import (
    case14_powerfactory_graph,
    topological_positions,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

MODEL_LABELS = {
    "cnn": "CNN",
    "lstm": "LSTM",
    "stgat": "ST-GAT",
}

TOPOLOGY_LABELS = {
    "real": "IEEE-14 topology",
    "shuffled": "Shuffled topology",
}


def _save(fig, name: str):
    """Save each thesis figure as both PNG and vector PDF."""
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {name}.png / {name}.pdf")


def _repeated_summary() -> pd.DataFrame:
    candidates = [
        REPORTS / "baseline" / "repeated_summary.csv",
        REPORTS / "repeated_summary.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError("Could not find repeated_summary.csv")


def figure_ieee14_topology():
    g, _, meta = case14_powerfactory_graph()
    pos = topological_positions(g)

    fig, ax = plt.subplots(figsize=(8, 6))

    line_edges = [
        (u, v)
        for u, v, d in g.edges(data=True)
        if d.get("kind") == "line"
    ]
    transformer_edges = [
        (u, v)
        for u, v, d in g.edges(data=True)
        if d.get("kind") == "transformer"
    ]

    nx.draw_networkx_edges(
        g,
        pos,
        edgelist=line_edges,
        width=1.6,
        ax=ax,
    )
    nx.draw_networkx_edges(
        g,
        pos,
        edgelist=transformer_edges,
        width=1.6,
        style="dashed",
        ax=ax,
    )

    generators = set(meta["generator_buses"])
    other_buses = [n for n in g.nodes if n not in generators]

    nx.draw_networkx_nodes(
        g,
        pos,
        nodelist=other_buses,
        node_size=750,
        node_shape="o",
        ax=ax,
    )
    nx.draw_networkx_nodes(
        g,
        pos,
        nodelist=sorted(generators),
        node_size=850,
        node_shape="s",
        ax=ax,
    )

    labels = {n: str(n + 1) for n in g.nodes}
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=9, ax=ax)

    ax.scatter([], [], marker="s", s=90, label="Observed generator / PMU bus")
    ax.scatter([], [], marker="o", s=90, label="Unobserved bus")
    ax.plot([], [], linestyle="-", label="Transmission line")
    ax.plot([], [], linestyle="--", label="Transformer")

    ax.set_title("IEEE-14 network and observed generator buses")
    ax.legend(frameon=False, loc="best")
    ax.axis("off")

    _save(fig, "01_ieee14_topology")


def figure_baseline_line_accuracy():
    df = _repeated_summary().copy()
    df["label"] = df["model"].map(MODEL_LABELS)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        df["label"],
        df["line_accuracy_mean"],
        yerr=df["line_accuracy_std"],
        capsize=5,
    )
    ax.set_ylabel("Fault-line accuracy")
    ax.set_xlabel("Model")
    ax.set_title("Fault-line localization across five independent runs")
    ax.set_ylim(0, 0.8)
    ax.grid(axis="y", alpha=0.25)

    _save(fig, "02_baseline_line_accuracy")


def figure_baseline_position_mae():
    df = _repeated_summary().copy()
    df["label"] = df["model"].map(MODEL_LABELS)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        df["label"],
        df["position_mae_expected_pct_mean"],
        yerr=df["position_mae_expected_pct_std"],
        capsize=5,
    )
    ax.set_ylabel("Position MAE (percentage points)")
    ax.set_xlabel("Model")
    ax.set_title("Fault-position estimation across five independent runs")
    ax.grid(axis="y", alpha=0.25)

    _save(fig, "03_baseline_position_mae")


def figure_topology_ablation():
    path = REPORTS / "topology_ablation_summary.csv"
    df = pd.read_csv(path).copy()
    df["label"] = df["topology"].map(TOPOLOGY_LABELS)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        df["label"],
        df["line_accuracy_mean"],
        yerr=df["line_accuracy_std"],
        capsize=5,
    )
    ax.set_ylabel("Fault-line accuracy")
    ax.set_xlabel("ST-GAT graph")
    ax.set_title("Topology ablation: physical vs shuffled connectivity")
    ax.set_ylim(0, 0.8)
    ax.grid(axis="y", alpha=0.25)

    _save(fig, "04_topology_ablation_line_accuracy")


def _robustness(perturbation: str) -> pd.DataFrame:
    path = REPORTS / "robustness.csv"
    df = pd.read_csv(path)
    out = df[df["perturbation"].eq(perturbation)].copy()
    if out.empty:
        raise ValueError(f"No rows found for perturbation={perturbation!r}")
    out["label"] = out["model"].map(MODEL_LABELS)
    return out


def _line_plot(df: pd.DataFrame, metric: str, ylabel: str, title: str, filename: str):
    fig, ax = plt.subplots(figsize=(7.5, 5))

    for model in ["cnn", "lstm", "stgat"]:
        part = df[df["model"].eq(model)].sort_values("level")
        ax.plot(
            part["level"],
            part[metric],
            marker="o",
            linewidth=2,
            label=MODEL_LABELS[model],
        )

    ax.set_xlabel("Perturbation level")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    _save(fig, filename)


def figure_missing_pmu_line_accuracy():
    df = _robustness("missing_pmus")
    _line_plot(
        df,
        metric="line_accuracy",
        ylabel="Fault-line accuracy",
        title="Fault-line localization under missing PMU observations",
        filename="05_missing_pmus_line_accuracy",
    )


def figure_missing_pmu_fault_type():
    df = _robustness("missing_pmus")
    _line_plot(
        df,
        metric="fault_type_macro_f1",
        ylabel="Fault-type Macro-F1",
        title="Fault classification under missing PMU observations",
        filename="06_missing_pmus_fault_type_f1",
    )


def figure_noise_line_accuracy():
    df = _robustness("gaussian_noise")
    _line_plot(
        df,
        metric="line_accuracy",
        ylabel="Fault-line accuracy",
        title="Fault-line localization under Gaussian measurement noise",
        filename="07_gaussian_noise_line_accuracy",
    )


def main():
    print(f"Writing figures to: {FIGURES}")
    figure_ieee14_topology()
    figure_baseline_line_accuracy()
    figure_baseline_position_mae()
    figure_topology_ablation()
    figure_missing_pmu_line_accuracy()
    figure_missing_pmu_fault_type()
    figure_noise_line_accuracy()

    print("\nDone.")
    print("Use the PDF files in the thesis when possible; they are vector graphics.")
    print("Use the PNG files for README, slides or quick previews.")


if __name__ == "__main__":
    main()