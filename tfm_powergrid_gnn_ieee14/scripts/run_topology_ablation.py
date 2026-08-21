from __future__ import annotations

from pathlib import Path
import copy
import json

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

from powergrid_faults.topology import degree_preserving_shuffled_edge_index
from powergrid_faults.trainlib import train_one
from powergrid_faults.utils import load_config

ROOT = Path(__file__).resolve().parents[1]
cfg0 = load_config(ROOT / "configs/default.yaml")

SEEDS = [11, 22, 33, 44, 55]
SHUFFLE_SEED = 20260821
N_SWAPS = 250

METRICS = [
    "fault_type_macro_f1",
    "line_accuracy",
    "position_mae_expected_pct",
    "security_macro_f1",
    "instability_auroc",
]

# Higher is better except for position MAE.
HIGHER_IS_BETTER = {
    "fault_type_macro_f1": True,
    "line_accuracy": True,
    "position_mae_expected_pct": False,
    "security_macro_f1": True,
    "instability_auroc": True,
}

reports = ROOT / "reports"
reports.mkdir(exist_ok=True)

# Reuse the already-computed five-seed ST-GAT results with the real topology.
baseline_candidates = [
    reports / "baseline" / "repeated_runs.csv",
    reports / "repeated_runs.csv",
]
baseline_path = next((p for p in baseline_candidates if p.exists()), None)
if baseline_path is None:
    raise FileNotFoundError(
        "Could not find baseline repeated_runs.csv. Expected either "
        "reports/baseline/repeated_runs.csv or reports/repeated_runs.csv"
    )

baseline = pd.read_csv(baseline_path)
required = {"seed", "model", *METRICS}
missing = sorted(required - set(baseline.columns))
if missing:
    raise ValueError(f"Baseline repeated_runs.csv is missing columns: {missing}")

real = (
    baseline[
        baseline["model"].eq("stgat")
        & baseline["seed"].isin(SEEDS)
    ][["seed", *METRICS]]
    .sort_values("seed")
    .copy()
)

if real["seed"].tolist() != SEEDS:
    raise ValueError(
        "Expected real-topology ST-GAT results for seeds "
        f"{SEEDS}, found {real['seed'].tolist()}"
    )

real.insert(1, "topology", "real")

# One fixed shuffled topology is used for every training seed.
# This isolates topology from model-initialization randomness.
shuffled_edge_index, graph_meta = degree_preserving_shuffled_edge_index(
    seed=SHUFFLE_SEED,
    n_swaps=N_SWAPS,
)

(reports / "topology_ablation_graph.json").write_text(
    json.dumps(graph_meta, indent=2),
    encoding="utf-8",
)

print("Topology ablation")
print("-----------------")
print(f"Baseline: {baseline_path}")
print(f"Training seeds: {SEEDS}")
print(f"Shuffle seed: {SHUFFLE_SEED}")
print(f"Physical edge objects: {graph_meta['n_undirected_edge_objects']}")
print(f"Directed GAT edges: {graph_meta['n_directed_edges']}")
print(f"Connected: {graph_meta['connected']}")
print(
    "Real/shuffled edge overlap: "
    f"{graph_meta['edge_object_overlap_fraction']:.1%}"
)
print("Degree sequence preserved:",
      graph_meta["degree_sequence_real"] == graph_meta["degree_sequence_shuffled"])
print()

artifact_root = ROOT / "artifacts_topology_ablation"
artifact_root.mkdir(exist_ok=True)

shuffled_rows = []

# Run ST-GAT with the shuffled topology for each seed.
for seed in SEEDS:
    print(f"\n=== ST-GAT shuffled topology | seed {seed} ===")
    cfg = copy.deepcopy(cfg0)
    cfg["seed"] = seed

    seed_root = artifact_root / f"seed_{seed}"
    metrics = train_one(
        "stgat",
        cfg,
        artifact_root=seed_root,
        edge_index_override=shuffled_edge_index,
    )

    shuffled_rows.append(
        {
            "seed": seed,
            "topology": "shuffled",
            **{m: metrics[m] for m in METRICS},
        }
    )

    # Save progress after every completed run.
    partial = pd.concat(
        [real, pd.DataFrame(shuffled_rows)],
        ignore_index=True,
    )
    partial.to_csv(reports / "topology_ablation_runs.csv", index=False)

shuffled = pd.DataFrame(shuffled_rows)
all_runs = pd.concat([real, shuffled], ignore_index=True)
all_runs.to_csv(reports / "topology_ablation_runs.csv", index=False)

summary = all_runs.groupby("topology")[METRICS].agg(["mean", "std"])
summary.columns = [
    f"{metric}_{stat}"
    for metric, stat in summary.columns
]
summary = summary.reset_index()
summary.to_csv(reports / "topology_ablation_summary.csv", index=False)

stats_rows = []
for metric in METRICS:
    a = real.sort_values("seed")[metric].to_numpy(dtype=float)
    b = shuffled.sort_values("seed")[metric].to_numpy(dtype=float)

    raw_diff = a - b
    advantage = raw_diff if HIGHER_IS_BETTER[metric] else -raw_diff

    t = ttest_rel(a, b, nan_policy="omit")
    try:
        w = wilcoxon(a, b)
        w_stat, w_p = float(w.statistic), float(w.pvalue)
    except ValueError:
        w_stat, w_p = np.nan, np.nan

    std_adv = np.nanstd(advantage, ddof=1)
    dz = (
        float(np.nanmean(advantage) / std_adv)
        if std_adv > 0
        else float("inf")
    )

    stats_rows.append(
        {
            "metric": metric,
            "higher_is_better": HIGHER_IS_BETTER[metric],
            "real_mean": float(np.nanmean(a)),
            "shuffled_mean": float(np.nanmean(b)),
            "real_minus_shuffled": float(np.nanmean(raw_diff)),
            "real_advantage": float(np.nanmean(advantage)),
            "cohens_dz_real_advantage": dz,
            "paired_t_p": float(t.pvalue),
            "wilcoxon_stat": w_stat,
            "wilcoxon_p": w_p,
        }
    )

# Save the significance test results to CSV.
stats = pd.DataFrame(stats_rows)
stats.to_csv(reports / "topology_ablation_significance.csv", index=False)

print("\n=== TOPOLOGY ABLATION SUMMARY ===")
print(summary.to_string(index=False))

print("\n=== PAIRED REAL vs SHUFFLED ===")
print(stats.to_string(index=False))

print("\nSaved:")
print(" - reports/topology_ablation_graph.json")
print(" - reports/topology_ablation_runs.csv")
print(" - reports/topology_ablation_summary.csv")
print(" - reports/topology_ablation_significance.csv")
print(" - artifacts_topology_ablation/seed_<seed>/stgat/best.pt")