from pathlib import Path
import copy

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

from powergrid_faults.trainlib import train_one
from powergrid_faults.utils import load_config

ROOT = Path(__file__).resolve().parents[1]
cfg0 = load_config(ROOT / "configs/default.yaml")
seeds = [11, 22, 33, 44, 55]
rows = []

repeated_root = ROOT / "artifacts_repeated"
repeated_root.mkdir(exist_ok=True)

for seed in seeds:
    for model in ["cnn", "lstm", "stgat"]:
        cfg = copy.deepcopy(cfg0)
        cfg["seed"] = seed
        artifact_root = repeated_root / f"seed_{seed}"
        m = train_one(model, cfg, artifact_root=artifact_root)
        rows.append({"seed": seed, **m})

reports = ROOT / "reports"
reports.mkdir(exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(reports / "repeated_runs.csv", index=False)

metrics = [
    "fault_type_macro_f1",
    "line_accuracy",
    "line_top3",
    "position_mae_expected_pct",
    "security_macro_f1",
    "instability_auroc",
]

# Mean ± standard deviation table for direct use in the thesis.
summary = df.groupby("model")[metrics].agg(["mean", "std"])
summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
summary.reset_index().to_csv(reports / "repeated_summary.csv", index=False)

results = []
for baseline in ["cnn", "lstm"]:
    for metric in metrics:
        a = df[df.model.eq("stgat")].sort_values("seed")[metric].to_numpy(dtype=float)
        b = df[df.model.eq(baseline)].sort_values("seed")[metric].to_numpy(dtype=float)
        if len(a) == len(b) and len(a) >= 2:
            t = ttest_rel(a, b, nan_policy="omit")
            try:
                w = wilcoxon(a, b)
                w_stat, w_p = w.statistic, w.pvalue
            except ValueError:
                w_stat, w_p = np.nan, np.nan
            diff = a - b
            std = np.nanstd(diff, ddof=1)
            dz = float(np.nanmean(diff) / std) if std > 0 else float("inf")
            results.append({
                "comparison": f"stgat_vs_{baseline}",
                "metric": metric,
                "mean_diff": float(np.nanmean(diff)),
                "cohens_dz": dz,
                "paired_t_p": float(t.pvalue),
                "wilcoxon_stat": float(w_stat),
                "wilcoxon_p": float(w_p),
            })

pd.DataFrame(results).to_csv(reports / "significance.csv", index=False)
print("\nRepeated-run summary (mean ± std):")
print(summary)
print("\nSaved:")
print(" - reports/repeated_runs.csv")
print(" - reports/repeated_summary.csv")
print(" - reports/significance.csv")
print(" - artifacts_repeated/seed_<seed>/<model>/best.pt")
