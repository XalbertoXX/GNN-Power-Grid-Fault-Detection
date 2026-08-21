from pathlib import Path
import pandas as pd
from powergrid_faults.trainlib import robustness_table, train_one
from powergrid_faults.utils import load_config

# This script trains all models and generates a comparison report.
ROOT = Path(__file__).resolve().parents[1]
cfg = load_config(ROOT / "configs/default.yaml")
rows, robust = [], []
for name in ["cnn", "lstm", "stgat"]:
    print(f"\n=== Training {name.upper()} ===")
    rows.append(train_one(name, cfg))
    robust.append(robustness_table(name, cfg))
Path("reports").mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv("reports/comparison.csv", index=False)
pd.concat(robust, ignore_index=True).to_csv("reports/robustness.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
