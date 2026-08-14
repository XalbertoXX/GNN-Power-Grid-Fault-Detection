from pathlib import Path
import pandas as pd

from powergrid_faults.data import (
    META_COLS, PRE_COLS, dynamic_columns, infer_operating_condition_groups,
    read_ieee14_excel, reshape_dynamic,
)
from powergrid_faults.utils import load_config

ROOT = Path(__file__).resolve().parents[1]
cfg = load_config(ROOT / "configs/default.yaml")
d = cfg["dataset"]
df = read_ieee14_excel(d["excel_path"], d.get("sheet", "row data"))

print("=" * 80)
print("IEEE-14 DATASET CHECK")
print("=" * 80)
print("shape:", df.shape)
for col in ["FCT", "FaultType", "lname", "POL", "output"]:
    print(f"\n{col}: unique={df[col].nunique()}")
    print(df[col].value_counts().sort_index())

n_fault_combos = len(df[META_COLS].drop_duplicates())
reps = len(df) / n_fault_combos
print("\ndynamic columns:", len(dynamic_columns(df)))
print("dynamic reshape:", reshape_dynamic(df).shape)
print("fault combinations FCT×type×line×POL:", n_fault_combos)
print("repetitions per fault combination:", reps)
print("factorization:", f"{df['FCT'].nunique()} × {df['FaultType'].nunique()} × {df['lname'].nunique()} × {df['POL'].nunique()} × {int(reps)} = {len(df)}")

groups, info = infer_operating_condition_groups(df, int(d.get("oc_round_decimals", 5)))
print("\noperating-condition grouping:", info)
print(pd.Series(groups).value_counts().sort_index())

print("\nsecurity raw-label interpretation by published counts:")
print("0=unstable, 1=urgent, 2=strong, 3=alarm, 4=normal")
print("\nPre-fault columns:", PRE_COLS)
print("\nIMPORTANT: FaultType/lname/POL/output are targets and are never included in X.")
