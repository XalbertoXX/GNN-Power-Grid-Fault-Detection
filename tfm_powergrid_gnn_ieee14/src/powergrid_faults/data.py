from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from .topology import GENERATOR_BUSES_1BASED, LINE_NAMES

META_COLS = ["FCT", "FaultType", "lname", "POL"]
PRE_P = [f"pGen{i:02d}" for i in range(1, 6)]
PRE_Q = [f"qGen{i:02d}" for i in range(1, 6)]
PRE_V = [f"vGen{i:02d}" for i in range(1, 6)]
PRE_COLS = PRE_P + PRE_Q + PRE_V
SECURITY_NAMES_RAW = ["unstable", "urgent", "strong", "alarm", "normal"]
POSITION_VALUES = np.asarray([21.4, 67.2, 91.7], dtype=np.float32)

# Check that the Excel sheet has the expected schema. The Figshare release has 230 columns:
def _hash_row(values: np.ndarray, decimals: int) -> str:
    rounded = np.round(values.astype(np.float64), decimals=decimals)
    return hashlib.sha1(rounded.tobytes()).hexdigest()[:12]

# Read the IEEE 14 bus test system data from an Excel file.
def read_ieee14_excel(path: str | Path, sheet: str = "row data") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Excel not found: {path}. Put 'IEEE 14 bus test system data.xlsx' at that path "
            "or update dataset.excel_path in configs/default.yaml."
        )
    df = pd.read_excel(path, sheet_name=sheet)
    expected = set(META_COLS + PRE_COLS + ["output"])
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")
    if df.shape[1] != 230:
        raise ValueError(f"Expected 230 columns, got {df.shape[1]}; inspect the Excel schema before training.")
    if df.isna().any().any():
        raise ValueError("Dataset contains NaN values; inspect before training instead of silently imputing.")
    return df

# Get the column names for the dynamic features.
def dynamic_columns(df: pd.DataFrame) -> list[str]:
    # Exact schema discovered in the Figshare Excel: metadata[0:4], pre-fault[4:19],
    # 14 groups x 15 generator features[19:229], output[229].
    cols = list(df.columns[19:229])
    if len(cols) != 210:
        raise ValueError(f"Expected 210 dynamic columns, got {len(cols)}")
    return cols

# Reshape the dynamic features into a 4D array.
def reshape_dynamic(df: pd.DataFrame) -> np.ndarray:
    """Return [samples, 14 published dynamic blocks, 5 monitored generators, 3 features P/Q/V]."""
    arr = df[dynamic_columns(df)].to_numpy(np.float32)
    arr = arr.reshape(len(df), 14, 3, 5)      # group -> P(5), Q(5), V(5)
    arr = np.transpose(arr, (0, 1, 3, 2))     # [S,T,G,3]
    return arr

# Reshape the pre-fault features into a 3D array.
def reshape_prefault(df: pd.DataFrame) -> np.ndarray:
    arr = df[PRE_COLS].to_numpy(np.float32).reshape(len(df), 3, 5)
    return np.transpose(arr, (0, 2, 1))        # [S,G,3]

# Infer the operating condition groups from the pre-fault features.
def infer_operating_condition_groups(df: pd.DataFrame, decimals: int = 5) -> tuple[np.ndarray, dict]:
    """Infer repeated operating-condition IDs from pre-fault P/Q/V signatures.

    The dataset has 21120/(10*4*16*3)=11 repetitions. We only use the signature-based
    grouping when it actually produces a small repeated set; otherwise we fall back to a
    deterministic within-combination repetition index and report that fact.
    """
    pre = df[PRE_COLS].to_numpy(np.float64)
    sig = np.asarray([_hash_row(row, decimals) for row in pre])
    uniq, counts = np.unique(sig, return_counts=True)

    info = {
        "method": "prefault_signature",
        "n_groups": int(len(uniq)),
        "min_group_size": int(counts.min()),
        "max_group_size": int(counts.max()),
    }
    if len(uniq) == 11:
        map_id = {s: i for i, s in enumerate(sorted(uniq))}
        groups = np.asarray([map_id[s] for s in sig], dtype=np.int64)
        return groups, info

    # Fallback: each FCT/type/line/POL combination has exactly 11 samples in this release.
    groups = df.groupby(META_COLS, sort=False).cumcount().to_numpy(np.int64)
    n_groups = int(groups.max() + 1)
    info.update({"method": "within_combination_rank", "n_groups": n_groups})
    if n_groups != 11:
        raise ValueError(
            f"Could not recover the expected 11 operating-condition groups (got {n_groups}). "
            "Do not use a random row split because it may leak operating conditions."
        )
    return groups, info

# Split the dataset into train/val/test indices based on operating condition groups.
def split_by_groups(groups: np.ndarray, seed: int, val_groups: int = 1, test_groups: int = 2):
    unique = np.unique(groups)
    if len(unique) < val_groups + test_groups + 1:
        raise ValueError("Not enough operating-condition groups for group-wise split")
    rng = np.random.default_rng(seed)
    order = rng.permutation(unique)
    test_g = order[:test_groups]
    val_g = order[test_groups:test_groups + val_groups]
    train_g = order[test_groups + val_groups:]
    return (
        np.flatnonzero(np.isin(groups, train_g)),
        np.flatnonzero(np.isin(groups, val_g)),
        np.flatnonzero(np.isin(groups, test_g)),
        {"train_groups": train_g.tolist(), "val_groups": val_g.tolist(), "test_groups": test_g.tolist()},
    )

# Fit scalers for dynamic and pre-fault features based on the training indices.
def _fit_scalers(dynamic: np.ndarray, prefault: np.ndarray, train_idx: np.ndarray):
    dyn_scaler = StandardScaler().fit(dynamic[train_idx].reshape(-1, 3))
    pre_scaler = StandardScaler().fit(prefault[train_idx].reshape(-1, 3))
    return dyn_scaler, pre_scaler

# Build the input tensor for the bus features, including dynamic and pre-fault data.
def build_bus_tensor(dynamic: np.ndarray, prefault: np.ndarray, dyn_scaler, pre_scaler,
                     generator_buses: list[int]) -> np.ndarray:
    """Create [S,T,14,7] = dyn(P,Q,V), pre(P,Q,V), observed-mask."""
    s, t, g, _ = dynamic.shape
    dyn = dyn_scaler.transform(dynamic.reshape(-1, 3)).reshape(s, t, g, 3).astype(np.float32)
    pre = pre_scaler.transform(prefault.reshape(-1, 3)).reshape(s, g, 3).astype(np.float32)

    X = np.zeros((s, t, 14, 7), dtype=np.float32)
    for gi, bus1 in enumerate(generator_buses):
        b = bus1 - 1
        X[:, :, b, 0:3] = dyn[:, :, gi, :]
        X[:, :, b, 3:6] = pre[:, None, gi, :]
        X[:, :, b, 6] = 1.0
    return X

# Build the target arrays for fault type, line, position, POL, security, and FCT.
def build_targets(df: pd.DataFrame):
    fault_type = df["FaultType"].to_numpy(np.int64) - 1
    line = df["lname"].to_numpy(np.int64) - 1
    pol_raw = df["POL"].to_numpy(np.float32)
    pos_map = {float(v): i for i, v in enumerate(POSITION_VALUES.tolist())}
    try:
        position = np.asarray([pos_map[float(v)] for v in pol_raw], dtype=np.int64)
    except KeyError as e:
        raise ValueError(f"Unexpected POL value: {e}") from e
    security = df["output"].to_numpy(np.int64)
    if not set(np.unique(security)).issubset(set(range(5))):
        raise ValueError("Expected output labels 0..4")
    fct = df["FCT"].to_numpy(np.float32)
    return fault_type, line, position, pol_raw, security, fct

# Prepare the IEEE 14 bus dataset by reading the Excel file, reshaping features, splitting into train/val/test, and saving to disk.
def prepare_ieee14_dataset(cfg: dict):
    dcfg = cfg["dataset"]
    df = read_ieee14_excel(dcfg["excel_path"], dcfg.get("sheet", "row data"))

    # Integrity checks derived directly from the released dataset.
    checks = {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "fct_unique": sorted(map(float, df["FCT"].unique())),
        "fault_type_unique": sorted(map(int, df["FaultType"].unique())),
        "line_unique": sorted(map(int, df["lname"].unique())),
        "pol_unique": sorted(map(float, df["POL"].unique())),
        "output_counts": {str(int(k)): int(v) for k, v in df["output"].value_counts().sort_index().items()},
    }
    expected_product = df["FCT"].nunique() * df["FaultType"].nunique() * df["lname"].nunique() * df["POL"].nunique()
    checks["fault_combinations"] = int(df[META_COLS].drop_duplicates().shape[0])
    checks["repetitions_per_fault_combination"] = float(len(df) / expected_product)

    dynamic = reshape_dynamic(df)
    prefault = reshape_prefault(df)
    groups, group_info = infer_operating_condition_groups(df, int(dcfg.get("oc_round_decimals", 5)))
    train_idx, val_idx, test_idx, split_meta = split_by_groups(
        groups, int(cfg["seed"]), int(dcfg.get("val_groups", 1)), int(dcfg.get("test_groups", 2))
    )

    dyn_scaler, pre_scaler = _fit_scalers(dynamic, prefault, train_idx)
    generator_buses = list(map(int, dcfg.get("generator_buses", GENERATOR_BUSES_1BASED)))
    X = build_bus_tensor(dynamic, prefault, dyn_scaler, pre_scaler, generator_buses)
    y_type, y_line, y_pos, y_pol, y_security, fct = build_targets(df)

    # Optional FCT feature. FCT is in cycles in the released data, not seconds.
    if bool(dcfg.get("include_fct_as_feature", False)):
        fct_scaler = StandardScaler().fit(fct[train_idx, None])
        z = fct_scaler.transform(fct[:, None]).astype(np.float32)[:, 0]
        extra = np.broadcast_to(z[:, None, None, None], (len(df), X.shape[1], X.shape[2], 1))
        X = np.concatenate([X, extra], axis=-1)
    else:
        fct_scaler = None

    out_dir = Path(dcfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "dataset.npz",
        X=X,
        y_type=y_type,
        y_line=y_line,
        y_position=y_pos,
        y_pol=y_pol,
        y_security=y_security,
        fct=fct,
        oc_group=groups,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )
    joblib.dump({"dynamic": dyn_scaler, "prefault": pre_scaler, "fct": fct_scaler}, out_dir / "scalers.joblib")

    metadata = {
        "dataset": "IEEE 14 bus test systems row data (Figshare 30590399)",
        "n_samples": int(len(df)),
        "n_timesteps": int(X.shape[1]),
        "n_buses": 14,
        "in_features": int(X.shape[-1]),
        "generator_buses_1based": generator_buses,
        "feature_names": ["dynamic_P", "dynamic_Q", "dynamic_V", "prefault_P", "prefault_Q", "prefault_V", "observed"] + (["FCT_cycles"] if bool(dcfg.get("include_fct_as_feature", False)) else []),
        "fault_type_classes_raw": [1, 2, 3, 4],
        "line_classes_raw": list(range(1, 17)),
        "line_names_reference_order": LINE_NAMES,
        "line_mapping_status": "PowerFactory reference order; verify lname-to-line identity before making endpoint-specific claims",
        "position_values_percent": POSITION_VALUES.tolist(),
        "security_raw_to_name": {str(i): name for i, name in enumerate(SECURITY_NAMES_RAW)},
        "group_detection": group_info,
        "split": split_meta,
        "integrity": checks,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    Path("reports").mkdir(exist_ok=True)
    (Path("reports") / "ieee14_schema_report.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata

# Dataset class for the IEEE 14 bus test system
class IEEE14Dataset(Dataset):
    def __init__(self, npz_path: str | Path, split: str):
        data = np.load(npz_path, allow_pickle=False)
        idx = data[f"{split}_idx"]
        self.X = torch.from_numpy(data["X"][idx]).float()
        self.y_type = torch.from_numpy(data["y_type"][idx]).long()
        self.y_line = torch.from_numpy(data["y_line"][idx]).long()
        self.y_position = torch.from_numpy(data["y_position"][idx]).long()
        self.y_pol = torch.from_numpy(data["y_pol"][idx]).float()
        self.y_security = torch.from_numpy(data["y_security"][idx]).long()
        self.fct = torch.from_numpy(data["fct"][idx]).float()
        self.oc_group = torch.from_numpy(data["oc_group"][idx]).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return {
            "x": self.X[i], "y_type": self.y_type[i], "y_line": self.y_line[i],
            "y_position": self.y_position[i], "y_pol": self.y_pol[i],
            "y_security": self.y_security[i], "fct": self.fct[i], "oc_group": self.oc_group[i],
        }
