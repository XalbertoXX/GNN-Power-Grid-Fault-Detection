from pathlib import Path
from powergrid_faults.data import prepare_ieee14_dataset
from powergrid_faults.utils import load_config

# This script prepares the IEEE 14 bus test system dataset.
ROOT = Path(__file__).resolve().parents[1]
cfg = load_config(ROOT / "configs/default.yaml")
meta = prepare_ieee14_dataset(cfg)
print("\nPrepared IEEE-14 dataset")
print("samples:", meta["n_samples"])
print("shape: [samples, time, buses, features] =", [meta["n_samples"], meta["n_timesteps"], meta["n_buses"], meta["in_features"]])
print("group detection:", meta["group_detection"])
print("split:", meta["split"])
