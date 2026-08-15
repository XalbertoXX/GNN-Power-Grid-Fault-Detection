from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import IEEE14Dataset
from .metrics import all_metrics
from .models import build_model
from .topology import case14_powerfactory_graph
from .utils import get_device, save_json, seed_everything


def multitask_loss(out, batch, cfg, security_weights=None):
    ce = nn.CrossEntropyLoss()
    sec_ce = nn.CrossEntropyLoss(weight=security_weights) if security_weights is not None else ce
    losses = {
        "fault_type": ce(out["fault_type"], batch["y_type"]),
        "line": ce(out["line"], batch["y_line"]),
        "position": ce(out["position"], batch["y_position"]),
        "security": sec_ce(out["security"], batch["y_security"]),
    }
    tr = cfg["training"]
    total = (
        float(tr["lambda_fault_type"]) * losses["fault_type"] +
        float(tr["lambda_line"]) * losses["line"] +
        float(tr["lambda_position"]) * losses["position"] +
        float(tr["lambda_security"]) * losses["security"]
    )
    return total, {k: float(v.detach()) for k, v in losses.items()}


def _move(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def predict(model, loader, edge_index, device, perturb=None):
    model.eval()
    acc = {k: [] for k in ["y_type", "p_type", "y_line", "p_line", "y_position", "p_position", "y_pol", "y_security", "p_security"]}
    latencies = []
    with torch.no_grad():
        for batch in loader:
            batch = _move(batch, device)
            x = batch["x"]
            if perturb:
                x = perturb(x)
            if device.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(x, edge_index)
            if device.type == "cuda": torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) / x.size(0))

            acc["y_type"].append(batch["y_type"].cpu().numpy())
            acc["p_type"].append(torch.softmax(out["fault_type"], -1).cpu().numpy())
            acc["y_line"].append(batch["y_line"].cpu().numpy())
            acc["p_line"].append(torch.softmax(out["line"], -1).cpu().numpy())
            acc["y_position"].append(batch["y_position"].cpu().numpy())
            acc["p_position"].append(torch.softmax(out["position"], -1).cpu().numpy())
            acc["y_pol"].append(batch["y_pol"].cpu().numpy())
            acc["y_security"].append(batch["y_security"].cpu().numpy())
            acc["p_security"].append(torch.softmax(out["security"], -1).cpu().numpy())
    vals = {k: np.concatenate(v) for k, v in acc.items()}
    vals["latency_ms_per_sample"] = 1000.0 * float(np.mean(latencies))
    return vals


def train_one(model_name: str, cfg: dict):
    seed_everything(int(cfg["seed"]))
    device = get_device(cfg.get("device", "auto"))
    processed = Path(cfg["dataset"]["processed_dir"])
    meta = json.loads((processed / "metadata.json").read_text(encoding="utf-8"))

    train_ds = IEEE14Dataset(processed / "dataset.npz", "train")
    val_ds = IEEE14Dataset(processed / "dataset.npz", "val")
    test_ds = IEEE14Dataset(processed / "dataset.npz", "test")
    bs = int(cfg["training"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)

    _, edge_index, _ = case14_powerfactory_graph()
    edge_index = edge_index.to(device)
    _, _, graph_meta = case14_powerfactory_graph()
    kw = dict(
        in_features=int(meta["in_features"]), n_buses=14, hidden=int(cfg["model"]["hidden"]),
        heads=int(cfg["model"]["gat_heads"]), dropout=float(cfg["model"]["dropout"]),
        temporal_layers=int(cfg["model"]["temporal_layers"]),
        fault_line_pairs=[list(p) for p in graph_meta["fault_lines"]],
        edge_localization_head=bool(cfg["model"].get("edge_localization_head", False)),
    )
    model = build_model(model_name, **kw).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"]["weight_decay"]))

    security_weights = None
    if str(cfg["training"].get("security_class_weighting", "none")).lower() == "balanced":
        counts = torch.bincount(train_ds.y_security, minlength=5).float()
        security_weights = (counts.sum() / (len(counts) * counts.clamp_min(1.0))).to(device)

    out_dir = Path("artifacts") / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    history, best, bad = [], float("inf"), 0
    patience = int(cfg["training"]["patience"])

    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train(); train_losses = []
        for batch in tqdm(train_loader, desc=f"{model_name} epoch {epoch}", leave=False):
            batch = _move(batch, device)
            opt.zero_grad(set_to_none=True)
            out = model(batch["x"], edge_index)
            loss, _ = multitask_loss(out, batch, cfg, security_weights)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
            train_losses.append(float(loss.detach()))

        model.eval(); val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = _move(batch, device)
                loss, _ = multitask_loss(model(batch["x"], edge_index), batch, cfg, security_weights)
                val_losses.append(float(loss.detach()))
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": float(np.mean(val_losses))}
        history.append(row)
        if row["val_loss"] < best - 1e-5:
            best, bad = row["val_loss"], 0
            torch.save({"state_dict": model.state_dict(), "model_name": model_name, "kwargs": kw, "meta": meta}, out_dir / "best.pt")
        else:
            bad += 1
            if bad >= patience: break

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    pred = predict(model, test_loader, edge_index, device)
    met = all_metrics(pred["y_type"], pred["p_type"], pred["y_line"], pred["p_line"], pred["y_position"], pred["p_position"], pred["y_pol"], pred["y_security"], pred["p_security"])
    met.update({"latency_ms_per_sample": pred["latency_ms_per_sample"], "parameters": int(sum(p.numel() for p in model.parameters())), "model": model_name})
    save_json(met, out_dir / "metrics.json")
    return met


def robustness_table(model_name: str, cfg: dict):
    device = get_device(cfg.get("device", "auto"))
    processed = Path(cfg["dataset"]["processed_dir"])
    ds = IEEE14Dataset(processed / "dataset.npz", "test")
    loader = DataLoader(ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=False)
    _, edge_index, meta_graph = case14_powerfactory_graph(); edge_index = edge_index.to(device)
    ckpt = torch.load(Path("artifacts") / model_name / "best.pt", map_location=device)
    model = build_model(model_name, **ckpt["kwargs"]).to(device); model.load_state_dict(ckpt["state_dict"])
    rows = []

    for std in cfg["robustness"]["noise_std"]:
        def noise(x, s=float(std)):
            if s <= 0: return x
            z = x.clone()
            # only perturb the six continuous node features, not observed mask/FCT
            z[..., :6] += torch.randn_like(z[..., :6]) * s
            return z
        pred = predict(model, loader, edge_index, device, noise)
        met = all_metrics(pred["y_type"], pred["p_type"], pred["y_line"], pred["p_line"], pred["y_position"], pred["p_position"], pred["y_pol"], pred["y_security"], pred["p_security"])
        rows.append({"model": model_name, "perturbation": "gaussian_noise", "level": std, **met})

    observed = meta_graph["generator_buses"]
    for frac in cfg["robustness"]["missing_pmu_fraction"]:
        def missing(x, f=float(frac)):
            if f <= 0: return x
            z = x.clone(); n_drop = max(1, int(round(len(observed) * f)))
            idx = torch.randperm(len(observed), device=x.device)[:n_drop]
            observed_tensor = torch.tensor(observed, dtype=torch.long, device=x.device)
            buses = observed_tensor[idx]
            z[:, :, buses, :] = 0
            return z
        pred = predict(model, loader, edge_index, device, missing)
        met = all_metrics(pred["y_type"], pred["p_type"], pred["y_line"], pred["p_line"], pred["y_position"], pred["p_position"], pred["y_pol"], pred["y_security"], pred["p_security"])
        rows.append({"model": model_name, "perturbation": "missing_pmus", "level": frac, **met})
    return pd.DataFrame(rows)
