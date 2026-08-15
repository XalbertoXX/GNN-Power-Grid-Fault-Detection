from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

POSITION_VALUES = np.asarray([21.4, 67.2, 91.7], dtype=np.float32)


def _class_metrics(y, probs, prefix):
    pred = probs.argmax(1)
    return {
        f"{prefix}_accuracy": float(accuracy_score(y, pred)),
        f"{prefix}_balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        f"{prefix}_macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def all_metrics(y_type, p_type, y_line, p_line, y_position, p_position,
                y_pol, y_security, p_security):
    out = {}
    out.update(_class_metrics(y_type, p_type, "fault_type"))
    out.update(_class_metrics(y_line, p_line, "line"))
    out.update(_class_metrics(y_position, p_position, "position"))
    out.update(_class_metrics(y_security, p_security, "security"))

    top3 = np.argsort(-p_line, axis=1)[:, :3]
    out["line_top3"] = float(np.mean([int(t) in row for t, row in zip(y_line, top3)]))

    # Reference PowerFactory representation contains two parallel 1-2 circuits.
    # Corridor accuracy merges classes 0 and 1 to avoid overstating an ambiguity
    # that cannot be resolved from endpoint topology alone.
    pred_line = p_line.argmax(1)
    corridor_true = np.where(y_line == 1, 0, y_line)
    corridor_pred = np.where(pred_line == 1, 0, pred_line)
    out["line_corridor_accuracy"] = float(np.mean(corridor_true == corridor_pred))

    expected_pol = p_position @ POSITION_VALUES
    pred_pol_class = POSITION_VALUES[p_position.argmax(1)]
    err_expected = np.abs(expected_pol - y_pol)
    err_class = np.abs(pred_pol_class - y_pol)
    out["position_mae_expected_pct"] = float(err_expected.mean())
    out["position_mae_class_pct"] = float(err_class.mean())
    out["position_rmse_expected_pct"] = float(np.sqrt(np.mean((expected_pol - y_pol) ** 2)))
    out["position_within_5pct"] = float(np.mean(err_expected <= 5.0))
    out["position_within_10pct"] = float(np.mean(err_expected <= 10.0))

    # Raw Excel label order inferred from the exact published class counts:
    # 0 unstable, 1 urgent, 2 strong, 3 alarm, 4 normal.
    sec_pred = p_security.argmax(1)
    out["security_ordinal_mae"] = float(np.mean(np.abs(sec_pred - y_security)))

    # Binary instability assessment is legitimate; it is NOT fault/no-fault detection.
    y_unstable = (np.asarray(y_security) == 0).astype(int)
    p_unstable = np.asarray(p_security)[:, 0]
    pred_unstable = (p_unstable >= 0.5).astype(int)
    try:
        out["instability_auroc"] = float(roc_auc_score(y_unstable, p_unstable))
    except ValueError:
        out["instability_auroc"] = float("nan")
    out["instability_f1"] = float(f1_score(y_unstable, pred_unstable, zero_division=0))
    out["instability_precision"] = float(precision_score(y_unstable, pred_unstable, zero_division=0))
    out["instability_recall"] = float(recall_score(y_unstable, pred_unstable, zero_division=0))
    return out
