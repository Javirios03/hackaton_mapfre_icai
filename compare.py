"""
Compare autoresearch model vs solution model side by side.
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, confusion_matrix
)

DATA_FILE = Path(__file__).resolve().parent / "data" / "ticdata2000.txt"
COL_NAMES = [f"M{i}" for i in range(1, 86)] + ["CARAVAN"]


def evaluate_model(module_path, label):
    """Import a caravan_model module, train it, and return metrics."""
    # Clean imports
    for mod in list(sys.modules):
        if "caravan_model" in mod or "model" in mod:
            del sys.modules[mod]

    sys.path.insert(0, str(module_path))
    try:
        import caravan_model
        # Reset caches
        caravan_model._cached_df = None
        caravan_model._cached_metrics = None
        caravan_model.load_model.cache_clear()

        start = time.time()
        model = caravan_model.load_model()
        train_time = time.time() - start

        feature_names = caravan_model.get_feature_names()
        n_features = len(feature_names)

        # Load data and split with same seed
        df = pd.read_csv(DATA_FILE, sep="\t", header=None, names=COL_NAMES)
        y = (df["CARAVAN"] > 0).astype(int)
        X = df[feature_names]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_proba)
        auc_pr = average_precision_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        top_k = min(800, len(y_test))
        top_idx = np.argsort(y_proba)[::-1][:top_k]
        recall_at_800 = int(y_test.iloc[top_idx].sum())
        total_pos = int(y_test.sum())

        # Get model type
        model_type = caravan_model._cached_metrics.get("model_type", "unknown") if caravan_model._cached_metrics else "unknown"

        return {
            "label": label,
            "model_type": model_type,
            "n_features": n_features,
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "recall_at_800": f"{recall_at_800}/{total_pos} ({recall_at_800/total_pos:.1%})",
            "confusion_matrix": cm.tolist(),
            "train_time_s": round(train_time, 2),
        }
    finally:
        sys.path.remove(str(module_path))
        for mod in list(sys.modules):
            if "caravan_model" in mod:
                del sys.modules[mod]


def print_comparison(ours, theirs):
    print("=" * 72)
    print("  COMPARISON: autoresearch vs solution de referencia")
    print("=" * 72)

    metrics = [
        ("Model type",    "model_type",    None),
        ("N features",    "n_features",    None),
        ("AUC-ROC",       "auc_roc",       ".6f"),
        ("AUC-PR",        "auc_pr",        ".6f"),
        ("Accuracy",      "accuracy",      ".6f"),
        ("Precision",     "precision",     ".6f"),
        ("Recall",        "recall",        ".6f"),
        ("F1",            "f1",            ".6f"),
        ("Recall@800",    "recall_at_800", None),
        ("Train time (s)","train_time_s",  None),
    ]

    header = f"{'Metric':<20} {'Autoresearch':>25} {'Solution':>25} {'Delta':>15}"
    print(header)
    print("-" * len(header))

    for name, key, fmt in metrics:
        v_ours = ours[key]
        v_theirs = theirs[key]
        if fmt and isinstance(v_ours, (int, float)):
            s_ours = f"{v_ours:{fmt}}"
            s_theirs = f"{v_theirs:{fmt}}"
            delta = v_ours - v_theirs
            if delta > 0:
                s_delta = f"+{delta:{fmt}}"
            elif delta < 0:
                s_delta = f"{delta:{fmt}}"
            else:
                s_delta = "0"
        else:
            s_ours = str(v_ours)
            s_theirs = str(v_theirs)
            s_delta = ""

        print(f"{name:<20} {s_ours:>25} {s_theirs:>25} {s_delta:>15}")

    print()
    print("Confusion matrices (TN, FP / FN, TP):")
    print(f"  Autoresearch: {ours['confusion_matrix']}")
    print(f"  Solution:     {theirs['confusion_matrix']}")
    print()

    # Winner
    diff = ours["auc_roc"] - theirs["auc_roc"]
    if diff > 0:
        print(f">>> AUTORESEARCH WINS by +{diff:.4f} AUC-ROC ({diff/theirs['auc_roc']:.1%} improvement)")
    elif diff < 0:
        print(f">>> SOLUTION WINS by +{-diff:.4f} AUC-ROC")
    else:
        print(">>> TIE on AUC-ROC")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent

    print("Evaluating autoresearch model...")
    ours = evaluate_model(root / "backend", "autoresearch")

    print("Evaluating solution model...")
    theirs = evaluate_model(root / "solution" / "backend", "solution")

    print()
    print_comparison(ours, theirs)
