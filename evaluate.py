"""
evaluate.py — Fixed evaluation harness for autoresearch (DO NOT MODIFY).

Loads caravan_model, trains it, evaluates on held-out test set, and prints
metrics in a standardized format. The agent compares val_auc_roc across
experiments to decide keep/discard.

Usage:
    python evaluate.py
"""

import json
import sys
import time
import traceback
from pathlib import Path

# Add backend to path so we can import caravan_model
sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

def run_evaluation():
    start = time.time()
    try:
        # Force fresh import (no caching across experiments)
        if "caravan_model" in sys.modules:
            del sys.modules["caravan_model"]
        if "model" in sys.modules:
            del sys.modules["model"]

        import caravan_model

        # Reset caches so model retrains from scratch
        caravan_model._cached_df = None
        caravan_model._cached_metrics = None
        caravan_model.load_model.cache_clear()

        # Load data
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (
            accuracy_score, roc_auc_score, average_precision_score,
            confusion_matrix, recall_score, precision_score, f1_score
        )

        df = caravan_model._load_data()
        y = (df[caravan_model._TARGET_COL] > 0).astype(int)
        X = df[caravan_model._DEFAULT_FEATURE_COLS]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train model via caravan_model's pipeline
        model = caravan_model.load_model()

        # Evaluate
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_proba) if y_test.nunique() > 1 else 0.0
        auc_pr = average_precision_score(y_test, y_proba) if y_test.nunique() > 1 else 0.0
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()

        # recall@800: sort by predicted probability descending, take top 800,
        # count how many are actually positive
        import numpy as np
        top_k = 800
        top_indices = np.argsort(y_proba)[::-1][:top_k]
        recall_at_800 = int(y_test.iloc[top_indices].sum())
        total_positives = int(y_test.sum())
        recall_at_800_pct = recall_at_800 / total_positives if total_positives > 0 else 0.0

        elapsed = time.time() - start
        n_features = len(caravan_model._DEFAULT_FEATURE_COLS)
        model_type = type(model.named_steps.get("clf", model)).__name__ if hasattr(model, "named_steps") else type(model).__name__

        # Print in standardized format (grep-friendly)
        print("---")
        print(f"val_auc_roc: {auc_roc:.6f}")
        print(f"val_auc_pr: {auc_pr:.6f}")
        print(f"val_accuracy: {acc:.6f}")
        print(f"val_precision: {precision:.6f}")
        print(f"val_recall: {recall:.6f}")
        print(f"val_f1: {f1:.6f}")
        print(f"recall_at_800: {recall_at_800}/{total_positives} ({recall_at_800_pct:.4f})")
        print(f"confusion_matrix: {json.dumps(cm)}")
        print(f"n_features: {n_features}")
        print(f"model_type: {model_type}")
        print(f"n_train: {len(X_train)}")
        print(f"n_test: {len(X_test)}")
        print(f"total_seconds: {elapsed:.1f}")

        # Also dump JSON for machine parsing
        results = {
            "val_auc_roc": round(auc_roc, 6),
            "val_auc_pr": round(auc_pr, 6),
            "val_accuracy": round(acc, 6),
            "val_precision": round(precision, 6),
            "val_recall": round(recall, 6),
            "val_f1": round(f1, 6),
            "recall_at_800": recall_at_800,
            "recall_at_800_pct": round(recall_at_800_pct, 4),
            "total_positives": total_positives,
            "confusion_matrix": cm,
            "n_features": n_features,
            "model_type": model_type,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "total_seconds": round(elapsed, 1),
            "status": "ok"
        }
        with open("eval_result.json", "w") as f:
            json.dump(results, f, indent=2)

    except Exception as e:
        elapsed = time.time() - start
        print(f"CRASH: {e}")
        traceback.print_exc()
        results = {
            "status": "crash",
            "error": str(e),
            "total_seconds": round(elapsed, 1)
        }
        with open("eval_result.json", "w") as f:
            json.dump(results, f, indent=2)
        sys.exit(1)

if __name__ == "__main__":
    run_evaluation()
