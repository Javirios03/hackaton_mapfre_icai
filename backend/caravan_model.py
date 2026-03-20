"""
Modelo COIL 2000: predicción de interés en seguro de caravana.

Carga los datos desde data/TICDATA2000.txt (UCI Insurance Company Benchmark).
Los alumnos deben sustituir este pipeline por su propio modelo, feature selection
y fine-tuning; implementar get_metrics() y get_dataset_info() para exponer
resultados en la API y en Streamlit.

Enlace UCI: https://archive.ics.uci.edu/ml/datasets/Insurance+Company+Benchmark+(COIL+2000)
"""

from pathlib import Path
from functools import lru_cache
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Ruta al dataset (desde backend/, data/ está en la raíz del proyecto)
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_TRAIN_FILE = _DATA_DIR / "ticdata2000.txt"
# 86 atributos: M1..M85 (sociodemográficos + productos) y CARAVAN (target)
_COL_NAMES = [f"M{i}" for i in range(1, 86)] + ["CARAVAN"]
# TODO (alumno): Tras tu análisis de feature selection, sustituye por tu lista de variables
# (mismo orden que usarás en el modelo). Por defecto se usan las 10 primeras.
_DEFAULT_FEATURE_COLS = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10"]
_TARGET_COL = "CARAVAN"

# Cache de datos y métricas para get_dataset_info / get_metrics
_cached_df: pd.DataFrame | None = None
_cached_metrics: dict | None = None


def _load_data() -> pd.DataFrame:
    """Carga TICDATA2000.txt (tab-delimited, sin cabecera)."""
    if not _TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo {_TRAIN_FILE}. "
            "Descarga COIL 2000 desde UCI y coloca TICDATA2000.txt en data/. "
            "Ver GUIA_HACKATON.md (sección 1)."
        )
    df = pd.read_csv(_TRAIN_FILE, sep="\t", header=None, names=_COL_NAMES)
    return df


def _get_df() -> pd.DataFrame:
    """Devuelve el dataset en memoria (cache)."""
    global _cached_df
    if _cached_df is None:
        _cached_df = _load_data()
    return _cached_df


@lru_cache(maxsize=1)
def load_model() -> Any:
    """
    Entrena un pipeline por defecto sobre COIL 2000 (subconjunto de features).

    Los alumnos deben reemplazar esta lógica por su propio modelo, feature
    selection y fine-tuning; y opcionalmente exponer métricas con get_metrics().
    """
    df = _get_df()
    y = (df[_TARGET_COL] > 0).astype(int)  # binarizar: tiene póliza caravana = 1
    X = df[_DEFAULT_FEATURE_COLS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TODO (alumno): Sustituye por tu modelo y/o hiperparámetros (p. ej. RandomForest, XGBoost)
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced")),
        ]
    )
    pipeline.fit(X_train, y_train)

    # Calcular y cachear métricas para get_metrics()
    global _cached_metrics
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    _cached_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "auc_roc": float(roc_auc_score(y_test, y_proba)) if y_test.nunique() > 1 else 0.0,
        "n_features": len(_DEFAULT_FEATURE_COLS),
        "model_type": "LogisticRegression",
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    return pipeline


def get_feature_names() -> list[str]:
    """Devuelve los nombres de variables en el orden esperado por el modelo."""
    return _DEFAULT_FEATURE_COLS.copy()


def predict_single(features: list[float]) -> dict:
    """
    Predicción para una instancia: probabilidad de interés en seguro de caravana.

    Returns
    -------
    dict con "prediction" (0/1) y "probability" (prob. clase positiva).
    """
    model = load_model()
    proba = model.predict_proba([features])[0][1]
    pred = int(proba >= 0.5)
    return {"prediction": pred, "probability": float(proba)}


def get_metrics() -> dict | None:
    """
    Métricas del modelo (calculadas al entrenar). Usado por GET /metrics.

    Los alumnos pueden extender este dict (AUC-PR, recall@k, etc.).
    """
    load_model()  # asegura que _cached_metrics está rellenado
    return _cached_metrics


def get_dataset_info() -> dict | None:
    """
    Resumen del dataset para GET /dataset_info y la pestaña Streamlit.
    """
    try:
        df = _get_df()
    except FileNotFoundError:
        return None
    y = (df[_TARGET_COL] > 0).astype(int)
    counts = y.value_counts().to_dict()
    return {
        "n_samples": len(df),
        "n_features": len(_COL_NAMES) - 1,
        "feature_names": get_feature_names(),
        "target_name": _TARGET_COL,
        "class_distribution": {str(k): int(v) for k, v in counts.items()},
    }


def get_dataset_sample(n: int = 100) -> list[dict] | None:
    """
    Devuelve hasta n filas del dataset para mostrar en Streamlit.
    Usado por GET /dataset_sample.
    """
    try:
        df = _get_df()
    except FileNotFoundError:
        return None
    subset = df[get_feature_names() + [_TARGET_COL]].head(n)
    return subset.to_dict(orient="records")
