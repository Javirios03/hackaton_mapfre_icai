# autoresearch: COIL 2000 Caravan Insurance

Autonomous ML research loop adapted from [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) for the MAPFRE-ICAI Hackathon.

**Goal:** Predict caravan insurance interest using the COIL 2000 dataset. Maximize **val_auc_roc** (primary metric) while keeping the API contract intact.

## Setup

To set up a new experiment session:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar20`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files** for full context:
   - `GUIA_HACKATON.md` — hackathon rules, evaluation criteria, task list.
   - `evaluate.py` — fixed evaluation harness. DO NOT MODIFY.
   - `backend/caravan_model.py` — the file you modify. Model, features, pipeline.
   - `backend/main.py` — API endpoints (read-only, do not modify).
   - `backend/schemas.py` — Pydantic schemas (read-only, do not modify).
   - `backend/model.py` — interface re-exports (read-only, do not modify).
4. **Verify data exists**: Check that `data/ticdata2000.txt` exists.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row.
6. **Confirm and go**: Confirm setup looks good, then kick off experimentation.

## What you CAN do

- Modify **`backend/caravan_model.py`** — this is the ONLY file you edit. Everything is fair game:
  - Feature selection (`_DEFAULT_FEATURE_COLS`)
  - Model choice (LogisticRegression, RandomForest, XGBoost, LightGBM, etc.)
  - Hyperparameters
  - Pipeline steps (scaling, encoding, feature engineering)
  - Class imbalance handling (SMOTE, class weights, thresholds)
  - Ensemble methods
- Add imports for packages already in requirements.txt (pandas, scikit-learn, xgboost, lightgbm, optuna, shap, numpy).

## What you CANNOT do

- Modify `evaluate.py`. It is read-only. It contains the fixed evaluation.
- Modify `backend/main.py`, `backend/schemas.py`, or `backend/model.py`.
- Install new packages or add dependencies not in `backend/requirements.txt`.
- Change the API contract: `load_model()`, `get_feature_names()`, `predict_single(features)`, `get_metrics()`, `get_dataset_info()`, `get_dataset_sample(n)` must all keep working.

## The goal

**Primary metric: val_auc_roc** — higher is better. Secondary: val_auc_pr, recall_at_800.

Since this is a classification task with severe class imbalance (~6% positive), AUC-ROC and AUC-PR are much more informative than accuracy. The hackathon judges weight AUC-ROC highest.

## Constraints

- The model must work with the existing API. `predict_single(features)` must accept a list of floats matching `get_feature_names()` and return `{"prediction": int, "probability": float}`.
- `get_metrics()` must return a dict with at least `accuracy`, `auc_roc`, `n_features`, `model_type`.
- `get_dataset_info()` and `get_dataset_sample(n)` must keep working.
- The `load_model()` function must be cacheable (use `@lru_cache` or similar).

## Simplicity criterion

All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

## Running an experiment

```bash
python evaluate.py > run.log 2>&1
```

Extract the key metric:
```bash
grep "^val_auc_roc:" run.log
```

If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace.

## Logging results

Log every experiment to `results.tsv` (tab-separated):

```
commit	val_auc_roc	val_auc_pr	n_features	status	description
```

1. git commit hash (short, 7 chars)
2. val_auc_roc achieved — use 0.000000 for crashes
3. val_auc_pr achieved — use 0.000000 for crashes
4. number of features used
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:
```
commit	val_auc_roc	val_auc_pr	n_features	status	description
a1b2c3d	0.720000	0.150000	10	keep	baseline: LogisticRegression with M1-M10
b2c3d4e	0.765000	0.180000	20	keep	RandomForest with top-20 features by importance
c3d4e5f	0.710000	0.140000	85	discard	all features, no selection
d4e5f6g	0.000000	0.000000	0	crash	XGBoost with bad hyperparams (import error)
```

## The experiment loop

LOOP FOREVER:

1. Look at the git state
2. Modify `backend/caravan_model.py` with an experimental idea
3. `git add backend/caravan_model.py && git commit -m "experiment: <description>"`
4. Run: `python evaluate.py > run.log 2>&1`
5. Read results: `grep "^val_auc_roc:\|^val_auc_pr:\|^n_features:" run.log`
6. If grep is empty → crash. Run `tail -n 50 run.log`, attempt fix or skip.
7. Record in results.tsv (do NOT commit results.tsv)
8. If val_auc_roc improved → **keep** (advance the branch)
9. If val_auc_roc is equal or worse → **discard** (`git reset --hard HEAD~1`)

## Research ideas to try (in rough order)

1. **Baseline**: run as-is to establish starting metrics
2. **Use all 85 features** instead of just M1-M10
3. **Feature selection**: mutual information, chi2, or model-based importance to pick top 15-25
4. **RandomForestClassifier** with class_weight="balanced"
5. **GradientBoostingClassifier** or **HistGradientBoostingClassifier**
6. **XGBClassifier** with scale_pos_weight for imbalance
7. **LGBMClassifier** with is_unbalance=True
8. **Hyperparameter tuning** with RandomizedSearchCV or Optuna
9. **Feature engineering**: interaction terms, polynomial features, binning
10. **SMOTE** or other oversampling on training set only
11. **Ensemble**: VotingClassifier or StackingClassifier combining best models
12. **Threshold optimization**: find optimal probability threshold for F1/recall
13. **Add AUC-PR, recall_at_800** to get_metrics() for Streamlit display

## NEVER STOP

Once the experiment loop has begun, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?". The human might be away and expects you to continue working *indefinitely* until manually stopped. You are autonomous. If you run out of ideas, think harder — try combining previous near-misses, try more radical changes, revisit discarded ideas with tweaks.
