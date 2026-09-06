# Traffic Demand Prediction — Simplified Production Pipeline

A regression pipeline that predicts geospatial traffic demand from road, weather, and time features using a stacked ensemble of CatBoost and a small neural network, combined with a Ridge meta-learner.

## Overview

Given a geohash cell, a time slot, and contextual road/weather attributes, the model predicts a continuous `demand` value. The pipeline is a single, linear notebook: load data → engineer features → train a 5-fold stacked ensemble → generate predictions → save a submission file.

**Out-of-fold R² ≈ 0.95** on the training set.

## Data

| File | Rows | Description |
|---|---|---|
| `train.csv` | ~77,300 | Labeled training data (includes `demand`) |
| `test.csv` | ~41,779 | Unlabeled evaluation data |

**Columns:**
- `Index` — row identifier
- `geohash` — spatial cell identifier
- `day` — day index
- `timestamp` — time of day (`H:M`)
- `demand` — target variable (train only)
- `RoadType` — e.g. Residential
- `NumberofLanes` — integer
- `LargeVehicles` — Allowed / Not Allowed
- `Landmarks` — Yes / No
- `Temperature` — numeric, contains missing values
- `Weather` — Sunny / Rainy / Foggy / Snowy, contains missing values

## Pipeline

The full pipeline lives in `trafficdemandpred.ipynb`:

```
Raw CSV
  │
  ├── engineer_features()     # hour, dayofweek, is_weekend, rush_hour,
  │                           # temp_lane, geohash_target
  │
  ├── ultra_fe()              # temp_lane_hour, is_rush_weekend,
  │                           # geo_freq, label-coded categoricals
  │
  ├── StandardScaler          # fit on full train, applied per fold
  │
  └── 5-Fold KFold
        ├── CatBoostRegressor (iter=2000, depth=10, lr=0.03, early_stop=50)
        │     → out-of-fold preds + averaged test preds
        └── TrafficNN (256→128→64→1, Adam lr=0.002, 30 epochs)
              → out-of-fold preds + averaged test preds
                     │
               Ridge(alpha=1.0)    ← meta-learner trained on OOF predictions
                     │
            final_predictions.clip(min=0)
                     │
               submission.csv
```

### 1. Missing-value handling
`RoadType` mode, `Weather` mode, and `Temperature` mean are computed once on the training set and reused for the test set (no leakage). Only `Temperature` is actually imputed in the feature-engineering step.

### 2. Feature engineering (two layers)

**Layer 1 — `engineer_features()`**
- `hour`, `dayofweek` (`day % 7`), `is_weekend`, `rush_hour` (8, 9, 17, 18)
- `temp_lane` — Temperature × NumberofLanes
- `geohash_target` — mean training demand per geohash (target encoding; the mapping is learned on train and applied to test, with a `0` fallback for unseen geohashes)

**Layer 2 — `ultra_fe()`**
- `temp_lane_hour` — Temperature × NumberofLanes × hour
- `is_rush_weekend` — rush hour AND weekend
- `geo_freq` — frequency count of each geohash in the training data
- Label-encodes `RoadType`, `Weather`, `LargeVehicles`, `Landmarks` into integer codes (needed for the neural network; CatBoost is separately told which columns are categorical)

### 3. Model — 5-fold stacked ensemble
For each of 5 folds (`KFold`, shuffled, `random_state=42`):
- **CatBoostRegressor** (2000 iterations, depth 10, lr 0.03, early stopping after 50 rounds) trained on raw (unscaled) features with categorical columns passed explicitly.
- **TrafficNN**, a 4-layer MLP (`Linear(256) → ReLU → Dropout(0.2) → Linear(128) → ReLU → Linear(64) → ReLU → Linear(1)`) trained on standardized numeric features with Adam (lr 0.002) for 30 epochs.

Each model produces out-of-fold predictions on its validation fold and averaged predictions on the test set.

### 4. Meta-learner
A `Ridge(alpha=1.0)` regressor is fit on the two out-of-fold prediction vectors (CatBoost, TrafficNN) to learn per-model blend weights. Because it's trained on OOF predictions rather than in-sample ones, this is a proper stacked generalizer.

### 5. Output
Final test predictions are the meta-learner's blend of the two averaged base-model predictions, clipped at a minimum of 0, and written to `submission.csv` with columns `Index, demand`.

## Requirements

- Python 3
- `numpy`, `pandas`, `scikit-learn`
- `torch`
- `catboost` (installed via `pip install catboost` in the notebook)

## Usage

1. Place `train.csv` and `test.csv` alongside the notebook (paths in the notebook default to `/content/sample_data/`, adjust for your environment).
2. Run `trafficdemandpred.ipynb` top to bottom.
3. `submission.csv` is written to the working directory with predicted demand per test row.

## Notes

- The `StandardScaler` is fit once on the full training set and reused inside every fold — a deliberate simplification rather than fitting a fresh scaler per fold.
- `geohash_target` and `geo_freq` are both computed from the full training set, not recomputed per fold, so treat OOF metrics as a slightly optimistic estimate of generalization to genuinely unseen geohashes.
