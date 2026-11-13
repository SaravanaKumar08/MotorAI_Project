
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor


# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "individual_claim_severity.csv"

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports"

ARTIFACT_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


# ==================================================
# 2. FEATURES
# ==================================================

FEATURE_COLUMNS = [
    "Exposure",
    "VehPower",
    "VehAge",
    "DrivAge",
    "BonusMalus",
    "VehBrand",
    "VehGas",
    "Area",
    "Density",
    "Region"
]

NUMERICAL_FEATURES = [
    "Exposure",
    "VehPower",
    "VehAge",
    "DrivAge",
    "BonusMalus",
    "Density"
]

CATEGORICAL_FEATURES = [
    "VehBrand",
    "VehGas",
    "Area",
    "Region"
]


# ==================================================
# 3. LOAD DATA
# ==================================================

def load_dataset():

    print("\nLoading individual claim dataset...")

    data = pd.read_csv(DATA_FILE)

    print("Dataset shape:", data.shape)

    return data


# ==================================================
# 4. PREPARE DATA
# ==================================================

def prepare_data(data):

    X = data[FEATURE_COLUMNS].copy()

    y = data["ClaimAmount"].copy()

    # Used only for group splitting.
    groups = data["IDpol"].copy()

    if y.isna().any():
        raise ValueError("ClaimAmount contains missing values.")

    if (y < 0).any():
        raise ValueError("Negative claim amounts found.")

    return X, y, groups


# ==================================================
# 5. PREPROCESSOR
# ==================================================

def create_preprocessor():

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                NUMERICAL_FEATURES
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES
            )
        ]
    )


# ==================================================
# 6. EVALUATION
# ==================================================

def evaluate_model(
    model_name,
    model,
    X_test,
    y_test,
    baseline_mae
):

    # Direct prediction: no expm1 conversion.
    predictions = model.predict(X_test)

    predictions = np.maximum(predictions, 0)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    improvement = (
        (baseline_mae - mae) / baseline_mae
    ) * 100

    print(f"\n========== {model_name} RESULTS ==========")

    print("MAE :", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2  :", round(r2, 4))

    print(
        "Improvement over baseline:",
        round(improvement, 2),
        "%"
    )

    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Baseline_MAE": baseline_mae,
        "Improvement_Percent": improvement
    }


# ==================================================
# 7. TRAIN MODELS
# ==================================================

def train_models(
    X_train,
    X_test,
    y_train,
    y_test,
    baseline_mae
):

    results = []
    trained_models = {}

    # ----------------------------------------------
    # RANDOM FOREST - MAE ORIENTED
    # ----------------------------------------------

    print("\nTraining Random Forest with absolute_error...")

    random_forest = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor()
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=150,
                    max_depth=12,
                    min_samples_leaf=5,
                    criterion="absolute_error",
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    random_forest.fit(
        X_train,
        y_train
    )

    rf_result = evaluate_model(
        "Random Forest MAE",
        random_forest,
        X_test,
        y_test,
        baseline_mae
    )

    results.append(rf_result)
    trained_models["Random Forest MAE"] = random_forest

    # ----------------------------------------------
    # XGBOOST - MAE OBJECTIVE
    # ----------------------------------------------

    print("\nTraining XGBoost with reg:absoluteerror...")

    xgboost_model = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor()
            ),
            (
                "model",
                XGBRegressor(
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="reg:absoluteerror",
                    eval_metric="mae",
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    xgboost_model.fit(
        X_train,
        y_train
    )

    xgb_result = evaluate_model(
        "XGBoost MAE",
        xgboost_model,
        X_test,
        y_test,
        baseline_mae
    )

    results.append(xgb_result)
    trained_models["XGBoost MAE"] = xgboost_model

    return results, trained_models


# ==================================================
# 8. SAVE RESULTS
# ==================================================

def save_results(results, trained_models):

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="MAE"
    ).reset_index(drop=True)

    print("\n========== MODEL COMPARISON ==========")

    print(results_df.to_string(index=False))

    # Select best ML model.
    best_model_name = results_df.iloc[0]["Model"]

    best_model = trained_models[best_model_name]

    best_model_mae = results_df.iloc[0]["MAE"]

    # Save model.
    model_file = (
        ARTIFACT_DIR
        / "best_individual_claim_model_mae.pkl"
    )

    joblib.dump(
        {
            "model": best_model,
            "model_name": best_model_name,
            "target_transform": "none",
            "feature_columns": FEATURE_COLUMNS
        },
        model_file
    )

    # Save metrics separately from the previous experiment.
    metrics_file = (
        REPORT_DIR
        / "individual_claim_model_mae_comparison.csv"
    )

    results_df.to_csv(
        metrics_file,
        index=False
    )

    print("\n========== BEST ML MODEL ==========")

    print("Best model:", best_model_name)
    print("Best model MAE:", round(best_model_mae, 2))

    print("Saved model:", model_file)
    print("Saved metrics:", metrics_file)


# ==================================================
# 9. MAIN
# ==================================================

def main():

    print("=" * 65)
    print("MOTORAI - INDIVIDUAL CLAIM MAE EXPERIMENT")
    print("=" * 65)

    data = load_dataset()

    X, y, groups = prepare_data(data)

    # Group-based train/test split.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )

    train_indices, test_indices = next(
        splitter.split(
            X,
            y,
            groups=groups
        )
    )

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]

    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    train_groups = set(groups.iloc[train_indices])
    test_groups = set(groups.iloc[test_indices])

    overlap = train_groups.intersection(test_groups)

    print("\n========== GROUP SPLIT VALIDATION ==========")

    print("Training claim records:", len(X_train))
    print("Testing claim records:", len(X_test))

    print("Training policies:", len(train_groups))
    print("Testing policies:", len(test_groups))

    print("Overlapping policy IDs:", len(overlap))

    if overlap:
        raise ValueError("Policy ID leakage detected!")

    print("Data leakage check: PASSED")

    # Median baseline.
    baseline_prediction = y_train.median()

    baseline_predictions = np.full(
        len(y_test),
        baseline_prediction
    )

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_predictions
    )

    print("\n========== BASELINE ==========")

    print("Training median:", round(baseline_prediction, 2))
    print("Baseline MAE:", round(baseline_mae, 2))

    # Train models.
    results, trained_models = train_models(
        X_train,
        X_test,
        y_train,
        y_test,
        baseline_mae
    )

    # Save results.
    save_results(
        results,
        trained_models
    )

    print("\n========== EXPERIMENT COMPLETED ==========")


if __name__ == "__main__":
    main()
