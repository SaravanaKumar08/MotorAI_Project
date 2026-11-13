
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "claim_severity_dataset.csv"
)

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports"

ARTIFACT_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

def load_dataset():

    print("\nLoading processed dataset...")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    data = pd.read_csv(DATA_FILE)

    print("Dataset shape:", data.shape)

    return data


# --------------------------------------------------
# 3. SEPARATE FEATURES AND TARGET
# --------------------------------------------------

def prepare_data(data):

    target_column = "TotalClaimAmount"

    feature_columns = [
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

    X = data[feature_columns].copy()
    y = data[target_column].copy()

    numerical_features = [
        "Exposure",
        "VehPower",
        "VehAge",
        "DrivAge",
        "BonusMalus",
        "Density"
    ]

    categorical_features = [
        "VehBrand",
        "VehGas",
        "Area",
        "Region"
    ]

    return (
        X,
        y,
        numerical_features,
        categorical_features
    )


# --------------------------------------------------
# 4. PREPROCESS CATEGORICAL FEATURES
# --------------------------------------------------

def create_preprocessor(
    numerical_features,
    categorical_features
):

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                numerical_features
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ]
    )

    return preprocessor


# --------------------------------------------------
# 5. EVALUATE MODEL
# --------------------------------------------------

def evaluate_model(
    model_name,
    model,
    X_test,
    y_test
):

    # Model was trained on log1p(target).
    # Convert predictions back to original scale.
    log_predictions = model.predict(X_test)

    predictions = np.expm1(log_predictions)

    # Claim amounts cannot be negative.
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

    print(f"\n========== {model_name} RESULTS ==========")

    print("MAE :", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2  :", round(r2, 4))

    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# --------------------------------------------------
# 6. TRAIN MODELS
# --------------------------------------------------

def train_models(
    X_train,
    X_test,
    y_train,
    y_test,
    numerical_features,
    categorical_features
):

    preprocessor = create_preprocessor(
        numerical_features,
        categorical_features
    )

    results = []
    trained_models = {}

    # Transform skewed target.
    y_train_log = np.log1p(y_train)

    # -------------------------------
    # MODEL 1: RANDOM FOREST
    # -------------------------------

    print("\nTraining Random Forest...")

    random_forest = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=15,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    random_forest.fit(
        X_train,
        y_train_log
    )

    rf_result = evaluate_model(
        "Random Forest",
        random_forest,
        X_test,
        y_test
    )

    results.append(rf_result)
    trained_models["Random Forest"] = random_forest

    # -------------------------------
    # MODEL 2: XGBOOST
    # -------------------------------

    print("\nTraining XGBoost...")

    xgboost_model = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor(
                numerical_features,
                categorical_features
            )),
            (
                "model",
                XGBRegressor(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="reg:squarederror",
                    eval_metric="rmse",
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    xgboost_model.fit(
        X_train,
        y_train_log
    )

    xgb_result = evaluate_model(
        "XGBoost",
        xgboost_model,
        X_test,
        y_test
    )

    results.append(xgb_result)
    trained_models["XGBoost"] = xgboost_model

    return results, trained_models


# --------------------------------------------------
# 7. BASELINE COMPARISON
# --------------------------------------------------

def evaluate_baseline(y_train, y_test):

    print("\n========== BASELINE MODEL ==========")

    # Predict the training-set median for every test row.
    baseline_predictions = np.full(
        len(y_test),
        y_train.median()
    )

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_predictions
    )

    print(
        "Baseline MAE:",
        round(baseline_mae, 2)
    )


# --------------------------------------------------
# 8. SAVE BEST MODEL AND METRICS
# --------------------------------------------------

def save_results(results, trained_models):

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="MAE"
    ).reset_index(drop=True)

    print("\n========== MODEL COMPARISON ==========")
    print(results_df.to_string(index=False))

    # Lower MAE means lower average absolute prediction error.
    best_model_name = results_df.iloc[0]["Model"]

    best_model = trained_models[best_model_name]

    model_file = ARTIFACT_DIR / "best_claim_model.pkl"

    joblib.dump(
        {
            "model": best_model,
            "model_name": best_model_name,
            "target_transform": "log1p",
            "feature_columns": [
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
        },
        model_file
    )

    metrics_file = REPORT_DIR / "model_comparison.csv"

    results_df.to_csv(
        metrics_file,
        index=False
    )

    print("\nBest model:", best_model_name)
    print("Saved model:", model_file)
    print("Saved metrics:", metrics_file)


# --------------------------------------------------
# 9. MAIN
# --------------------------------------------------

def main():

    print("=" * 60)
    print("MOTORAI - CLAIM COST MODEL TRAINING")
    print("=" * 60)

    data = load_dataset()

    (
        X,
        y,
        numerical_features,
        categorical_features
    ) = prepare_data(data)

    # Split before fitting preprocessing or models.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print("\n========== TRAIN / TEST SPLIT ==========")

    print("Training records:", len(X_train))
    print("Testing records :", len(X_test))

    evaluate_baseline(
        y_train,
        y_test
    )

    results, trained_models = train_models(
        X_train,
        X_test,
        y_train,
        y_test,
        numerical_features,
        categorical_features
    )

    save_results(
        results,
        trained_models
    )

    print("\n========== MODEL TRAINING COMPLETED ==========")


if __name__ == "__main__":
    main()
