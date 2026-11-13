
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "claim_frequency_dataset.csv"
MODEL_FILE = PROJECT_ROOT / "artifacts" / "best_claim_frequency_model.pkl"
REPORT_FILE = PROJECT_ROOT / "reports" / "claim_frequency_model_comparison.csv"


def evaluate_model(name, actual, predictions, exposure):

    predictions = np.maximum(predictions, 1e-10)

    deviance = mean_poisson_deviance(
        actual,
        predictions,
        sample_weight=exposure
    )

    mae = mean_absolute_error(actual, predictions)

    print(f"\n{name}")
    print(f"Poisson Deviance: {deviance:.6f}")
    print(f"MAE: {mae:.6f}")

    return {
        "Model": name,
        "PoissonDeviance": deviance,
        "MAE": mae
    }


def train_models():

    print("Loading dataset...")

    df = pd.read_csv(DATA_FILE)

    feature_columns = [
        "VehPower", "VehAge", "DrivAge", "BonusMalus",
        "VehBrand", "VehGas", "Area", "Density", "Region"
    ]

    numerical_features = [
        "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"
    ]

    categorical_features = [
        "VehBrand", "VehGas", "Area", "Region"
    ]

    X = df[feature_columns]
    y = df["ClaimFrequency"]
    exposure = df["Exposure"]

    # Same split for fair comparison
    X_train, X_test, y_train, y_test, exp_train, exp_test = (
        train_test_split(
            X,
            y,
            exposure,
            test_size=0.20,
            random_state=42
        )
    )

    print("Training policies:", len(X_train))
    print("Testing policies:", len(X_test))

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    print("\nPreprocessing data...")

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    results = []

    # 1. Baseline
    baseline_rate = np.average(y_train, weights=exp_train)

    baseline_predictions = np.full(len(y_test), baseline_rate)

    results.append(
        evaluate_model(
            "Baseline",
            y_test,
            baseline_predictions,
            exp_test
        )
    )

    # 2. Poisson Regression
    print("\nTraining Poisson Regression...")

    poisson_model = PoissonRegressor(
        alpha=1.0,
        max_iter=1000
    )

    poisson_model.fit(
        X_train_processed,
        y_train,
        sample_weight=exp_train
    )

    poisson_predictions = poisson_model.predict(X_test_processed)

    results.append(
        evaluate_model(
            "Poisson Regression",
            y_test,
            poisson_predictions,
            exp_test
        )
    )

    # 3. Random Forest
    print("\nTraining Random Forest...")

    rf_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(
        X_train_processed,
        y_train,
        sample_weight=exp_train
    )

    rf_predictions = rf_model.predict(X_test_processed)

    results.append(
        evaluate_model(
            "Random Forest",
            y_test,
            rf_predictions,
            exp_test
        )
    )

    # Compare results
    comparison = pd.DataFrame(results)

    comparison = comparison.sort_values(
        by="PoissonDeviance"
    )

    baseline_deviance = comparison.loc[
        comparison["Model"] == "Baseline",
        "PoissonDeviance"
    ].iloc[0]

    comparison["ImprovementVsBaselinePercent"] = (
        (baseline_deviance - comparison["PoissonDeviance"])
        / baseline_deviance
    ) * 100

    print("\n========== FINAL MODEL COMPARISON ==========")
    print(comparison.to_string(index=False))

    # Save report
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(REPORT_FILE, index=False)

    # Save best model
    best_name = comparison.iloc[0]["Model"]

    if best_name == "Baseline":
        print("\nBaseline is still the best model.")
        print("No ML model will be selected as the best model.")

    else:
        best_model = (
            poisson_model
            if best_name == "Poisson Regression"
            else rf_model
        )

        MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "model": best_model,
                "preprocessor": preprocessor,
                "model_name": best_name,
                "feature_columns": feature_columns,
                "baseline_rate": baseline_rate
            },
            MODEL_FILE
        )

        print("\nBest model:", best_name)
        print("Saved model:", MODEL_FILE)

    print("\nSaved comparison report:", REPORT_FILE)


if __name__ == "__main__":
    train_models()
