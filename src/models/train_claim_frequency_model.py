
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "claim_frequency_dataset.csv"
MODEL_FILE = PROJECT_ROOT / "artifacts" / "claim_frequency_model.pkl"
REPORT_FILE = PROJECT_ROOT / "reports" / "claim_frequency_model_comparison.csv"


def train_model():

    print("Loading claim frequency dataset...")

    df = pd.read_csv(DATA_FILE)

    # Target: annualized claim frequency
    y = df["ClaimFrequency"]

    # Exposure is used as sample weight, not as a model feature
    exposure = df["Exposure"]

    # Exclude IDpol, ClaimNb, Exposure and ClaimFrequency
    feature_columns = [
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

    X = df[feature_columns]

    numerical_features = [
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

    # Split policies into training and testing
    X_train, X_test, y_train, y_test, exp_train, exp_test = (
        train_test_split(
            X,
            y,
            exposure,
            test_size=0.20,
            random_state=42
        )
    )

    print("\n========== DATA SPLIT ==========")
    print("Training policies:", len(X_train))
    print("Testing policies:", len(X_test))

    # Numerical preprocessing
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    # Combine preprocessing
    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    print("\nPreprocessing training data...")

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Baseline: overall training claim rate
    baseline_rate = np.average(
        y_train,
        weights=exp_train
    )

    baseline_predictions = np.full(
        len(y_test),
        baseline_rate
    )

    baseline_deviance = mean_poisson_deviance(
        y_test,
        baseline_predictions,
        sample_weight=exp_test
    )

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_predictions
    )

    print("\n========== BASELINE RESULTS ==========")
    print(f"Baseline claim rate: {baseline_rate:.6f}")
    print(f"Baseline Poisson deviance: {baseline_deviance:.6f}")
    print(f"Baseline MAE: {baseline_mae:.6f}")

    # Train Poisson Regression
    print("\nTraining Poisson Regression...")

    model = PoissonRegressor(
        alpha=1.0,
        max_iter=1000
    )

    model.fit(
        X_train_processed,
        y_train,
        sample_weight=exp_train
    )

    # Predict claim frequency
    predictions = model.predict(X_test_processed)

    # Avoid zero predictions for Poisson deviance
    predictions = np.maximum(predictions, 1e-10)

    # Evaluate model
    model_deviance = mean_poisson_deviance(
        y_test,
        predictions,
        sample_weight=exp_test
    )

    model_mae = mean_absolute_error(
        y_test,
        predictions
    )

    improvement = (
        (baseline_deviance - model_deviance)
        / baseline_deviance
    ) * 100

    print("\n========== POISSON MODEL RESULTS ==========")
    print(f"Model Poisson deviance: {model_deviance:.6f}")
    print(f"Model MAE: {model_mae:.6f}")
    print(f"Improvement over baseline: {improvement:.2f}%")

    # Save comparison report
    comparison = pd.DataFrame([
        {
            "Model": "Baseline",
            "PoissonDeviance": baseline_deviance,
            "MAE": baseline_mae
        },
        {
            "Model": "Poisson Regression",
            "PoissonDeviance": model_deviance,
            "MAE": model_mae
        }
    ])

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(REPORT_FILE, index=False)

    # Save model and preprocessing together
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "preprocessor": preprocessor,
            "feature_columns": feature_columns,
            "baseline_rate": baseline_rate
        },
        MODEL_FILE
    )

    print("\n========== FILES SAVED ==========")
    print("Model:", MODEL_FILE)
    print("Comparison report:", REPORT_FILE)


if __name__ == "__main__":
    train_model()
