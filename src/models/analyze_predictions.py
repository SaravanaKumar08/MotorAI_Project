
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "claim_severity_dataset.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "best_claim_model.pkl"
)


def main():

    print("=" * 60)
    print("MOTORAI - PREDICTION ERROR ANALYSIS")
    print("=" * 60)

    # Load dataset
    data = pd.read_csv(DATA_FILE)

    target = "TotalClaimAmount"

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

    X = data[feature_columns]
    y = data[target]

    # Use exactly the same split as training
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # Load saved model
    saved_model = joblib.load(MODEL_FILE)

    model = saved_model["model"]

    # Predict
    log_predictions = model.predict(X_test)

    predictions = np.expm1(log_predictions)

    predictions = np.maximum(predictions, 0)

    # Compare with baseline
    baseline_predictions = np.full(
        len(y_test),
        y_train.median()
    )

    print("\n========== MAE COMPARISON ==========")

    print(
        "Baseline MAE:",
        round(
            mean_absolute_error(
                y_test,
                baseline_predictions
            ),
            2
        )
    )

    print(
        "Model MAE:",
        round(
            mean_absolute_error(
                y_test,
                predictions
            ),
            2
        )
    )

    # Create prediction report
    results = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": predictions
    })

    results["AbsoluteError"] = abs(
        results["Actual"] - results["Predicted"]
    )

    results["ActualGroup"] = pd.cut(
        results["Actual"],
        bins=[0, 1000, 5000, 20000, 100000, float("inf")],
        labels=[
            "0-1000",
            "1000-5000",
            "5000-20000",
            "20000-100000",
            "Above 100000"
        ]
    )

    print("\n========== ERROR BY CLAIM AMOUNT ==========")

    print(
        results.groupby(
            "ActualGroup",
            observed=False
        ).agg(
            Records=("Actual", "count"),
            AverageActual=("Actual", "mean"),
            AveragePredicted=("Predicted", "mean"),
            AverageAbsoluteError=("AbsoluteError", "mean")
        ).round(2)
    )

    # Show largest prediction errors
    print("\n========== TOP 10 PREDICTION ERRORS ==========")

    print(
        results.sort_values(
            by="AbsoluteError",
            ascending=False
        ).head(10).round(2).to_string(index=False)
    )

    # Save report
    output_file = (
        PROJECT_ROOT
        / "reports"
        / "prediction_error_analysis.csv"
    )

    results.to_csv(output_file, index=False)

    print("\nReport saved:", output_file)


if __name__ == "__main__":
    main()
