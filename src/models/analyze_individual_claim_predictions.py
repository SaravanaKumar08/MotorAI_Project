
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "individual_claim_severity.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "best_individual_claim_model_mae.pkl"
)

REPORT_DIR = PROJECT_ROOT / "reports"


def main():

    print("=" * 65)
    print("MOTORAI - INDIVIDUAL CLAIM ERROR ANALYSIS")
    print("=" * 65)

    # Load data and model.
    data = pd.read_csv(DATA_FILE)

    saved_data = joblib.load(MODEL_FILE)

    model = saved_data["model"]

    feature_columns = saved_data["feature_columns"]

    X = data[feature_columns]

    y = data["ClaimAmount"]

    groups = data["IDpol"]

    # Recreate the same grouped split.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )

    train_indices, test_indices = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]

    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    # Predict using saved model.
    predictions = model.predict(X_test)

    predictions = np.maximum(predictions, 0)

    baseline_prediction = y_train.median()

    baseline_predictions = np.full(
        len(y_test),
        baseline_prediction
    )

    model_mae = mean_absolute_error(
        y_test,
        predictions
    )

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_predictions
    )

    print("\n========== OVERALL PERFORMANCE ==========")

    print("Model MAE:", round(model_mae, 2))

    print("Baseline MAE:", round(baseline_mae, 2))

    # Create prediction report.
    result = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": predictions
    })

    result["AbsoluteError"] = abs(
        result["Actual"] - result["Predicted"]
    )

    result["ClaimCategory"] = pd.cut(
        result["Actual"],
        bins=[-1, 1000, 5000, 20000, 100000, np.inf],
        labels=[
            "0-1000",
            "1000-5000",
            "5000-20000",
            "20000-100000",
            "Above 100000"
        ]
    )

    # Group-wise analysis.
    analysis = result.groupby(
        "ClaimCategory",
        observed=False
    ).agg(
        ClaimCount=("Actual", "count"),
        AverageActual=("Actual", "mean"),
        AveragePredicted=("Predicted", "mean"),
        AverageAbsoluteError=("AbsoluteError", "mean")
    ).reset_index()

    print("\n========== CLAIM CATEGORY ANALYSIS ==========")

    print(analysis.to_string(index=False))

    # Save report.
    output_file = (
        REPORT_DIR
        / "individual_claim_prediction_error_analysis.csv"
    )

    analysis.to_csv(
        output_file,
        index=False
    )

    print("\nSaved analysis:", output_file)

    print("\n========== ANALYSIS COMPLETED ==========")


if __name__ == "__main__":
    main()
