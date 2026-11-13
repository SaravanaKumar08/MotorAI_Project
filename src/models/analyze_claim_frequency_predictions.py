
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_poisson_deviance
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "claim_frequency_dataset.csv"
MODEL_FILE = PROJECT_ROOT / "artifacts" / "best_claim_frequency_model.pkl"
REPORT_FILE = PROJECT_ROOT / "reports" / "claim_frequency_prediction_analysis.csv"


def analyze_predictions():

    print("Loading dataset and trained model...")

    df = pd.read_csv(DATA_FILE)

    saved_data = joblib.load(MODEL_FILE)

    model = saved_data["model"]
    preprocessor = saved_data["preprocessor"]
    feature_columns = saved_data["feature_columns"]

    X = df[feature_columns]
    y = df["ClaimFrequency"]
    exposure = df["Exposure"]

    # Recreate the original test split
    X_train, X_test, y_train, y_test, exp_train, exp_test = (
        train_test_split(
            X,
            y,
            exposure,
            test_size=0.20,
            random_state=42
        )
    )

    # Transform test data
    X_test_processed = preprocessor.transform(X_test)

    # Predict claim frequency
    predictions = model.predict(X_test_processed)
    predictions = np.maximum(predictions, 1e-10)

    # Overall metrics
    deviance = mean_poisson_deviance(
        y_test,
        predictions,
        sample_weight=exp_test
    )

    mae = mean_absolute_error(y_test, predictions)

    print("\n========== OVERALL PERFORMANCE ==========")
    print(f"Test policies: {len(y_test)}")
    print(f"Poisson deviance: {deviance:.6f}")
    print(f"MAE: {mae:.6f}")

    print("\nActual frequency statistics:")
    print(y_test.describe())

    print("\nPredicted frequency statistics:")
    print(pd.Series(predictions).describe())

    # Create analysis dataframe
    analysis = pd.DataFrame({
        "ActualFrequency": y_test.values,
        "PredictedFrequency": predictions,
        "Exposure": exp_test.values
    })

    # Group actual frequency values
    bins = [-0.000001, 0, 1, 5, float("inf")]

    labels = [
        "Zero claims",
        "0 to 1",
        "1 to 5",
        "Above 5"
    ]

    analysis["FrequencyCategory"] = pd.cut(
        analysis["ActualFrequency"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    # Category-wise analysis
    category_analysis = (
        analysis.groupby(
            "FrequencyCategory",
            observed=False
        )
        .apply(
            lambda group: pd.Series({
                "PolicyCount": len(group),
                "AverageActualFrequency": group["ActualFrequency"].mean(),
                "AveragePredictedFrequency": group["PredictedFrequency"].mean(),
                "AverageAbsoluteError": (
                    group["ActualFrequency"] -
                    group["PredictedFrequency"]
                ).abs().mean(),
                "TotalExposure": group["Exposure"].sum()
            }),
            include_groups=False
        )
        .reset_index()
    )

    print("\n========== FREQUENCY CATEGORY ANALYSIS ==========")
    print(category_analysis.to_string(index=False))

    # Save report
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    category_analysis.to_csv(
        REPORT_FILE,
        index=False
    )

    print("\nSaved analysis:", REPORT_FILE)


if __name__ == "__main__":
    analyze_predictions()
