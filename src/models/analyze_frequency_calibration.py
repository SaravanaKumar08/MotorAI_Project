
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "claim_frequency_dataset.csv"
MODEL_FILE = PROJECT_ROOT / "artifacts" / "best_claim_frequency_model.pkl"
REPORT_FILE = PROJECT_ROOT / "reports" / "claim_frequency_calibration.csv"


def analyze_calibration():

    df = pd.read_csv(DATA_FILE)

    saved_data = joblib.load(MODEL_FILE)

    model = saved_data["model"]
    preprocessor = saved_data["preprocessor"]
    feature_columns = saved_data["feature_columns"]

    X = df[feature_columns]
    y = df["ClaimFrequency"]
    exposure = df["Exposure"]

    # Recreate the same test split
    X_train, X_test, y_train, y_test, exp_train, exp_test = (
        train_test_split(
            X,
            y,
            exposure,
            test_size=0.20,
            random_state=42
        )
    )

    # Predict frequency
    X_test_processed = preprocessor.transform(X_test)

    predictions = model.predict(X_test_processed)

    # Build analysis dataframe
    result = pd.DataFrame({
        "ActualClaimNb": (
            df.loc[X_test.index, "ClaimNb"].values
        ),
        "ActualFrequency": y_test.values,
        "PredictedFrequency": predictions,
        "Exposure": exp_test.values
    })

    # Divide policies into 10 groups by predicted frequency
    result["PredictionDecile"] = pd.qcut(
        result["PredictedFrequency"].rank(method="first"),
        q=10,
        labels=False
    ) + 1

    # Calculate exposure-weighted observed and predicted rates
    calibration = (
        result.groupby("PredictionDecile")
        .apply(
            lambda group: pd.Series({
                "PolicyCount": len(group),
                "TotalClaims": group["ActualClaimNb"].sum(),
                "TotalExposure": group["Exposure"].sum(),
                "ActualClaimRate": (
                    group["ActualClaimNb"].sum()
                    / group["Exposure"].sum()
                ),
                "PredictedClaimRate": (
                    (
                        group["PredictedFrequency"]
                        * group["Exposure"]
                    ).sum()
                    / group["Exposure"].sum()
                )
            }),
            include_groups=False
        )
        .reset_index()
    )

    print("\n========== FREQUENCY CALIBRATION ==========")
    print(calibration.to_string(index=False))

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    calibration.to_csv(REPORT_FILE, index=False)

    print("\nSaved calibration report:", REPORT_FILE)


if __name__ == "__main__":
    analyze_calibration()
