
from pathlib import Path

import joblib
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "best_individual_claim_model_mae.pkl"
)


def main():

    print("=" * 60)
    print("MOTORAI - INDIVIDUAL CLAIM PREDICTION")
    print("=" * 60)

    saved_data = joblib.load(MODEL_FILE)

    model = saved_data["model"]

    print("\nLoaded model:", saved_data["model_name"])

    new_claim = pd.DataFrame([{
        "Exposure": 0.75,
        "VehPower": 7,
        "VehAge": 3,
        "DrivAge": 35,
        "BonusMalus": 60,
        "VehBrand": "B12",
        "VehGas": "Regular",
        "Area": "C",
        "Density": 500,
        "Region": "Ile-de-France"
    }])

    prediction = model.predict(new_claim)

    predicted_amount = max(0, float(prediction[0]))

    print("\n========== INPUT DETAILS ==========")

    print(new_claim.to_string(index=False))

    print("\n========== PREDICTION ==========")

    print(
        "Predicted individual claim amount:",
        round(predicted_amount, 2)
    )


if __name__ == "__main__":
    main()
