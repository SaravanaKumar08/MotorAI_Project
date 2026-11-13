
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

POLICY_FILE = PROJECT_ROOT / "data" / "raw" / "freMTPL2freq.csv"
CLAIM_FILE = PROJECT_ROOT / "data" / "raw" / "freMTPL2sev.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "individual_claim_severity.csv"


def main():

    print("=" * 60)
    print("MOTORAI - INDIVIDUAL CLAIM DATA PREPARATION")
    print("=" * 60)

    policies = pd.read_csv(POLICY_FILE)
    claims = pd.read_csv(CLAIM_FILE)

    print("\nOriginal claim records:", len(claims))

    # Attach policy information to each individual claim.
    data = claims.merge(
        policies,
        on="IDpol",
        how="inner",
        validate="many_to_one"
    )

    print("Matched individual claims:", len(data))

    # Keep IDpol for group-based train/test splitting.
    # Do not use IDpol as a model input.
    feature_columns = [
        "IDpol",
        "Exposure",
        "VehPower",
        "VehAge",
        "DrivAge",
        "BonusMalus",
        "VehBrand",
        "VehGas",
        "Area",
        "Density",
        "Region",
        "ClaimAmount"
    ]

    data = data[feature_columns].copy()

    if data.isnull().sum().sum() > 0:
        raise ValueError("Missing values found.")

    if (data["ClaimAmount"] <= 0).any():
        raise ValueError("Non-positive claim amounts found.")

    data.to_csv(OUTPUT_FILE, index=False)

    print("\nFinal dataset shape:", data.shape)
    print("Unique policies:", data["IDpol"].nunique())

    print("\nIndividual claim amount statistics:")
    print(data["ClaimAmount"].describe())

    print("\nFirst five records:")
    print(data.head())

    print("\nSaved:", OUTPUT_FILE)
    print("\nINDIVIDUAL CLAIM DATA PREPARATION COMPLETED")


if __name__ == "__main__":
    main()
