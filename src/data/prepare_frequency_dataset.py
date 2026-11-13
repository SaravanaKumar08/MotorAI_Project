
from pathlib import Path

import pandas as pd


# Project folders
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "freMTPL2freq.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "claim_frequency_dataset.csv"


def prepare_frequency_dataset():

    print("Loading frequency dataset...")

    df = pd.read_csv(INPUT_FILE)

    print("Original dataset shape:", df.shape)

    # Required columns
    required_columns = [
        "IDpol",
        "ClaimNb",
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

    # Validate columns
    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    df = df[required_columns].copy()

    # Check duplicate policy IDs
    duplicate_count = df["IDpol"].duplicated().sum()

    print("Duplicate policy IDs:", duplicate_count)

    if duplicate_count > 0:
        raise ValueError("Duplicate policy IDs found. Investigate before proceeding.")

    # Validate exposure
    invalid_exposure = (df["Exposure"] <= 0).sum()
    print("Policies with invalid exposure:", invalid_exposure)

    if invalid_exposure > 0:
        raise ValueError("Invalid exposure values found.")

    # Validate claim count
    if (df["ClaimNb"] < 0).any():
        raise ValueError("Negative claim counts found.")

    # Calculate claim frequency
    df["ClaimFrequency"] = df["ClaimNb"] / df["Exposure"]

    # Save processed dataset
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n========== DATASET PREPARATION SUMMARY ==========")

    print("Final dataset shape:", df.shape)

    print("\nClaim count distribution:")
    print(df["ClaimNb"].value_counts().sort_index())

    print("\nClaim frequency statistics:")
    print(df["ClaimFrequency"].describe())

    print("\nTotal policies:", len(df))
    print("Policies with claims:", (df["ClaimNb"] > 0).sum())
    print("Policies without claims:", (df["ClaimNb"] == 0).sum())

    print("\nSaved dataset:", OUTPUT_FILE)


if __name__ == "__main__":
    prepare_frequency_dataset()
