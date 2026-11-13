
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "freMTPL2freq.csv"

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "policy_segmentation_dataset.csv"
)


def prepare_policy_segmentation():

    print("Loading policy dataset...")

    df = pd.read_csv(INPUT_FILE)

    print("Original dataset shape:", df.shape)

    # Features used for clustering
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

    # Keep these only for identification and later cluster analysis
    analysis_columns = [
        "IDpol",
        "ClaimNb",
        "Exposure"
    ]

    required_columns = feature_columns + analysis_columns

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    # Select required columns
    df = df[required_columns].copy()

    # Calculate claim frequency for later cluster evaluation
    df["ClaimFrequency"] = df["ClaimNb"] / df["Exposure"]

    # Check duplicate policy IDs
    duplicate_count = df["IDpol"].duplicated().sum()

    print("Duplicate policy IDs:", duplicate_count)

    if duplicate_count > 0:
        raise ValueError("Duplicate policy IDs found.")

    # Check missing values
    print("\nMissing values:")
    print(df.isnull().sum())

    # Feature summary
    print("\nNumerical feature summary:")
    print(df[[
        "VehPower",
        "VehAge",
        "DrivAge",
        "BonusMalus",
        "Density"
    ]].describe())

    print("\nCategorical feature summary:")

    for column in ["VehBrand", "VehGas", "Area", "Region"]:
        print(f"\n{column}:")
        print(df[column].value_counts().head(10))

    # Save prepared dataset
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print("\n========== SEGMENTATION DATASET SUMMARY ==========")

    print("Final dataset shape:", df.shape)

    print("Total policies:", len(df))

    print(
        "Unique policies:",
        df["IDpol"].nunique()
    )

    print(
        "Average claim frequency:",
        round(df["ClaimFrequency"].mean(), 6)
    )

    print("\nSaved dataset:", OUTPUT_FILE)

    print("\nClustering features:")
    for column in feature_columns:
        print("-", column)

    print("\nExcluded from clustering:")
    print("- IDpol")

    print("- ClaimNb")
    print("- Exposure")
    print("- ClaimFrequency")


if __name__ == "__main__":
    prepare_policy_segmentation()
