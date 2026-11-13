
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

POLICY_FILE = PROJECT_ROOT / "data" / "raw" / "freMTPL2freq.csv"
CLAIM_FILE = PROJECT_ROOT / "data" / "raw" / "freMTPL2sev.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "claim_severity_dataset.csv"


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

def load_data():

    print("\nLoading datasets...")

    policies = pd.read_csv(POLICY_FILE)
    claims = pd.read_csv(CLAIM_FILE)

    print("Policy records:", len(policies))
    print("Claim records :", len(claims))

    return policies, claims


# --------------------------------------------------
# 3. CHECK DATA QUALITY
# --------------------------------------------------

def check_data_quality(policies, claims):

    print("\n========== DATA QUALITY CHECK ==========")

    print(
        "Duplicate policy rows:",
        policies.duplicated().sum()
    )

    print(
        "Duplicate claim rows:",
        claims.duplicated().sum()
    )

    print(
        "Missing policy values:",
        policies.isnull().sum().sum()
    )

    print(
        "Missing claim values:",
        claims.isnull().sum().sum()
    )

    # Do not automatically remove duplicate claim rows.
    # They need investigation before deciding whether
    # they are genuine repeated claims or data duplicates.


# --------------------------------------------------
# 4. AGGREGATE CLAIMS BY POLICY
# --------------------------------------------------

def aggregate_claims(claims):

    print("\nAggregating claims by policy...")

    policy_claims = (
        claims.groupby("IDpol")
        .agg(
            TotalClaimAmount=("ClaimAmount", "sum"),
            ClaimRecordCount=("ClaimAmount", "count")
        )
        .reset_index()
    )

    print(
        "Unique policies with claim records:",
        len(policy_claims)
    )

    return policy_claims


# --------------------------------------------------
# 5. MERGE POLICY AND CLAIM DATA
# --------------------------------------------------

def merge_datasets(policies, policy_claims):

    print("\nMerging policy and claim datasets...")

    merged = policy_claims.merge(
        policies,
        on="IDpol",
        how="left",
        indicator=True
    )

    matched = merged[merged["_merge"] == "both"].copy()

    unmatched = merged[merged["_merge"] == "left_only"].copy()

    print("Matched policies  :", len(matched))
    print("Unmatched policies:", len(unmatched))

    if not unmatched.empty:
        unmatched[["IDpol", "TotalClaimAmount"]].to_csv(
            OUTPUT_DIR / "unmatched_claim_policies.csv",
            index=False
        )

        print(
            "Unmatched policy report saved:",
            OUTPUT_DIR / "unmatched_claim_policies.csv"
        )

    matched.drop(columns=["_merge"], inplace=True)

    return matched


# --------------------------------------------------
# 6. SELECT FEATURES AND TARGET
# --------------------------------------------------

def prepare_features(data):

    print("\nPreparing ML features...")

    # Target variable
    target_column = "TotalClaimAmount"

    # Features available from policy information.
    # IDpol is excluded because it is only an identifier.
    # ClaimNb is excluded to avoid possible target leakage.
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

    model_data = data[
        feature_columns + [target_column]
    ].copy()

    # Basic validation
    if model_data.isnull().sum().sum() > 0:
        raise ValueError(
            "Missing values found in the prepared dataset."
        )

    if (model_data[target_column] <= 0).any():
        raise ValueError(
            "Non-positive target values found."
        )

    if not model_data[target_column].map(
        lambda value: pd.notna(value) and value < float("inf")
    ).all():
        raise ValueError("Invalid target values found.")

    return model_data


# --------------------------------------------------
# 7. SAVE PROCESSED DATASET
# --------------------------------------------------

def save_dataset(data):

    data.to_csv(OUTPUT_FILE, index=False)

    print("\nProcessed dataset saved successfully!")
    print("File:", OUTPUT_FILE)


# --------------------------------------------------
# 8. MAIN
# --------------------------------------------------

def main():

    print("=" * 60)
    print("MOTORAI - DATA PREPARATION")
    print("=" * 60)

    policies, claims = load_data()

    check_data_quality(policies, claims)

    policy_claims = aggregate_claims(claims)

    merged_data = merge_datasets(
        policies,
        policy_claims
    )

    model_data = prepare_features(merged_data)

    print("\n========== FINAL DATASET ==========")

    print("Dataset shape:", model_data.shape)

    print("\nFeatures:")
    print(model_data.columns.tolist())

    print("\nTarget statistics:")
    print(model_data["TotalClaimAmount"].describe())

    print("\nFirst five records:")
    print(model_data.head())

    save_dataset(model_data)

    print("\n========== PHASE 4 PREPARATION COMPLETED ==========")


if __name__ == "__main__":
    main()
