
from pathlib import Path
import pandas as pd

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset paths
POLICY_FILE = PROJECT_ROOT / "data" / "raw" / "freMTPL2freq.csv"
CLAIM_FILE = PROJECT_ROOT / "data" / "raw" / "freMTPL2sev.csv"


def inspect_dataset():
    print("=" * 60)
    print("MOTORAI - DATASET EXPLORATION")
    print("=" * 60)

    # Check whether files exist
    if not POLICY_FILE.exists():
        raise FileNotFoundError(f"Policy dataset not found: {POLICY_FILE}")

    if not CLAIM_FILE.exists():
        raise FileNotFoundError(f"Claim dataset not found: {CLAIM_FILE}")

    # Load CSV files
    policies = pd.read_csv(POLICY_FILE)
    claims = pd.read_csv(CLAIM_FILE)

    # Dataset dimensions
    print("\n1. DATASET SHAPE")
    print("-" * 60)
    print("Policy dataset:", policies.shape)
    print("Claim dataset :", claims.shape)

    # First five records
    print("\n2. POLICY DATA - FIRST 5 RECORDS")
    print("-" * 60)
    print(policies.head())

    print("\n3. CLAIM DATA - FIRST 5 RECORDS")
    print("-" * 60)
    print(claims.head())

    # Column information
    print("\n4. POLICY DATA INFORMATION")
    print("-" * 60)
    policies.info()

    print("\n5. CLAIM DATA INFORMATION")
    print("-" * 60)
    claims.info()

    # Missing values
    print("\n6. MISSING VALUES")
    print("-" * 60)
    print("Policy dataset:")
    print(policies.isnull().sum())

    print("\nClaim dataset:")
    print(claims.isnull().sum())

    # Duplicate records
    print("\n7. DUPLICATE RECORDS")
    print("-" * 60)
    print("Duplicate policy rows:", policies.duplicated().sum())
    print("Duplicate claim rows :", claims.duplicated().sum())

    # Claim statistics
    print("\n8. CLAIM STATISTICS")
    print("-" * 60)
    print(policies["ClaimNb"].describe())
    print("\nClaim amount statistics:")
    print(claims["ClaimAmount"].describe())

    # Relationship between policy and claim datasets
    print("\n9. POLICY AND CLAIM RELATIONSHIP")
    print("-" * 60)

    policy_ids = set(policies["IDpol"])
    claim_ids = set(claims["IDpol"])

    matched_ids = policy_ids.intersection(claim_ids)
    unmatched_claim_ids = claim_ids - policy_ids

    print("Unique policies:", len(policy_ids))
    print("Unique policies with claims:", len(matched_ids))
    print("Claim policy IDs not found in policy data:",
          len(unmatched_claim_ids))

    print("\nDataset inspection completed successfully!")


if __name__ == "__main__":
    inspect_dataset()
