
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

POLICY_FILE = PROJECT_ROOT / "data" / "raw" / "freMTPL2freq.csv"
CLAIM_FILE = PROJECT_ROOT / "data" / "raw" / "freMTPL2sev.csv"

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

def load_data():

    if not POLICY_FILE.exists():
        raise FileNotFoundError(f"Missing file: {POLICY_FILE}")

    if not CLAIM_FILE.exists():
        raise FileNotFoundError(f"Missing file: {CLAIM_FILE}")

    policies = pd.read_csv(POLICY_FILE)
    claims = pd.read_csv(CLAIM_FILE)

    return policies, claims


# --------------------------------------------------
# 3. BASIC DATA ANALYSIS
# --------------------------------------------------

def basic_analysis(policies, claims):

    print("\n========== DATASET OVERVIEW ==========")

    print("Policy shape:", policies.shape)
    print("Claim shape :", claims.shape)

    print("\nPolicy columns:")
    print(policies.columns.tolist())

    print("\nClaim columns:")
    print(claims.columns.tolist())

    print("\n========== POLICY STATISTICS ==========")

    print(policies.describe(include="all").T)

    print("\n========== CLAIM STATISTICS ==========")

    print(claims["ClaimAmount"].describe())

    print("\n========== MISSING VALUES ==========")

    print("Policy missing values:")
    print(policies.isnull().sum())

    print("\nClaim missing values:")
    print(claims.isnull().sum())

    print("\n========== DUPLICATE ROWS ==========")

    print("Policy duplicates:", policies.duplicated().sum())
    print("Claim duplicates :", claims.duplicated().sum())


# --------------------------------------------------
# 4. CLAIM FREQUENCY ANALYSIS
# --------------------------------------------------

def claim_frequency_analysis(policies):

    print("\n========== CLAIM FREQUENCY ==========")

    claim_distribution = (
        policies["ClaimNb"]
        .value_counts()
        .sort_index()
    )

    print(claim_distribution)

    zero_claims = (policies["ClaimNb"] == 0).sum()
    policies_with_claims = (policies["ClaimNb"] > 0).sum()

    print("\nPolicies without claims:", zero_claims)
    print("Policies with claims   :", policies_with_claims)

    total_exposure = policies["Exposure"].sum()

    if total_exposure > 0:
        frequency = policies["ClaimNb"].sum() / total_exposure
        print("Overall claim frequency:", round(frequency, 5))

    # Save claim distribution chart
    plt.figure(figsize=(8, 5))

    claim_distribution.plot(kind="bar")

    plt.title("Number of Claims per Policy")
    plt.xlabel("Number of Claims")
    plt.ylabel("Number of Policies")
    plt.tight_layout()

    plt.savefig(REPORT_DIR / "claim_frequency.png")
    plt.close()


# --------------------------------------------------
# 5. CLAIM SEVERITY ANALYSIS
# --------------------------------------------------

def claim_severity_analysis(claims):

    print("\n========== CLAIM SEVERITY ==========")

    print("Average claim amount:",
          round(claims["ClaimAmount"].mean(), 2))

    print("Median claim amount:",
          round(claims["ClaimAmount"].median(), 2))

    print("Maximum claim amount:",
          round(claims["ClaimAmount"].max(), 2))

    print("Minimum claim amount:",
          round(claims["ClaimAmount"].min(), 2))

    # Claim amount distribution
    plt.figure(figsize=(8, 5))

    plt.hist(
        claims["ClaimAmount"].dropna(),
        bins=50
    )

    plt.title("Distribution of Claim Amounts")
    plt.xlabel("Claim Amount")
    plt.ylabel("Number of Claims")
    plt.tight_layout()

    plt.savefig(REPORT_DIR / "claim_amount_distribution.png")
    plt.close()

    # Log scale helps us inspect a highly skewed distribution
    positive_amounts = claims.loc[
        claims["ClaimAmount"] > 0, "ClaimAmount"
    ]

    plt.figure(figsize=(8, 5))

    plt.hist(
        positive_amounts,
        bins=50,
        log=True
    )

    plt.title("Claim Amount Distribution - Log Count Scale")
    plt.xlabel("Claim Amount")
    plt.ylabel("Number of Claims (log scale)")
    plt.tight_layout()

    plt.savefig(REPORT_DIR / "claim_amount_log_scale.png")
    plt.close()


# --------------------------------------------------
# 6. DRIVER AGE ANALYSIS
# --------------------------------------------------

def driver_age_analysis(policies):

    print("\n========== DRIVER AGE ANALYSIS ==========")

    age_groups = [18, 25, 35, 45, 55, 65, 75, 101]

    age_labels = [
        "18-24",
        "25-34",
        "35-44",
        "45-54",
        "55-64",
        "65-74",
        "75+"
    ]

    data = policies.copy()

    data["DriverAgeGroup"] = pd.cut(
        data["DrivAge"],
        bins=age_groups,
        labels=age_labels,
        right=False
    )

    age_analysis = data.groupby(
        "DriverAgeGroup",
        observed=True
    ).agg(
        TotalPolicies=("IDpol", "count"),
        TotalClaims=("ClaimNb", "sum"),
        TotalExposure=("Exposure", "sum")
    )

    age_analysis["ClaimFrequency"] = (
        age_analysis["TotalClaims"] /
        age_analysis["TotalExposure"]
    )

    print(age_analysis)

    age_analysis.to_csv(
        REPORT_DIR / "driver_age_analysis.csv"
    )


# --------------------------------------------------
# 7. DATA QUALITY CHECKS
# --------------------------------------------------

def data_quality_checks(policies, claims):

    print("\n========== DATA QUALITY CHECKS ==========")

    print("Policies with non-positive exposure:",
          (policies["Exposure"] <= 0).sum())

    print("Policies with negative claim count:",
          (policies["ClaimNb"] < 0).sum())

    print("Claims with non-positive amount:",
          (claims["ClaimAmount"] <= 0).sum())

    policy_ids = set(policies["IDpol"])
    claim_ids = set(claims["IDpol"])

    print("Unique policy IDs:", len(policy_ids))
    print("Unique claim policy IDs:", len(claim_ids))

    print("Claim policy IDs not in policy data:",
          len(claim_ids - policy_ids))

    print("Policy IDs without claim records:",
          len(policy_ids - claim_ids))

    # IQR-based outlier inspection
    q1 = claims["ClaimAmount"].quantile(0.25)
    q3 = claims["ClaimAmount"].quantile(0.75)

    iqr = q3 - q1

    upper_limit = q3 + (1.5 * iqr)

    outliers = claims[
        claims["ClaimAmount"] > upper_limit
    ]

    print("Upper IQR outlier threshold:", round(upper_limit, 2))
    print("Potential high-value claim outliers:", len(outliers))


# --------------------------------------------------
# 8. CLAIM AND POLICY DATA RELATIONSHIP
# --------------------------------------------------

def merge_analysis(policies, claims):

    print("\n========== CLAIM AND POLICY MERGE ==========")

    merged = claims.merge(
        policies.drop(columns=["ClaimNb"]),
        on="IDpol",
        how="left",
        indicator=True
    )

    print("Merged claim rows:", len(merged))

    print("\nMerge status:")
    print(merged["_merge"].value_counts())

    print("\nMissing policy details after merge:")
    print(merged.drop(columns=["_merge"]).isnull().sum())

    merged.drop(columns=["_merge"]).to_csv(
        REPORT_DIR / "claim_policy_analysis.csv",
        index=False
    )


# --------------------------------------------------
# 9. MAIN FUNCTION
# --------------------------------------------------

def main():

    print("=" * 60)
    print("MOTORAI - EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    policies, claims = load_data()

    basic_analysis(policies, claims)

    claim_frequency_analysis(policies)

    claim_severity_analysis(claims)

    driver_age_analysis(policies)

    data_quality_checks(policies, claims)

    merge_analysis(policies, claims)

    print("\n========== EDA COMPLETED ==========")
    print("Reports saved inside:", REPORT_DIR)


if __name__ == "__main__":
    main()
