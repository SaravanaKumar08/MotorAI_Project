
from pathlib import Path

import joblib
import pandas as pd

from sklearn.cluster import MiniBatchKMeans


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "policy_segmentation_dataset.csv"
)

PREPROCESSOR_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "segmentation_preprocessor.pkl"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "policy_segmentation_model.pkl"
)

ASSIGNED_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "policy_segments.csv"
)

PROFILE_FILE = (
    PROJECT_ROOT
    / "reports"
    / "policy_segment_profiles.csv"
)


def create_policy_segments():

    print("Loading policy dataset...")

    df = pd.read_csv(DATA_FILE)

    # Load fitted preprocessing pipeline
    saved_preprocessor = joblib.load(PREPROCESSOR_FILE)

    preprocessor = saved_preprocessor["preprocessor"]

    feature_columns = (
        saved_preprocessor["numerical_features"]
        + saved_preprocessor["categorical_features"]
    )

    X = df[feature_columns].copy()

    # Apply the same density transformation used during cluster selection
    import numpy as np

    X["Density"] = np.log1p(X["Density"])

    print("\nTransforming policy features...")

    X_processed = preprocessor.transform(X)

    print("Processed shape:", X_processed.shape)

    # Train MiniBatch K-Means using all policies
    print("\nTraining MiniBatch K-Means with 3 clusters...")

    model = MiniBatchKMeans(
        n_clusters=3,
        random_state=42,
        batch_size=2048,
        n_init=5,
        max_iter=100
    )

    cluster_labels = model.fit_predict(X_processed)

    # Attach cluster labels to original dataset
    df["Cluster"] = cluster_labels

    # Save individual policy assignments
    ASSIGNED_FILE.parent.mkdir(parents=True, exist_ok=True)

    df[[
        "IDpol",
        "Cluster",
        "ClaimNb",
        "Exposure",
        "ClaimFrequency"
    ]].to_csv(ASSIGNED_FILE, index=False)

    # Numerical cluster profiles
    profile = (
        df.groupby("Cluster")
        .agg(
            PolicyCount=("IDpol", "count"),
            TotalClaims=("ClaimNb", "sum"),
            TotalExposure=("Exposure", "sum"),
            AverageDriverAge=("DrivAge", "mean"),
            AverageVehicleAge=("VehAge", "mean"),
            AverageVehiclePower=("VehPower", "mean"),
            AverageBonusMalus=("BonusMalus", "mean"),
            AverageDensity=("Density", "mean"),
            AverageClaimFrequency=("ClaimFrequency", "mean")
        )
        .reset_index()
    )

    # Exposure-weighted observed claim rate
    profile["ObservedClaimRate"] = (
        profile["TotalClaims"] / profile["TotalExposure"]
    )

    # Most common categorical characteristics
    categorical_features = [
        "VehBrand",
        "VehGas",
        "Area",
        "Region"
    ]

    for column in categorical_features:

        modes = (
            df.groupby("Cluster")[column]
            .agg(lambda values: values.mode().iloc[0])
            .rename(f"MostCommon_{column}")
        )

        profile = profile.merge(
            modes,
            on="Cluster",
            how="left"
        )

    # Save cluster profiles
    PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)

    profile.to_csv(PROFILE_FILE, index=False)

    # Save model and preprocessing pipeline together
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "preprocessor": preprocessor,
            "feature_columns": feature_columns,
            "density_log_transform": True
        },
        MODEL_FILE
    )

    print("\n========== POLICY SEGMENT SUMMARY ==========")
    print(profile.to_string(index=False))

    print("\n========== FILES SAVED ==========")
    print("Policy assignments:", ASSIGNED_FILE)
    print("Segment profiles:", PROFILE_FILE)
    print("Segmentation model:", MODEL_FILE)


if __name__ == "__main__":
    create_policy_segments()
