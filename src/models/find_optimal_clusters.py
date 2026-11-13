
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "policy_segmentation_dataset.csv"
)

ARTIFACT_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "segmentation_preprocessor.pkl"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "optimal_cluster_analysis.csv"
)


def find_optimal_clusters():

    print("Loading segmentation dataset...")

    df = pd.read_csv(DATA_FILE)

    numerical_features = [
        "VehPower",
        "VehAge",
        "DrivAge",
        "BonusMalus",
        "Density"
    ]

    categorical_features = [
        "VehBrand",
        "VehGas",
        "Area",
        "Region"
    ]

    X = df[numerical_features + categorical_features].copy()

    # Reduce the skew in population density
    X["Density"] = np.log1p(X["Density"])

    # Scale numerical features
    numerical_pipeline = StandardScaler()

    # Encode categorical features
    categorical_pipeline = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, numerical_features),
            ("categorical", categorical_pipeline, categorical_features)
        ],
        sparse_threshold=1.0
    )

    print("\nPreprocessing features...")

    X_processed = preprocessor.fit_transform(X)

    print("Processed feature shape:", X_processed.shape)

    # Use a manageable sample for cluster selection
    rng = np.random.RandomState(42)

    sample_size = min(50000, X_processed.shape[0])

    sample_indices = rng.choice(
        X_processed.shape[0],
        size=sample_size,
        replace=False
    )

    X_sample = X_processed[sample_indices]

    # Separate fitting and validation samples
    X_fit, X_validation = train_test_split(
        X_sample,
        test_size=0.20,
        random_state=42
    )

    # Silhouette score is expensive, so use 3000 validation rows
    silhouette_size = min(3000, X_validation.shape[0])

    silhouette_indices = rng.choice(
        X_validation.shape[0],
        size=silhouette_size,
        replace=False
    )

    X_silhouette = X_validation[silhouette_indices]

    results = []

    print("\n========== CLUSTER EVALUATION ==========")

    for k in range(2, 9):

        print(f"\nTesting {k} clusters...")

        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=42,
            batch_size=2048,
            n_init=3,
            max_iter=100
        )

        model.fit(X_fit)

        # Elbow method: lower inertia is better
        inertia = model.inertia_

        # Silhouette score: higher is better
        silhouette_labels = model.predict(X_silhouette)

        score = silhouette_score(
            X_silhouette,
            silhouette_labels,
            sample_size=None
        )

        print(f"Inertia: {inertia:.2f}")
        print(f"Silhouette Score: {score:.6f}")

        results.append({
            "Clusters": k,
            "Inertia": inertia,
            "SilhouetteScore": score
        })

    results_df = pd.DataFrame(results)

    print("\n========== FINAL CLUSTER COMPARISON ==========")
    print(results_df.to_string(index=False))

    # Save report
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(
        REPORT_FILE,
        index=False
    )

    # Save fitted preprocessing pipeline
    ARTIFACT_FILE.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "preprocessor": preprocessor,
            "numerical_features": numerical_features,
            "categorical_features": categorical_features
        },
        ARTIFACT_FILE
    )

    print("\nSaved cluster analysis:", REPORT_FILE)
    print("Saved preprocessor:", ARTIFACT_FILE)


if __name__ == "__main__":
    find_optimal_clusters()
