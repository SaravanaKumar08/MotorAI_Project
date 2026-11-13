
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "individual_claim_severity.csv"

MODEL_FILE = PROJECT_ROOT / "artifacts" / "ann_individual_claim_model.pth"
PREPROCESSOR_FILE = PROJECT_ROOT / "artifacts" / "ann_claim_preprocessor.pkl"
REPORT_FILE = PROJECT_ROOT / "reports" / "ann_claim_model_metrics.csv"


# --------------------------------------------------
# 2. REPRODUCIBILITY
# --------------------------------------------------

np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device("cpu")

print("Using device:", DEVICE)


# --------------------------------------------------
# 3. LOAD DATASET
# --------------------------------------------------

df = pd.read_csv(DATA_FILE)

print("\nDataset shape:", df.shape)

target_column = "ClaimAmount"
group_column = "IDpol"

numerical_features = [
    "Exposure",
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

feature_columns = numerical_features + categorical_features

X = df[feature_columns].copy()
y = df[target_column].astype(float).values
groups = df[group_column].values


# --------------------------------------------------
# 4. TRAIN / TEST SPLIT
# --------------------------------------------------

# Claims belonging to the same policy must stay
# in either train or test, never both.

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_indices, test_indices = next(
    splitter.split(X, y, groups)
)

X_train = X.iloc[train_indices].copy()
X_test = X.iloc[test_indices].copy()

y_train = y[train_indices]
y_test = y[test_indices]

train_groups = set(groups[train_indices])
test_groups = set(groups[test_indices])

print("\n========== DATA SPLIT ==========")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Training policies:", len(train_groups))
print("Testing policies:", len(test_groups))
print("Policy overlap:", len(train_groups.intersection(test_groups)))


# --------------------------------------------------
# 5. DATA PREPROCESSING
# --------------------------------------------------

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

X_train_processed = np.asarray(
    X_train_processed, dtype=np.float32
)

X_test_processed = np.asarray(
    X_test_processed, dtype=np.float32
)

print("\nProcessed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)


# --------------------------------------------------
# 6. TARGET TRANSFORMATION
# --------------------------------------------------

# Claim amounts have a highly skewed distribution.
# log1p helps reduce the influence of extreme values.

y_train_log = np.log1p(y_train)

y_train_tensor = torch.tensor(
    y_train_log,
    dtype=torch.float32
).reshape(-1, 1)

X_train_tensor = torch.tensor(
    X_train_processed,
    dtype=torch.float32
)

X_test_tensor = torch.tensor(
    X_test_processed,
    dtype=torch.float32
)


# --------------------------------------------------
# 7. CREATE PYTORCH DATA LOADER
# --------------------------------------------------

training_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

train_loader = DataLoader(
    training_dataset,
    batch_size=256,
    shuffle=True
)


# --------------------------------------------------
# 8. BUILD ANN MODEL
# --------------------------------------------------

class ClaimANN(nn.Module):

    def __init__(self, input_features):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_features, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):

        return self.network(x)


input_features = X_train_processed.shape[1]

model = ClaimANN(input_features).to(DEVICE)

print("\nANN input features:", input_features)
print("\nANN Architecture:")
print(model)


# --------------------------------------------------
# 9. LOSS FUNCTION AND OPTIMIZER
# --------------------------------------------------

loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# --------------------------------------------------
# 10. TRAIN THE ANN
# --------------------------------------------------

epochs = 100

print("\n========== ANN TRAINING ==========")

for epoch in range(epochs):

    model.train()

    total_loss = 0.0

    for batch_X, batch_y in train_loader:

        batch_X = batch_X.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        # Forward propagation
        predictions = model(batch_X)

        loss = loss_function(predictions, batch_y)

        # Clear previous gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += loss.item() * len(batch_X)

    average_loss = total_loss / len(training_dataset)

    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {average_loss:.5f}"
        )


# --------------------------------------------------
# 11. MODEL EVALUATION
# --------------------------------------------------

model.eval()

with torch.no_grad():

    predicted_log = model(
        X_test_tensor.to(DEVICE)
    ).cpu().numpy().flatten()

# Convert predictions back to original claim amount scale
y_pred = np.expm1(predicted_log)

# Claim amounts cannot be negative
y_pred = np.maximum(y_pred, 0)

ann_mae = mean_absolute_error(y_test, y_pred)

ann_rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

ann_r2 = r2_score(y_test, y_pred)


# --------------------------------------------------
# 12. BASELINE COMPARISON
# --------------------------------------------------

baseline_prediction = np.full(
    len(y_test),
    np.median(y_train)
)

baseline_mae = mean_absolute_error(
    y_test,
    baseline_prediction
)

improvement = (
    (baseline_mae - ann_mae) / baseline_mae
) * 100


print("\n========== MODEL RESULTS ==========")

print(f"Baseline MAE: {baseline_mae:.2f}")

print(f"ANN MAE: {ann_mae:.2f}")

print(f"ANN RMSE: {ann_rmse:.2f}")

print(f"ANN R2 Score: {ann_r2:.4f}")

print(f"Improvement over baseline: {improvement:.2f}%")


# --------------------------------------------------
# 13. SAVE MODEL AND PREPROCESSOR
# --------------------------------------------------

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_features": input_features,
        "architecture": [64, 32],
        "target_transform": "log1p"
    },
    MODEL_FILE
)

joblib.dump(
    {
        "preprocessor": preprocessor,
        "feature_columns": feature_columns
    },
    PREPROCESSOR_FILE
)


# --------------------------------------------------
# 14. SAVE EVALUATION REPORT
# --------------------------------------------------

results = pd.DataFrame([
    {
        "Model": "Median Baseline",
        "MAE": baseline_mae,
        "RMSE": np.nan,
        "R2": np.nan
    },
    {
        "Model": "PyTorch ANN",
        "MAE": ann_mae,
        "RMSE": ann_rmse,
        "R2": ann_r2
    }
])

results.to_csv(REPORT_FILE, index=False)


print("\n========== FILES SAVED ==========")
print("ANN model:", MODEL_FILE)
print("Preprocessor:", PREPROCESSOR_FILE)
print("Metrics report:", REPORT_FILE)

print("\nANN training completed successfully!")
