from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRIFT_DIR = PROJECT_ROOT / "drift"
HISTORICAL = PROJECT_ROOT / "online_shopping.csv"
RECENT = DRIFT_DIR / "current_labeled_data.csv"
DRIFT_REPORT = DRIFT_DIR / "drift_report.json"
CANDIDATE_MODEL_JOBLIB = DRIFT_DIR / "online_shopping_model_candidate.joblib"
CANDIDATE_MODEL_PKL = DRIFT_DIR / "student_result_model_candidate.pkl"
METRICS_FILE = DRIFT_DIR / "retraining_metrics.json"

TARGET = "Purchased"

NUMERICAL_FEATURES = [
    "Age",
    "MonthlyIncome",
    "WebsiteVisits",
    "PagesViewed",
    "TimeSpentMinutes",
    "PreviousPurchases",
    "CartItems",
    "CustomerRating",
    "PurchaseAmount",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "DiscountUsed",
    "DeviceType",
    "Membership",
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# 1. Gate check: Read drift report
if not DRIFT_REPORT.exists():
    raise FileNotFoundError(f"Drift report not found at {DRIFT_REPORT}. Run detect_drift.py first.")

with open(DRIFT_REPORT, encoding="utf-8") as f:
    drift_report = json.load(f)

if not drift_report.get("overall_drift", False):
    print("==================================================")
    print("No overall drift detected.")
    print("Retraining skipped to preserve production model.")
    print("==================================================")
    raise SystemExit(0)

# 2. Check for recent labeled ground truth data
if not RECENT.exists():
    raise FileNotFoundError(
        "Drift was detected, but current_labeled_data.csv is not available for supervised retraining."
    )

print("Drift detected! Initiating candidate model retraining pipeline...")

historical = pd.read_csv(HISTORICAL)
recent = pd.read_csv(RECENT)

required = ALL_FEATURES + [TARGET]
for name, frame in [("historical", historical), ("recent", recent)]:
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"{name} data missing required columns: {missing}")

# Combine historical data and recent ground truth batch
combined = pd.concat([historical[required], recent[required]], ignore_index=True).dropna(subset=[TARGET])
X = combined[ALL_FEATURES]
y = combined[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Build preprocessing and model pipeline
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer([
    ("num", num_pipe, NUMERICAL_FEATURES),
    ("cat", cat_pipe, CATEGORICAL_FEATURES),
])

candidate = Pipeline([
    ("preprocessor", preprocess),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
])

# Train candidate model
candidate.fit(X_train, y_train)

# Evaluate on held-out test split
predictions = candidate.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Save candidate model artifacts (kept separate from production until approval)
joblib.dump(candidate, CANDIDATE_MODEL_JOBLIB)
joblib.dump(candidate, CANDIDATE_MODEL_PKL)

metrics = {
    "candidate_accuracy": round(float(accuracy), 6),
    "training_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "historical_rows": int(len(historical)),
    "recent_labeled_rows": int(len(recent)),
}

with open(METRICS_FILE, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("\nRetraining Summary:")
print(json.dumps(metrics, indent=2))
print(f"Candidate model saved to: {CANDIDATE_MODEL_JOBLIB}")
