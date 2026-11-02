"""Train reproducible machine learning model for Online Shopping Purchase Prediction.
Saves model artifact and reference feature dataset for drift monitoring.
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent

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
TARGET = "Purchased"


def load_dataset():
    paths = [
        ROOT / "data" / "online_shopping.csv",
        ROOT / "online_shopping.csv",
        Path("c:/Users/Dell/Downloads/online_shopping.csv"),
    ]
    for p in paths:
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError("online_shopping.csv not found in data/ or root directory.")


def build_pipeline():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUMERICAL_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    clf = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1500, random_state=42)),
    ])
    return clf


def main():
    (ROOT / "models").mkdir(parents=True, exist_ok=True)
    (ROOT / "data").mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    print(f"Loaded dataset shape: {df.shape}")

    X = df[ALL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    clf = build_pipeline()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = round(accuracy_score(y_test, y_pred), 4)
    print("Hold-out accuracy:", acc)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["Not Purchased", "Purchased"]))

    artifact = {
        "model": clf,
        "features": ALL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "version": "demo-v1",
        "accuracy": acc,
    }

    joblib.dump(artifact, ROOT / "models" / "online_shopping_model.joblib")
    joblib.dump(artifact, ROOT / "models" / "student_result.joblib")
    joblib.dump(clf, ROOT / "online_shopping_model.joblib")

    X_train.to_csv(ROOT / "data" / "reference_data.csv", index=False)
    print(f"Saved reference dataset to {ROOT / 'data' / 'reference_data.csv'} ({len(X_train)} samples)")
    print("Saved model artifacts to models/online_shopping_model.joblib and models/student_result.joblib")


if __name__ == "__main__":
    main()
