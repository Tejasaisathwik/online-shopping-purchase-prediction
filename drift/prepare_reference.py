from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRIFT_DIR = PROJECT_ROOT / "drift"
SOURCE = PROJECT_ROOT / "online_shopping.csv"
OUTPUT = DRIFT_DIR / "reference_data.csv"

# Numerical features evaluated for statistical data drift (KS-test)
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

# All features including categorical attributes
ALL_FEATURES = NUMERICAL_FEATURES + [
    "Gender",
    "DiscountUsed",
    "DeviceType",
    "Membership",
]

if not SOURCE.exists():
    fallback = PROJECT_ROOT / "student_results.csv"
    if fallback.exists():
        SOURCE = fallback
        NUMERICAL_FEATURES = ["attendance", "internal_marks", "assignment_marks", "previous_score"]
        ALL_FEATURES = NUMERICAL_FEATURES

df = pd.read_csv(SOURCE)
missing = [c for c in ALL_FEATURES if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in {SOURCE}: {missing}")

reference = df[ALL_FEATURES].dropna().copy()
reference.to_csv(OUTPUT, index=False)

print("Reference dataset created.")
print("Rows:", len(reference))
print("File:", OUTPUT)
