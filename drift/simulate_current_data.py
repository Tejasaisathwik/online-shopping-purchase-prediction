from pathlib import Path
import argparse
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRIFT_DIR = PROJECT_ROOT / "drift"
REFERENCE = DRIFT_DIR / "reference_data.csv"
SOURCE_LABELED = PROJECT_ROOT / "online_shopping.csv"
OUTPUT_UNLABELED = DRIFT_DIR / "current_data.csv"
OUTPUT_LABELED = DRIFT_DIR / "current_labeled_data.csv"

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

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=["stable", "drift"],
    default="stable",
    help="Simulation mode: stable (no drift) or drift (shifted distribution)"
)
args = parser.parse_args()

rng = np.random.default_rng(42)

# Load reference data
reference = pd.read_csv(REFERENCE)
n = min(150, len(reference))
current = reference.sample(
    n=n,
    replace=True,
    random_state=42
).reset_index(drop=True)

# Add small realistic variation for stable mode
for feature in NUMERICAL_FEATURES:
    if feature in current.columns:
        std_val = max(0.5, float(current[feature].std() * 0.05))
        current[feature] = current[feature] + rng.normal(0, std_val, size=len(current))

# Introduce significant distribution shift for drift mode
if args.mode == "drift":
    if "MonthlyIncome" in current.columns:
        current["MonthlyIncome"] -= 30000.0
    if "WebsiteVisits" in current.columns:
        current["WebsiteVisits"] -= 4
    if "PagesViewed" in current.columns:
        current["PagesViewed"] -= 6
    if "TimeSpentMinutes" in current.columns:
        current["TimeSpentMinutes"] -= 12.0
    if "PurchaseAmount" in current.columns:
        current["PurchaseAmount"] -= 400.0
    if "CustomerRating" in current.columns:
        current["CustomerRating"] -= 1.2
    if "Age" in current.columns:
        current["Age"] -= 8

# Enforce valid physical bounds
if "Age" in current.columns:
    current["Age"] = current["Age"].clip(18, 100).round().astype(int)
if "MonthlyIncome" in current.columns:
    current["MonthlyIncome"] = current["MonthlyIncome"].clip(10000, None).round(2)
if "WebsiteVisits" in current.columns:
    current["WebsiteVisits"] = current["WebsiteVisits"].clip(0, None).round().astype(int)
if "PagesViewed" in current.columns:
    current["PagesViewed"] = current["PagesViewed"].clip(0, None).round().astype(int)
if "TimeSpentMinutes" in current.columns:
    current["TimeSpentMinutes"] = current["TimeSpentMinutes"].clip(1, None).round(1)
if "PreviousPurchases" in current.columns:
    current["PreviousPurchases"] = current["PreviousPurchases"].clip(0, None).round().astype(int)
if "CartItems" in current.columns:
    current["CartItems"] = current["CartItems"].clip(0, None).round().astype(int)
if "CustomerRating" in current.columns:
    current["CustomerRating"] = current["CustomerRating"].clip(1.0, 5.0).round(1)
if "PurchaseAmount" in current.columns:
    current["PurchaseAmount"] = current["PurchaseAmount"].clip(10.0, None).round(2)

# Save unlabelled current production input batch
current.to_csv(OUTPUT_UNLABELED, index=False)
print("Mode:", args.mode)
print("Rows:", len(current))
print("File:", OUTPUT_UNLABELED)

# Prepare corresponding ground truth labeled batch for retraining when needed
labeled_df = current.copy()
# Generate realistic Purchase outcome based on purchase amount and rating
prob = (
    (labeled_df["PurchaseAmount"] > 300).astype(int) * 0.4 +
    (labeled_df["CustomerRating"] > 3.0).astype(int) * 0.3 +
    (labeled_df["WebsiteVisits"] > 3).astype(int) * 0.2 +
    rng.uniform(0, 0.1, size=len(labeled_df))
)
labeled_df["Purchased"] = (prob >= 0.5).astype(int)
labeled_df.to_csv(OUTPUT_LABELED, index=False)
print("Labeled File for Retraining:", OUTPUT_LABELED)
