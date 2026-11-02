"""Generate API requests under normal or drifted feature distributions for Online Shopping."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent

NUMERICAL_FEATURES = [
    "Age", "MonthlyIncome", "WebsiteVisits", "PagesViewed", "TimeSpentMinutes",
    "PreviousPurchases", "CartItems", "CustomerRating", "PurchaseAmount"
]


def load_reference_sample():
    ref_path = ROOT / "data" / "reference_data.csv"
    if ref_path.exists():
        df = pd.read_csv(ref_path)
    else:
        csv_path = ROOT / "data" / "online_shopping.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError("Reference data not found. Please run train_model.py first.")

    # Fill any missing values in baseline reference with medians so JSON payloads are clean
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
    df["CustomerRating"] = df["CustomerRating"].fillna(df["CustomerRating"].median())
    return df


def generate_payload(mode: str, ref_df: pd.DataFrame, rng: np.random.Generator) -> dict:
    idx = int(rng.integers(0, len(ref_df)))
    base_row = ref_df.iloc[idx].to_dict()

    if mode == "normal":
        # Draw from baseline distribution with realistic slight variance
        base_income = base_row["MonthlyIncome"]
        if pd.isna(base_income):
            base_income = 81500.0

        base_rating = base_row["CustomerRating"]
        if pd.isna(base_rating):
            base_rating = 3.0

        age = int(np.clip(base_row["Age"] + rng.normal(0, 1), 18, 90))
        monthly_income = round(float(max(10000.0, float(base_income) + rng.normal(0, 1500))), 2)
        website_visits = int(np.clip(base_row["WebsiteVisits"] + rng.integers(-1, 2), 1, 50))
        pages_viewed = int(np.clip(base_row["PagesViewed"] + rng.integers(-1, 2), 1, 40))
        time_spent = int(np.clip(base_row["TimeSpentMinutes"] + rng.integers(-2, 3), 1, 120))
        previous_purchases = int(np.clip(base_row["PreviousPurchases"] + rng.integers(-1, 2), 0, 30))
        cart_items = int(np.clip(base_row["CartItems"] + rng.integers(-1, 2), 0, 20))
        rating = round(float(np.clip(float(base_rating) + rng.normal(0, 0.1), 1.0, 5.0)), 1)
        purchase_amt = round(float(max(50.0, float(base_row["PurchaseAmount"]) + rng.normal(0, 100))), 2)

        gender = str(base_row["Gender"]) if not pd.isna(base_row["Gender"]) else "Female"
        discount_used = str(base_row["DiscountUsed"]) if not pd.isna(base_row["DiscountUsed"]) else "No"
        device_type = str(base_row["DeviceType"]) if not pd.isna(base_row["DeviceType"]) else "Mobile"
        membership = str(base_row["Membership"]) if not pd.isna(base_row["Membership"]) else "Basic"
    else:
        # Deliberate data drift: significantly older demographic, lower income, low engagement
        age = int(np.clip(rng.normal(55, 6), 40, 95))
        monthly_income = round(float(np.clip(rng.normal(32000, 7000), 10000, 60000)), 2)
        website_visits = int(np.clip(rng.normal(4, 2), 1, 10))
        pages_viewed = int(np.clip(rng.normal(3, 1.5), 1, 8))
        time_spent = int(np.clip(rng.normal(7, 3), 1, 15))
        previous_purchases = int(np.clip(rng.normal(1, 1), 0, 5))
        cart_items = int(np.clip(rng.normal(1, 1), 0, 3))
        rating = round(float(np.clip(rng.normal(1.9, 0.4), 1.0, 3.0)), 1)
        purchase_amt = round(float(np.clip(rng.normal(650, 250), 50, 1800)), 2)

        gender = rng.choice(["Male", "Female"])
        discount_used = "No" if rng.random() > 0.15 else "Yes"
        device_type = "Desktop" if rng.random() > 0.4 else "Tablet"
        membership = "Basic" if rng.random() > 0.1 else "Premium"

    payload = {
        "Age": age,
        "Gender": gender,
        "MonthlyIncome": monthly_income,
        "WebsiteVisits": website_visits,
        "PagesViewed": pages_viewed,
        "TimeSpentMinutes": time_spent,
        "PreviousPurchases": previous_purchases,
        "CartItems": cart_items,
        "DiscountUsed": discount_used,
        "DeviceType": device_type,
        "Membership": membership,
        "CustomerRating": rating,
        "PurchaseAmount": purchase_amt,
    }
    return payload


def main():
    parser = argparse.ArgumentParser(description="Simulate prediction requests for monitoring.")
    parser.add_argument("--mode", choices=["normal", "drift"], required=True, help="Data generation mode")
    parser.add_argument("--count", type=int, default=250, help="Number of requests to send")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="API base URL")
    args = parser.parse_args()

    ref_df = load_reference_sample()
    seed = 123 if args.mode == "normal" else 456
    rng = np.random.default_rng(seed)

    tally = Counter()
    endpoint = f"{args.url.rstrip('/')}/predict"
    print(f"Sending {args.count} {args.mode.upper()} requests to {endpoint}...")

    session = requests.Session()
    for i in range(args.count):
        payload = generate_payload(args.mode, ref_df, rng)
        try:
            resp = session.post(endpoint, json=payload, timeout=10)
            resp.raise_for_status()
            result = resp.json()
            label = result.get("prediction", "Unknown")
            tally[label] += 1
        except requests.RequestException as exc:
            print(f"Request {i + 1} failed: {exc}")
            raise SystemExit(1) from exc

    print(f"\n[SUCCESS] Sent {args.count} {args.mode} requests.")
    print(f"Predicted labels distribution: {dict(tally)}")
    print(f"Check logs/predictions.jsonl and {args.url.rstrip('/')}/metrics")


if __name__ == "__main__":
    main()
