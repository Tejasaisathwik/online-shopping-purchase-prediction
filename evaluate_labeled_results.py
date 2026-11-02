"""Calculate true accuracy and classification metrics when actual outcome labels become available.
"""
import argparse
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Evaluate model predictions against actual outcome labels.")
    parser.add_argument(
        "--labels", required=True,
        help="Path to CSV containing request_id and actual_result ('Purchased' or 'Not Purchased')"
    )
    args = parser.parse_args()

    label_file = Path(args.labels)
    if not label_file.exists():
        raise SystemExit(f"Label file not found: {label_file}")

    labels = pd.read_csv(label_file, dtype=str)
    required = {"request_id", "actual_result"}
    if not required.issubset(labels.columns):
        raise SystemExit(f"Label CSV must contain columns: {required}")

    log_path = ROOT / "logs" / "predictions.jsonl"
    if not log_path.exists():
        raise SystemExit("Prediction log file not found at logs/predictions.jsonl")

    with log_path.open(encoding="utf-8") as f:
        log_rows = [json.loads(line) for line in f if line.strip()]

    log_df = pd.DataFrame(log_rows)
    if "predicted_label" not in log_df.columns:
        raise SystemExit("predicted_label missing from log rows.")

    joined = log_df[["request_id", "predicted_label"]].merge(
        labels[list(required)].drop_duplicates("request_id"),
        on="request_id",
        how="inner",
    )

    if joined.empty:
        raise SystemExit("No matching request_id values found between log and labels.")

    print(f"\n=======================================================")
    print(f"   GROUND-TRUTH MODEL PERFORMANCE EVALUATION")
    print(f"=======================================================")
    print(f"Matched labeled predictions: {len(joined)}")
    acc = accuracy_score(joined["actual_result"], joined["predicted_label"])
    print(f"Ground-Truth Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(joined["actual_result"], joined["predicted_label"], zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(joined["actual_result"], joined["predicted_label"]))
    print("=======================================================\n")


if __name__ == "__main__":
    main()
