"""Compare logged production inputs to training reference; create PSI and Evidently AI drift reports.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

FEATURES = [
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


def psi(reference: pd.Series, current: pd.Series) -> float:
    """Calculate Population Stability Index using five quantile bins of reference data."""
    ref_vals = reference.dropna().to_numpy(float)
    curr_vals = current.dropna().to_numpy(float)
    edges = np.unique(np.quantile(ref_vals, [0.2, 0.4, 0.6, 0.8]))
    bins = np.r_[-np.inf, edges, np.inf]
    r = np.histogram(ref_vals, bins=bins)[0].astype(float)
    c = np.histogram(curr_vals, bins=bins)[0].astype(float)
    r = np.clip(r / max(r.sum(), 1), 1e-6, None)
    c = np.clip(c / max(c.sum(), 1), 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))


def main():
    parser = argparse.ArgumentParser(description="Generate PSI and Evidently Data Drift reports.")
    parser.add_argument(
        "--last", type=int, default=250,
        help="How many MOST RECENT prediction logs to evaluate"
    )
    parser.add_argument(
        "--output-prefix", default="latest",
        help="Prefix for saved filenames, e.g. normal or drift"
    )
    parser.add_argument(
        "--skip-evidently", action="store_true",
        help="Only output simple PSI metrics (skip HTML generation)"
    )
    args = parser.parse_args()

    log_path = ROOT / "logs" / "predictions.jsonl"
    if not log_path.exists():
        raise SystemExit("No predictions found. Run simulate_requests.py first.")

    with log_path.open(encoding="utf-8") as source:
        records = [json.loads(line) for line in source if line.strip()]

    if len(records) < args.last:
        raise SystemExit(f"Need at least {args.last} log rows; only {len(records)} available.")

    current_records = records[-args.last:]
    current = pd.DataFrame(current_records)[FEATURES].apply(pd.to_numeric, errors="coerce")

    ref_path = ROOT / "data" / "reference_data.csv"
    if not ref_path.exists():
        raise SystemExit("Reference data not found. Please run train_model.py first.")

    reference = pd.read_csv(ref_path)[FEATURES].apply(pd.to_numeric, errors="coerce")

    folder = ROOT / "reports"
    folder.mkdir(parents=True, exist_ok=True)

    rows = []
    for feat in FEATURES:
        val = psi(reference[feat], current[feat])
        rows.append({
            "feature": feat,
            "reference_mean": round(float(reference[feat].mean()), 2),
            "current_mean": round(float(current[feat].mean()), 2),
            "psi": round(val, 4),
            "flag_for_review": bool(val > 0.20),
        })

    report = pd.DataFrame(rows)
    prefix = "".join(ch for ch in args.output_prefix if ch.isalnum() or ch in "_-")
    if not prefix:
        raise SystemExit("Use a non-empty output prefix.")

    csv_path = folder / f"{prefix}_psi_summary.csv"
    report.to_csv(csv_path, index=False)

    print(f"\n=======================================================")
    print(f"   DATA DRIFT ANALYSIS: BATCH '{prefix.upper()}' (Last {args.last} requests)")
    print(f"=======================================================")
    print(report.to_string(index=False))
    print(f"=======================================================")
    print(f"Saved PSI summary to: {csv_path}")
    print(f"Heuristic review threshold: PSI > 0.20 implies noticeable distribution shift.")

    if not args.skip_evidently:
        try:
            from evidently import DataDefinition, Dataset, Report
            from evidently.presets import DataDriftPreset
        except ImportError as exc:
            raise SystemExit("Evidently missing. Install via pip install evidently") from exc

        schema = DataDefinition(numerical_columns=FEATURES)
        current_set = Dataset.from_pandas(current, data_definition=schema)
        reference_set = Dataset.from_pandas(reference, data_definition=schema)

        print("\nGenerating Evidently AI Data Drift report (HTML)...")
        snapshot = Report([DataDriftPreset(method="psi")], include_tests=True).run(
            current_set, reference_set
        )
        html_path = folder / f"{prefix}_evidently_drift_report.html"
        snapshot.save_html(str(html_path))
        print(f"Saved Evidently HTML report to: {html_path}")
        print("You can double-click this file to inspect interactive distribution plots.")

    print("\nNote: Input drift indicates data shift, not necessarily a drop in true accuracy.")


if __name__ == "__main__":
    main()
