from pathlib import Path
import argparse
import json
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRIFT_DIR = PROJECT_ROOT / "drift"

CANDIDATE_JOBLIB = DRIFT_DIR / "online_shopping_model_candidate.joblib"
CANDIDATE_PKL = DRIFT_DIR / "student_result_model_candidate.pkl"

METRICS = DRIFT_DIR / "retraining_metrics.json"

APPROVED_JOBLIB = PROJECT_ROOT / "online_shopping_model.joblib"
APPROVED_PKL = PROJECT_ROOT / "student_result_model.pkl"

parser = argparse.ArgumentParser(description="Approve and promote retrained candidate model.")
parser.add_argument(
    "--min-accuracy",
    type=float,
    default=0.75,
    help="Minimum required accuracy threshold for quality gate (default: 0.75)"
)
args = parser.parse_args()

MIN_ACCURACY = args.min_accuracy

if not METRICS.exists():
    raise FileNotFoundError(f"Retraining metrics not found at {METRICS}. Run retrain_candidate.py first.")

with open(METRICS, encoding="utf-8") as f:
    metrics = json.load(f)

accuracy = float(metrics.get("candidate_accuracy", 0.0))

print("==================================================")
print(f"Candidate Evaluation Accuracy : {accuracy:.4f}")
print(f"Quality Gate Minimum Required : {MIN_ACCURACY:.4f}")
print("==================================================")

if accuracy < MIN_ACCURACY:
    print("QUALITY GATE: FAIL")
    print("Candidate model does not meet the specified quality threshold.")
    print("Current approved production model remains unchanged.")
    raise SystemExit(1)

# Promote candidate model to production
if CANDIDATE_JOBLIB.exists():
    shutil.copy2(CANDIDATE_JOBLIB, APPROVED_JOBLIB)
if CANDIDATE_PKL.exists():
    shutil.copy2(CANDIDATE_PKL, APPROVED_PKL)

print("QUALITY GATE: PASS")
print(f"Candidate successfully promoted to production model: {APPROVED_JOBLIB}")
