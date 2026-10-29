# Practical 15: Detect Data Drift and Trigger Model Retraining
**Project:** Online Shopping Purchase Prediction  
**Dataset:** `online_shopping.csv`  
**Repository:** [https://github.com/Tejasaisathwik/online-shopping-purchase-prediction](https://github.com/Tejasaisathwik/online-shopping-purchase-prediction)  
**Docker Hub:** [https://hub.docker.com/r/teja211/online-shopping-prediction](https://hub.docker.com/r/teja211/online-shopping-prediction)  

---

## 1. Aim & Learning Objectives
Extend the production MLOps pipeline for the Online Shopping Purchase Prediction system by:
1. Comparing recent production input data against a fixed baseline training-data reference.
2. Detecting significant data distribution shift using a non-parametric two-sample Kolmogorov-Smirnov (KS) test (`scipy.stats.ks_2samp`).
3. Retraining a candidate model conditionally only when statistically significant drift is confirmed AND recent ground truth labels are available.
4. Enforcing an automated model quality gate before candidate promotion to safeguard the production environment.
5. Packaging the promoted model into Docker images and publishing to Docker Hub and GHCR.

---

## 2. Monitored Features & Schema

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `Age` | Numerical | Customer age in years |
| `MonthlyIncome` | Numerical | Customer monthly income |
| `WebsiteVisits` | Numerical | Total website visits in the session |
| `PagesViewed` | Numerical | Number of pages viewed |
| `TimeSpentMinutes` | Numerical | Total duration spent on site (minutes) |
| `PreviousPurchases` | Numerical | Historical completed purchases |
| `CartItems` | Numerical | Number of items added to cart |
| `CustomerRating` | Numerical | Rating given by customer (1.0 - 5.0) |
| `PurchaseAmount` | Numerical | Dollar value of purchase |
| `Gender`, `DiscountUsed`, `DeviceType`, `Membership` | Categorical | Demographic & shopping attributes |
| **`Purchased`** | Target Label | 1 (Purchased) or 0 (Not Purchased) |

---

## 3. Step-by-Step Execution Workflow

### Step 1: Create the Reference Dataset
Extracts the baseline distribution tied to the currently deployed model:
```bash
python drift/prepare_reference.py
```
- Output: `drift/reference_data.csv` (951 samples)

### Step 2 & 4: Test No-Drift Case (Stable Production Data)
```bash
python drift/simulate_current_data.py --mode stable
python drift/detect_drift.py
```
- Result: `drifted_feature_count: 0`, `overall_drift: false`
- Quality Check:
```bash
python drift/retrain_candidate.py
```
- Output: `No overall drift detected. Retraining skipped to preserve production model.`

### Step 3 & 5: Test Drifted Case
```bash
python drift/simulate_current_data.py --mode drift
python drift/detect_drift.py
```
- Result: `drifted_feature_count: 6`, `overall_drift: true`
- Flagged Drifted Features: `Age`, `MonthlyIncome`, `WebsiteVisits`, `PagesViewed`, `TimeSpentMinutes`, `CustomerRating` ($p < 0.05$)
- Output Report: `drift/drift_report.json`

### Step 6 & 7: Conditional Retraining of Candidate Model
```bash
python drift/retrain_candidate.py
```
- Integrates `current_labeled_data.csv` with historical data.
- Trains `RandomForestClassifier` with full ColumnTransformer preprocessing.
- Output: `drift/online_shopping_model_candidate.joblib`
- Retraining Accuracy: **0.7609** (76.09%)

### Step 8: Model Quality Gate Enforcement
**Test 1: Rejection Case (Demonstration with higher threshold)**
```bash
python drift/approve_candidate.py --min-accuracy 0.85
```
- Result: `QUALITY GATE: FAIL` (Accuracy 0.7609 < 0.8500). Production model remains unchanged.

**Test 2: Promotion Case**
```bash
python drift/approve_candidate.py --min-accuracy 0.75
```
- Result: `QUALITY GATE: PASS` (Accuracy 0.7609 >= 0.7500).
- Candidate promoted to `online_shopping_model.joblib`.

### Step 9 & 10: Model & API Verification
```bash
pytest -v
```
- Result: **10/10 tests passed (100%)**

---

## 4. Observation Table

| Experiment Stage | Expected Observation | Actual Result |
| :--- | :--- | :--- |
| **Stable current batch** | Features statistically similar; `overall_drift: false` | Passed: 0 features drifted, `overall_drift: false` |
| **Drifted current batch** | Multiple features flagged ($p < 0.05$); `overall_drift: true` | Passed: 6 features flagged, `overall_drift: true` |
| **No-drift retraining check** | Retraining skipped to prevent unnecessary training | Passed: Retraining skipped automatically |
| **Drift with labeled data** | Retraining triggered; candidate model produced | Passed: Candidate model created (Acc: 0.7609) |
| **Candidate below threshold** | Quality gate fails; production model untouched | Passed: Exits with code 1, production file unchanged |
| **Candidate meets threshold** | Candidate promoted to production model file | Passed: Promoted to `online_shopping_model.joblib` |
| **Post-promotion test suite** | All API & unit tests pass on new model | Passed: 10/10 tests passed in 6.25s |

---

## 5. Artifacts Created & Preserved

- `drift/prepare_reference.py`
- `drift/simulate_current_data.py`
- `drift/detect_drift.py`
- `drift/retrain_candidate.py`
- `drift/approve_candidate.py`
- `drift/reference_data.csv`
- `drift/current_data.csv`
- `drift/current_labeled_data.csv`
- `drift/drift_report.json`
- `drift/retraining_metrics.json`
- `drift/online_shopping_model_candidate.joblib`
- `online_shopping_model.joblib`
