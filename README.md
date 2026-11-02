# Online Shopping Purchase Prediction — MLOps Monitoring Lab
**Technologies:** Python • scikit-learn • FastAPI • Evidently AI • Docker • Prometheus • Grafana  
**Dataset:** `online_shopping.csv` (1,000 real customer records)  
**GitHub Repository:** [https://github.com/Tejasaisathwik/online-shopping-purchase-prediction](https://github.com/Tejasaisathwik/online-shopping-purchase-prediction)  
**Docker Hub Image:** [https://hub.docker.com/r/teja211/online-shopping-prediction](https://hub.docker.com/r/teja211/online-shopping-prediction)  

---

## 1. Practical Aim & System Architecture

This project extends an end-to-end Machine Learning model for **Online Shopping Purchase Prediction** into a production-grade monitored system. It records inference events, detects data distribution shifts with **Evidently AI** and **PSI**, exposes operational and prediction metrics to **Prometheus**, visualizes live behavior across a **four-panel Grafana dashboard**, and evaluates alert conditions.

### System Data Flow
```
online_shopping.csv ──> train_model.py ──> models/online_shopping_model.joblib
                                       └──> data/reference_data.csv (baseline)

Customer features ──> POST /predict ──> FastAPI (app.py) ──> Prediction (Purchased / Not Purchased)
                                      ├──> logs/predictions.jsonl (Event logging)
                                      └──> GET /metrics ──> Prometheus ──> Grafana Dashboard

Reference CSV + recent prediction logs ──> drift_report.py ──> PSI CSV + Evidently HTML Reports
```

---

## 2. Project File Structure

```text
online-shopping-purchase-prediction/
├── app.py                         # Monitored FastAPI app + middleware + Prometheus metrics
├── train_model.py                 # Pipeline training on online_shopping.csv + reference baseline
├── simulate_requests.py           # Synthetic traffic generator (normal vs. drifted distributions)
├── drift_report.py                # Evidently AI HTML reports + Population Stability Index (PSI)
├── evaluate_labeled_results.py     # Ground-truth accuracy and classification evaluation
├── Dockerfile                     # Multi-stage containerization of API + model
├── docker-compose.yml             # Orchestration of API, Prometheus, and Grafana containers
├── requirements.txt               # Complete Python dependencies (including Evidently & Pytest)
├── requirements-api.txt           # Container runtime dependencies
├── OBSERVATIONS.md                # Completed observation sheet & Viva answers
├── data/
│   ├── online_shopping.csv        # Primary customer shopping dataset
│   ├── reference_data.csv         # 800-sample baseline feature distribution
│   └── actual_results_template.csv# Ground-truth label evaluation template
├── models/
│   ├── online_shopping_model.joblib # Serialized scikit-learn pipeline
│   └── student_result.joblib      # Alias artifact for compatibility
├── monitoring/
│   ├── prometheus.yml             # 5s interval scraping configuration
│   └── alerts.yml                 # Latency (>500ms) and target availability alerts
├── grafana/provisioning/
│   ├── datasources/datasource.yml # Pre-configured Prometheus datasource
│   └── dashboards/
│       ├── dashboards.yml         # Dashboard provider configuration
│       └── online_shopping_monitoring.json # Pre-provisioned 4-panel dashboard
├── tests/
│   └── test_app.py                # Automated pytest suite (health, predict, 422, metrics)
└── reports/
    ├── normal_psi_summary.csv     # Normal batch PSI table
    ├── normal_evidently_drift_report.html # Normal batch Evidently visual report
    ├── drift_psi_summary.csv      # Drift batch PSI table
    └── drift_evidently_drift_report.html  # Drift batch Evidently visual report
```

---

## 3. Step-by-Step Execution Guide

### Step 1: Environment Setup & Automated Tests
```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Train model and generate reference dataset
python train_model.py

# Run test suite
python -m pytest tests/test_app.py -q
```
*Result: 4 passed tests (Health check, Prediction output, Pydantic HTTP 422 input validation, Metrics endpoint).*

---

### Step 2: Data Drift Detection with Evidently AI & PSI

1. **Simulate 250 Normal Requests:**
```powershell
python simulate_requests.py --mode normal --count 250
python drift_report.py --last 250 --output-prefix normal
```
*Output: `reports/normal_psi_summary.csv` and `reports/normal_evidently_drift_report.html` (All $PSI < 0.035$, `flag_for_review: False`).*

2. **Simulate 250 Drifted Requests:**
```powershell
python simulate_requests.py --mode drift --count 250
python drift_report.py --last 250 --output-prefix drift
```
*Output: `reports/drift_psi_summary.csv` and `reports/drift_evidently_drift_report.html` (All $PSI > 5.0$, `flag_for_review: True`).*

---

### Step 3: Run Full Monitoring Stack with Docker Compose

```powershell
# Launch API, Prometheus, and Grafana containers
docker compose up --build -d

# Verify containers are running
docker compose ps
```

| Service | Port | Local URL |
| :--- | :--- | :--- |
| **FastAPI Service** | 8000 | [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) |
| **Prometheus** | 9090 | [http://127.0.0.1:9090/targets](http://127.0.0.1:9090/targets) |
| **Grafana Dashboard** | 3000 | [http://127.0.0.1:3000](http://127.0.0.1:3000) *(admin / admin)* |

Send live traffic to the Docker container:
```powershell
python simulate_requests.py --mode normal --count 250
```

---

## 4. Grafana Dashboard Panels & PromQL Queries

The dashboard is auto-provisioned at `http://127.0.0.1:3000/d/online-shopping-monitoring-dashboard`:

| Panel Title | Question Answered | PromQL Expression |
| :--- | :--- | :--- |
| **1. Prediction Requests (Last 5 Min)** | Recent API traffic volume | `sum(increase(ml_http_requests_total{path="/predict"}[5m]))` |
| **2. Prediction Latency (P95)** | 95th-percentile response time | `histogram_quantile(0.95, sum by (le) (rate(ml_http_request_duration_seconds_bucket{path="/predict"}[5m])))` |
| **3. Prediction API Error Rate** | HTTP 5xx server failure rate | `100 * (sum(rate(ml_http_requests_total{path="/predict",status=~"5.."}[5m])) or vector(0)) / clamp_min(sum(rate(ml_http_requests_total{path="/predict"}[5m])) or vector(0), 1e-9)` |
| **4. Prediction Distribution** | Purchased vs Not Purchased mix | `sum by (predicted_label) (increase(ml_predictions_total[5m]))` |

---

## 5. Docker Hub Publishing

Docker Hub images were built and pushed directly:
- **`teja211/online-shopping-prediction:latest`**
- **`teja211/online-shopping-prediction:practical-12`**

Pull and run anywhere with:
```bash
docker pull teja211/online-shopping-prediction:latest
docker run -p 8000:8000 teja211/online-shopping-prediction:latest
```
