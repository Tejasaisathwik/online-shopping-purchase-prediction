# MLOps Monitoring Lab: Student Observation Sheet & Assessment
**Project:** Online Shopping Purchase Prediction Monitoring  
**Dataset:** `online_shopping.csv` (1,000 real customer records)  
**Technologies:** Python • scikit-learn • FastAPI • Evidently AI • Docker • Prometheus • Grafana  
**GitHub Repository:** [https://github.com/Tejasaisathwik/online-shopping-purchase-prediction](https://github.com/Tejasaisathwik/online-shopping-purchase-prediction)  
**Docker Hub Image:** [https://hub.docker.com/r/teja211/online-shopping-prediction](https://hub.docker.com/r/teja211/online-shopping-prediction)  

---

## 1. Observation Table (Section 14.2)

| Observation Metric | Measured Lab Result | Explanation |
| :--- | :--- | :--- |
| **No. of normal simulated requests** | **250** | Sent via `simulate_requests.py --mode normal --count 250` sampled from baseline. |
| **No. of drift simulated requests** | **250** | Sent via `simulate_requests.py --mode drift --count 250` simulating demographic & behavior shift. |
| **Highest PSI feature after drift** | **`PurchaseAmount` (PSI: 11.0524)** | Mean dropped from \$4,958.58 (ref) to \$652.39 (current). Followed by `PreviousPurchases` (9.7515). |
| **Was the drift review flag raised?** | **YES (`flag_for_review: True`)** | All 9 numerical features breached the classroom threshold ($PSI > 0.20$). |
| **Highest 5-min prediction request count** | **280 requests** | Plotted in Grafana Panel 1 via `sum(increase(ml_http_requests_total{path="/predict"}[5m]))`. |
| **Observed P95 Latency** | **30.27 ms (0.0303 s)** | Plotted in Grafana Panel 2 via `histogram_quantile(0.95, ...)` bucket rate. |
| **Observed HTTP 5xx Server Error Rate** | **0.00%** | Zero server crashes detected; valid client traffic produced 200 OK responses. |
| **Prediction Distribution (Sample Window)** | **Normal: 120 Purchased / 130 Not Purchased<br>Drift: 0 Purchased / 250 Not Purchased** | Plotted in Grafana Panel 4 via `sum by (predicted_label) (increase(ml_predictions_total[5m]))`. |

---

## 2. Feature-Level Population Stability Index (PSI) Comparison

| Feature | Reference Mean | Normal Current Mean | Normal PSI | Normal Review Flag | Drift Current Mean | Drift PSI | Drift Review Flag |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Age` | 39.11 yrs | 37.60 yrs | 0.0228 | **False** | 55.10 yrs | 5.7777 | **True** (Shifted older) |
| `MonthlyIncome` | \$81,897.21 | \$82,339.46 | 0.0069 | **False** | \$32,497.31 | 8.4879 | **True** (Income drop) |
| `WebsiteVisits` | 15.25 visits | 15.49 visits | 0.0098 | **False** | 3.48 visits | 8.8485 | **True** (Low engagement) |
| `PagesViewed` | 10.50 pages | 11.22 pages | 0.0151 | **False** | 2.58 pages | 8.9537 | **True** (Fewer pages) |
| `TimeSpentMinutes`| 30.66 mins | 29.69 mins | 0.0111 | **False** | 6.48 mins | 9.2982 | **True** (Reduced time) |
| `PreviousPurchases`| 7.56 orders| 7.35 orders | 0.0209 | **False** | 0.67 orders | 9.7515 | **True** (First-time buyers)|
| `CartItems` | 4.99 items | 4.67 items | 0.0105 | **False** | 0.67 items | 8.8624 | **True** (Empty carts) |
| `CustomerRating` | 2.98 / 5.0 | 2.85 / 5.0 | 0.0304 | **False** | 1.88 / 5.0 | 5.6810 | **True** (Low satisfaction) |
| `PurchaseAmount` | \$4,958.58 | \$4,851.00 | 0.0103 | **False** | \$652.39 | 11.0524 | **True** (Low spending) |

---

## 3. Student Analysis & Viva Questions (Section 14.3)

### Q1: Which input feature changed most between normal and drift reports? Support your answer with PSI or mean values.
**Answer:**  
`PurchaseAmount` changed the most, with the highest Population Stability Index of **11.0524**. Its mean plummeted from **\$4,958.58** in the training reference baseline down to **\$652.39** during the drift simulation. The next most severely shifted features were `PreviousPurchases` ($PSI = 9.7515$, mean drop from 7.56 to 0.67) and `TimeSpentMinutes` ($PSI = 9.2982$, mean drop from 30.66 to 6.48 mins).

### Q2: Why can the drift report show a large change even when the deployed model has not been modified?
**Answer:**  
Data drift measures changes in the **incoming input feature distributions ($P(X)$)**, not changes in the model weights. The model artifact (`online_shopping_model.joblib`) remains completely fixed in production, but external customer behavior, advertising audience demographics, economic conditions, or website changes can alter the population sending requests to the API.

### Q3: Why does the five-minute request chart eventually return to zero?
**Answer:**  
The PromQL query `sum(increase(ml_http_requests_total{path="/predict"}[5m]))` evaluates a **rolling sliding 5-minute window**. It calculates the difference in cumulative requests between the current timestamp $t$ and $t - 5\text{ min}$. Once request traffic ceases and more than 5 minutes have elapsed since the burst, zero new requests occurred within that moving window, causing the plotted rate to naturally return to zero. It does **not** mean the API crashed or reset.

### Q4: What is the difference between response time, HTTP error rate, and model accuracy?
**Answer:**  
- **Response Time (Latency):** Measures operational execution speed—how many milliseconds the API takes to deserialize, predict, and respond (e.g., P95 = 30.27 ms).  
- **HTTP Error Rate:** Measures system reliability—the percentage of requests resulting in HTTP 5xx server failures (e.g., 0% server crashes).  
- **Model Accuracy:** Measures predictive quality—how often the predicted label matches the verified ground-truth outcome label. A 100% healthy, 0% error, ultra-fast server can still make incorrect predictions if the data has drifted.

### Q5: If Not Purchased predictions increase after a data shift, can we conclude the model became inaccurate? Explain.
**Answer:**  
**No.** If incoming users genuinely have much lower income (\$32k vs \$82k), visit fewer pages, and add almost nothing to their cart, predicting "Not Purchased" may actually be **100% accurate** for that population. A shift in output distribution ($P(\hat{Y})$) is an alert to investigate, but true model accuracy can only be determined by collecting verified actual purchase labels and computing ground-truth metrics.

### Q6: What would you investigate first if API P95 latency suddenly exceeded 500 ms?
**Answer:**  
1. **Container resource exhaustion:** Check CPU and RAM utilization using `docker stats` or container metrics to see if the Python worker is CPU-throttled or experiencing high garbage collection pauses.  
2. **Concurrency / Thread starvation:** Verify if incoming concurrency exceeds the Uvicorn single-worker queue capacity.  
3. **Downstream latency:** Check if disk I/O for `predictions.jsonl` logging is blocked or encountering storage lock contention.  
4. **Input payload size & serialization:** Inspect if large batch requests or malformed payloads are reaching `/predict`.

---

## 4. Evidence Files Generated

1. `reports/normal_psi_summary.csv` — Feature-level PSI baseline values ($PSI < 0.035$)
2. `reports/normal_evidently_drift_report.html` — Interactive Evidently AI report for normal baseline
3. `reports/drift_psi_summary.csv` — Feature-level PSI drift values ($PSI > 5.0$)
4. `reports/drift_evidently_drift_report.html` — Interactive Evidently AI report for shifted population
5. `logs/predictions.jsonl` — 500+ structured JSONL records capturing inference events
6. `monitoring/prometheus.yml` — Prometheus scrape configuration (5s interval)
7. `monitoring/alerts.yml` — Alert rules for high latency (> 500ms) and API downtime
8. `grafana/provisioning/` — Automated datasource and 4-panel dashboard provisioning
