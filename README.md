# Online Shopping Purchase Prediction - MLOps Pipeline

End-to-End MLOps pipeline for Customer Purchase Prediction based on `online_shopping.csv`.

## Repository Architecture & Practicals

- **Practical 11: Production Logging & Telemetry**
  - Unique Request Correlation (`X-Request-ID` header)
  - Structured JSON access and error logging
- **Practical 12: Real-time Metric Monitoring & Alerting**
  - Prometheus client metrics (`/metrics`)
  - Prometheus server scraping configuration (`Practical_12_Monitoring/prometheus.yml`)
  - Prometheus alerting rules (`Practical_12_Monitoring/online_shopping_alerts.yml`)
  - Grafana visual analytics dashboard (`Practical_12_Monitoring/grafana_dashboard.json`)
- **Practical 13: Kubernetes Container Orchestration**
  - Scalable Deployment & NodePort Service (`Practical_13_Kubernetes/`)
  - Rolling updates, self-healing pod verification, and rollback capability
- **Practical 14: Automated CI/CD & Production GitOps**
  - GitHub Actions automated pipeline (`.github/workflows/kubernetes-cicd.yml`)
  - Multi-stage image build, automated testing, container registry push (GHCR & Docker Hub)
  - Zero-downtime rolling update into Minikube cluster
- **Practical 15: Data Drift Detection & Controlled Model Retraining**
  - Non-parametric Kolmogorov-Smirnov (KS) test (`scipy.stats.ks_2samp`) on 9 numerical features
  - Stable batch vs. Drifted batch statistical evaluation
  - Automated conditional retraining gate
  - Quality gate promotion safeguard (`approve_candidate.py`)

## Registries & Links

- **GitHub Repository:** [https://github.com/Tejasaisathwik/online-shopping-purchase-prediction](https://github.com/Tejasaisathwik/online-shopping-purchase-prediction)
- **Docker Hub Images:** [https://hub.docker.com/r/teja211/online-shopping-prediction](https://hub.docker.com/r/teja211/online-shopping-prediction)
  - Tag `:practical-15`
  - Tag `:latest`
