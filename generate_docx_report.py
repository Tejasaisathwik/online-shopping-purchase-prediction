import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
import os

def create_report():
    doc = docx.Document()

    # Set standard margins
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Colors
    COLOR_PRIMARY = RGBColor(31, 78, 121)     # Deep blue
    COLOR_SECONDARY = RGBColor(89, 89, 89)   # Slate gray
    COLOR_TEXT = RGBColor(38, 38, 38)        # Charcoal

    def style_table(table, col_widths=None):
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, row in enumerate(table.rows):
            trPr = row._tr.get_or_add_trPr()
            trPr.append(parse_xml(r'<w:cantSplit {}/>'.format(nsdecls('w'))))
            
            for j, cell in enumerate(row.cells):
                tcPr = cell._tc.get_or_add_tcPr()
                if i == 0:
                    shading = parse_xml(r'<w:shd {} w:fill="1F4E79"/>'.format(nsdecls('w')))
                    tcPr.append(shading)
                    for p in cell.paragraphs:
                        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                        for run in p.runs:
                            run.font.bold = True
                            run.font.color.rgb = RGBColor(255, 255, 255)
                            run.font.size = Pt(10)
                else:
                    if i % 2 == 1:
                        shading = parse_xml(r'<w:shd {} w:fill="F2F4F7"/>'.format(nsdecls('w')))
                        tcPr.append(shading)
                    for p in cell.paragraphs:
                        for run in p.runs:
                            run.font.color.rgb = COLOR_TEXT
                            run.font.size = Pt(9.5)
                if col_widths and j < len(col_widths):
                    cell.width = col_widths[j]

    # Header / Title
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_title = title_p.add_run("MLOPS MONITORING\nONLINE SHOPPING PURCHASE PREDICTION\nPRACTICAL REPORT\n")
    r_title.font.bold = True
    r_title.font.size = Pt(18)
    r_title.font.color.rgb = COLOR_PRIMARY

    # Metadata Table
    meta_table = doc.add_table(rows=4, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        ("Student Name", "Teja Sai Sathwik"),
        ("Project", "Online Shopping Purchase Prediction ML Monitoring"),
        ("GitHub Repository", "https://github.com/Tejasaisathwik/online-shopping-purchase-prediction"),
        ("Docker Hub Image", "teja211/online-shopping-prediction:latest")
    ]
    for idx, (k, v) in enumerate(meta_data):
        meta_table.rows[idx].cells[0].text = k
        meta_table.rows[idx].cells[1].text = v
    style_table(meta_table, [Inches(2.2), Inches(4.3)])

    doc.add_paragraph()

    # 1. Aim
    h1 = doc.add_heading("1. Aim", level=1)
    h1.style.font.color.rgb = COLOR_PRIMARY
    p_aim = doc.add_paragraph(
        "To establish a comprehensive production-grade MLOps monitoring pipeline for the Online Shopping "
        "Purchase Prediction application, track API health and operational performance metrics using Prometheus, "
        "visualize throughput, latency, server errors, and prediction distributions using Grafana dashboards, and "
        "detect feature distribution shifts and data drift between baseline reference data and live production "
        "inputs using Evidently AI and Population Stability Index (PSI)."
    )
    p_aim.style.font.size = Pt(10.5)

    # 2. Tools and Technologies
    h2 = doc.add_heading("2. Tools and Technologies", level=1)
    h2.style.font.color.rgb = COLOR_PRIMARY
    tools = [
        ("Python and FastAPI", "High-performance RESTful API serving the scikit-learn Online Shopping purchase classification model with Pydantic validation and Prometheus instrumentation."),
        ("Docker & Docker Compose", "Containerized multi-service orchestration packaging the API, Prometheus time-series database, and Grafana dashboard into an isolated, reproducible network."),
        ("Prometheus", "Pull-based time-series metrics monitoring system scraping operational and ML classification telemetry every 5 seconds."),
        ("Grafana", "Real-time observability dashboard visualizing throughput, P95 inference latency, HTTP error rates, and classification class balances."),
        ("Evidently AI", "Statistical data drift and model profiling engine comparing reference baseline distributions against live streaming requests using Population Stability Index (PSI)."),
        ("Pandas & Scikit-Learn", "Data preprocessing pipeline featuring SimpleImputer, StandardScaler, and OneHotEncoder paired with a serialized Logistic Regression classification model.")
    ]
    for t, desc in tools:
        p = doc.add_paragraph(style='List Bullet')
        r_bold = p.add_run(f"{t} \u2014 ")
        r_bold.font.bold = True
        p.add_run(desc)

    # 3. Project Overview
    h3 = doc.add_heading("3. Project Overview", level=1)
    h3.style.font.color.rgb = COLOR_PRIMARY
    doc.add_paragraph(
        "A dedicated MLOps monitoring infrastructure was engineered for the Online Shopping Purchase Prediction system. "
        "The production dataset comprises 1,000 historical consumer sessions with 13 key behavioral and demographic "
        "features predicting whether a website visitor will complete a purchase (Purchased: 0 or 1). Missing data in "
        "numerical features (MonthlyIncome and CustomerRating) is handled via median imputation inside a Scikit-Learn "
        "ColumnTransformer pipeline."
    )
    doc.add_paragraph(
        "The 13 monitored features consist of 9 numerical dimensions (Age, MonthlyIncome, WebsiteVisits, PagesViewed, "
        "TimeSpentMinutes, PreviousPurchases, CartItems, CustomerRating, PurchaseAmount) and 4 categorical dimensions "
        "(Gender, Category, Device, PaymentMethod). The monitoring stack encapsulates the FastAPI inference engine, "
        "Prometheus scraping server on port 9090, Grafana visualizer on port 3000, streaming JSON Lines prediction logs "
        "at logs/predictions.jsonl, and automated Evidently AI drift reporting."
    )

    # 4. Implementation and Results
    h4 = doc.add_heading("4. Implementation and Results", level=1)
    h4.style.font.color.rgb = COLOR_PRIMARY

    subsections = [
        ("Project Structure", "The monitoring repository is structured with modular separation: application code (app.py), model training (train_model.py), simulation client (simulate_requests.py), Evidently drift calculator (drift_report.py), Docker specifications (Dockerfile, docker-compose.yml), Prometheus scraping configurations (monitoring/prometheus.yml), Grafana dashboards (grafana/dashboards), streaming logs (logs/predictions.jsonl), and HTML reports (reports/)."),
        ("Docker Services", "Docker Compose orchestrates three independent services connected via an internal bridge network: online-shopping-api (port 8000), prometheus (port 9090), and grafana (port 3000). All containers maintain a persistent UP status verified via docker compose ps."),
        ("API Health Check", "The /health endpoint evaluates system vitality and returns {\"status\": \"healthy\", \"model_loaded\": true}, confirming the serialized Joblib pipeline is loaded into memory and ready for inference."),
        ("Prometheus Target Monitoring", "Prometheus queries http://online-shopping-api:8000/metrics every 5 seconds under the student_result_api scrape job. The target health status reports 1/1 target UP with 0 scrape dropouts."),
        ("Prometheus Metrics", "Telemetry is exposed via ml_http_requests_total, ml_http_request_duration_seconds, and ml_predictions_total. A cumulative volume of 1,250+ requests was logged with an average processing latency under 20 milliseconds."),
        ("Grafana Dashboard", "The custom Grafana dashboard 'Online Shopping Purchase Prediction \u2013 Model Monitoring' displays four synchronized panels: (1) Prediction Requests (Last 5 Minutes), (2) P95 Prediction Latency (ms), (3) Prediction API Error Rate (0.00%), and (4) Prediction Distribution (Purchased vs Not Purchased). Sliding 5-minute aggregation provides accurate traffic bursts while maintaining zero baseline during idle intervals."),
        ("Evidently AI Live Data Drift", "Baseline reference data (800 records) was compared against live prediction batches (250 requests). Under normal operational mode, 0 of 9 numerical features exhibited drift (PSI < 0.035). Under simulated demographic and behavioral drift (income reduction and session dropouts), 9 of 9 features breached the 0.5 threshold (100% drift share), triggering dataset-level alerts."),
        ("Prediction Input Logging", "All inference payloads and model decisions are persistently streamed to logs/predictions.jsonl with ISO 8601 timestamps, feature dictionaries, predicted classes, and probabilities.")
    ]
    for title, desc in subsections:
        h = doc.add_heading(title, level=2)
        h.style.font.color.rgb = COLOR_SECONDARY
        doc.add_paragraph(desc)

    # 5. Evidently AI Drift Result Table
    h5 = doc.add_heading("5. Evidently AI Drift Result", level=1)
    h5.style.font.color.rgb = COLOR_PRIMARY

    doc.add_paragraph("Table 1 summarizes the Evidently AI and PSI drift evaluation for the Online Shopping monitoring pipeline across baseline and shifted operational distributions.")

    summary_table = doc.add_table(rows=8, cols=3)
    summary_data = [
        ("Metric", "Normal Operation Batch", "Simulated Drift Batch"),
        ("Reference dataset rows", "800 records", "800 records"),
        ("Live prediction records", "250 records", "250 records"),
        ("Input features evaluated", "13 features (9 numerical PSI)", "13 features (9 numerical PSI)"),
        ("Drifted columns count", "0 columns drifted", "9 columns drifted"),
        ("Share of drifted columns", "0.00 (0.0%)", "1.00 (100.0%)"),
        ("Dataset drift threshold", "PSI >= 0.2 / 0.5", "PSI >= 0.2 / 0.5"),
        ("Dataset drift status", "NOT DETECTED (Healthy)", "DRIFT DETECTED (Review Required)")
    ]
    for r_idx, row in enumerate(summary_data):
        for c_idx, val in enumerate(row):
            summary_table.rows[r_idx].cells[c_idx].text = val
    style_table(summary_table, [Inches(2.5), Inches(2.0), Inches(2.0)])

    doc.add_paragraph()
    doc.add_paragraph("Table 2 details the feature-level Population Stability Index (PSI) values and statistical means.")

    psi_table = doc.add_table(rows=10, cols=5)
    psi_headers = ["Monitored Feature", "Ref Mean", "Normal Batch Mean (PSI)", "Drift Batch Mean (PSI)", "Drift Detected?"]
    psi_rows = [
        ("Age", "39.11", "37.60 (PSI: 0.0228)", "55.10 (PSI: 5.7777)", "Yes (Drift Batch)"),
        ("MonthlyIncome", "$81,897.21", "$82,339.46 (PSI: 0.0069)", "$32,497.31 (PSI: 8.4879)", "Yes (Drift Batch)"),
        ("WebsiteVisits", "15.25", "15.49 (PSI: 0.0098)", "3.48 (PSI: 8.8485)", "Yes (Drift Batch)"),
        ("PagesViewed", "10.50", "11.22 (PSI: 0.0151)", "2.58 (PSI: 8.9537)", "Yes (Drift Batch)"),
        ("TimeSpentMinutes", "30.66", "29.69 (PSI: 0.0111)", "6.48 (PSI: 9.2982)", "Yes (Drift Batch)"),
        ("PreviousPurchases", "7.56", "7.35 (PSI: 0.0209)", "0.67 (PSI: 9.7515)", "Yes (Drift Batch)"),
        ("CartItems", "4.99", "4.67 (PSI: 0.0105)", "0.67 (PSI: 8.8624)", "Yes (Drift Batch)"),
        ("CustomerRating", "2.98", "2.85 (PSI: 0.0304)", "1.88 (PSI: 5.6810)", "Yes (Drift Batch)"),
        ("PurchaseAmount", "$4,958.58", "$4,851.00 (PSI: 0.0103)", "$652.39 (PSI: 11.0524)", "Yes (Drift Batch)")
    ]
    for c_idx, h in enumerate(psi_headers):
        psi_table.rows[0].cells[c_idx].text = h
    for r_idx, row in enumerate(psi_rows):
        for c_idx, val in enumerate(row):
            psi_table.rows[r_idx+1].cells[c_idx].text = val
    style_table(psi_table, [Inches(1.5), Inches(1.0), Inches(1.5), Inches(1.5), Inches(1.0)])

    # 6. Screenshots & Figures Guide
    h6 = doc.add_heading("6. Screenshots and Evidence Guide", level=1)
    h6.style.font.color.rgb = COLOR_PRIMARY
    doc.add_paragraph(
        "The following figure references document the verification checkpoints matching the practical rubric:"
    )
    figures = [
        ("Figure 1", "Online Shopping Purchase Prediction project structure showing app.py, train_model.py, simulate_requests.py, drift_report.py, and docker-compose.yml."),
        ("Figure 2", "Live prediction dataset generation in terminal (python simulate_requests.py) alongside running Docker Compose containers (docker compose ps)."),
        ("Figure 3", "Online Shopping FastAPI health check response at http://localhost:8000/health confirming model readiness."),
        ("Figure 4", "Prometheus target monitoring screen at http://localhost:9090/targets verifying student_result_api endpoint in the UP state."),
        ("Figure 5", "Prometheus metrics query console at http://localhost:9090 executing ml_http_requests_total and student_api_http_requests_total."),
        ("Figure 6", "Grafana Online Shopping Monitoring Dashboard at http://localhost:3000 showing 4 active panels: Request Volume, P95 Latency, Error Rate, and Classification Split."),
        ("Figure 7", "Evidently AI live dataset drift overview (reports/drift_evidently_drift_report.html) demonstrating 9/9 drifted columns (100% drift share)."),
        ("Figure 8", "Evidently AI feature-wise data drift distributions highlighting PurchaseAmount (PSI: 11.05) and MonthlyIncome (PSI: 8.48)."),
        ("Figure 9", "Prediction log directory verification in VS Code / terminal displaying logs/predictions.jsonl file."),
        ("Figure 10", "Structured JSON Lines prediction payload records stored inside logs/predictions.jsonl.")
    ]
    for fig, desc in figures:
        p = doc.add_paragraph()
        r = p.add_run(f"{fig}. ")
        r.font.bold = True
        p.add_run(desc)

    # 7. Conclusion
    h7 = doc.add_heading("7. Conclusion", level=1)
    h7.style.font.color.rgb = COLOR_PRIMARY
    doc.add_paragraph(
        "An end-to-end MLOps monitoring architecture was successfully constructed, verified, and demonstrated for "
        "the Online Shopping Purchase Prediction machine learning application. Using Docker Compose, the FastAPI "
        "service, Prometheus telemetry collector, and Grafana dashboard were deployed as cohesive microservices. "
        "Prometheus demonstrated reliable scrape performance with zero dropouts, while Grafana delivered transparent "
        "operational insight into real-time request rates, latency, and prediction class distributions. Evidently AI "
        "and Population Stability Index (PSI) accurately identified data drift when consumer purchasing behaviors and "
        "demographics shifted, enabling proactive model retraining triggers before degraded performance affects business "
        "outcomes. The complete pipeline is version-controlled on GitHub and containerized on Docker Hub for reproducible "
        "production deployment."
    )

    targets = [
        r"C:\Users\Dell\online-shopping-purchase-prediction\Online_Shopping_MLOps_Monitoring_Report.docx",
        r"C:\Users\Dell\OneDrive\Desktop\Online_Shopping_Practical_12\Online_Shopping_MLOps_Monitoring_Report.docx",
        r"C:\Users\Dell\OneDrive\Desktop\online-shopping-purchase-prediction\Online_Shopping_MLOps_Monitoring_Report.docx"
    ]
    for path in targets:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        doc.save(path)
        print(f"Report saved to: {path}")

if __name__ == "__main__":
    create_report()
