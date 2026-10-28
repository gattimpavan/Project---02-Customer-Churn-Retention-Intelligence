"""
One-Click Master Orchestrator for Project - 02: Customer Churn & Retention Intelligence
Executes subscriber telemetry data generation, predictive ensemble modeling, at-risk customer segmentation,
root cause diagnostics, generates the publication-grade PDF report, and launches the interactive web dashboard.
"""
import os
import sys

# Ensure local packages are importable
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from models.data_generator import generate_netflix_churn_data
from models.churn_model import ChurnPredictor
from models.segmentation import CustomerSegmenter
from models.root_cause_analyzer import RootCauseAnalyzer
from models.retention_engine import RetentionEngine
from report.generate_pdf_report import generate_pdf
import pandas as pd

def main():
    print("=" * 85)
    print(" PROJECT - 02: CUSTOMER CHURN & RETENTION INTELLIGENCE")
    print(" Netflix Subscription Attrition Diagnostics & Prescriptive AI Optimization Engine")
    print("=" * 85)
    
    data_dir = os.path.join(base_dir, "data")
    csv_path = os.path.join(data_dir, "netflix_subscribers.csv")
    report_path = os.path.join(base_dir, "report", "Customer_Churn_Retention_Report.pdf")

    # Step 1: Data Generation
    print("\n[Step 1/4] Checking and generating Netflix subscriber telemetry dataset...")
    if not os.path.exists(csv_path):
        df = generate_netflix_churn_data(data_dir, n_samples=10000)
    else:
        df = pd.read_csv(csv_path)
    churn_rate = (df['churn'].mean() * 100)
    print(f"  ✓ Processed {len(df):,} subscriber records. Observed Cohort Churn: {churn_rate:.2f}%.")

    # Step 2: AI Predictive Modeling & Segmentation
    print("\n[Step 2/4] Training Gradient Boosting Ensemble & Clustering At-Risk Archetypes...")
    predictor = ChurnPredictor(df)
    metrics = predictor.train_and_evaluate()
    print(f"  ✓ Model Converged | ROC-AUC: {metrics['roc_auc']:.4f} | Accuracy: {metrics['accuracy']*100:.2f}% | F1: {metrics['f1']:.4f}")
    
    scored_df = predictor.get_scored_data()
    segmenter = CustomerSegmenter(scored_df)
    segmented_df = segmenter.segment_customers()
    seg_summary = segmenter.get_segment_summary()
    print(f"  ✓ Identified {len(seg_summary)} Strategic Customer Segments:")
    for s in seg_summary:
        print(f"    • {s['segment']:<28} | Size: {s['count']:,} ({s['percentage']}%) | MRR at Risk: ${s['mrr_at_risk']:,.2f}")

    # Step 3: Prescriptive Retention Economics & PDF Generation
    print("\n[Step 3/4] Generating Publication-Grade Executive PDF Intelligence Report...")
    pdf_file = generate_pdf(report_path)
    file_size_kb = round(os.path.getsize(pdf_file) / 1024, 1)
    print(f"  ✓ PDF Intelligence Report generated: {pdf_file} ({file_size_kb} KB)")

    # Step 4: Web Dashboard Launch
    print("\n[Step 4/4] Starting Flask Interactive Executive Web Server...")
    print("  ✓ Access URL:           http://127.0.0.1:5001")
    print("  ✓ Direct PDF Download:  http://127.0.0.1:5001/api/download-pdf")
    print("  ✓ What-If Simulator:    http://127.0.0.1:5001/#simulator-section")
    print("  ✓ Subscriber Lookup:    http://127.0.0.1:5001/#lookup-section")
    print("\nLaunching Flask development server on port 5001 (Press Ctrl+C to terminate)...")
    
    from app import app
    app.run(host="127.0.0.1", port=5001, debug=False)

if __name__ == "__main__":
    main()
