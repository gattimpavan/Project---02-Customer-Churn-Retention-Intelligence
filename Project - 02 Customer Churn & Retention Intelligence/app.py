"""
Flask Application for Project - 02: Customer Churn & Retention Intelligence
Hosts the interactive executive dashboard, simulation REST APIs, customer XAI lookups, and PDF generation.
"""
import os
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request, send_file

from models.data_generator import generate_netflix_churn_data
from models.churn_model import ChurnPredictor
from models.segmentation import CustomerSegmenter
from models.root_cause_analyzer import RootCauseAnalyzer
from models.retention_engine import RetentionEngine
from report.generate_pdf_report import generate_pdf

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "netflix_subscribers.csv")
REPORT_PATH = os.path.join(BASE_DIR, "report", "Customer_Churn_Retention_Report.pdf")

# Global Cache to ensure lightning-fast dashboard responses
CACHE = {
    'df_segmented': None,
    'predictor': None,
    'segmenter': None,
    'rca': None,
    'ret_engine': None,
    'overview': None
}

def sanitize(val):
    """Converts numpy / pandas data types into JSON-serializable primitives."""
    if isinstance(val, (int, float, str, bool)) or val is None:
        return val
    elif hasattr(val, "item"):
        return val.item()
    elif isinstance(val, dict):
        return {k: sanitize(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple)):
        return [sanitize(v) for v in val]
    return str(val)

def initialize_models():
    """Initializes and caches models and pre-scored datasets on startup."""
    if not os.path.exists(CSV_PATH):
        generate_netflix_churn_data(DATA_DIR)
        
    df = pd.read_csv(CSV_PATH)
    predictor = ChurnPredictor(df)
    predictor.train_and_evaluate()
    scored_df = predictor.get_scored_data()
    
    segmenter = CustomerSegmenter(scored_df)
    segmented_df = segmenter.segment_customers()
    
    rca = RootCauseAnalyzer(segmented_df)
    ret_engine = RetentionEngine(segmented_df)
    
    CACHE['predictor'] = predictor
    CACHE['segmenter'] = segmenter
    CACHE['rca'] = rca
    CACHE['ret_engine'] = ret_engine
    CACHE['df_segmented'] = segmented_df
    
    # Pre-build Overview Payload
    metrics = predictor.metrics
    seg_summary = segmenter.get_segment_summary()
    root_causes = rca.get_root_cause_breakdown()
    survival_data = rca.compute_survival_curves()
    playbooks = ret_engine.get_playbook_catalog()
    default_sim = ret_engine.simulate_portfolio_retention(budget=30000.0)
    
    # Downsample probabilities for fast network payload
    all_probs = segmented_df['churn_probability'].tolist()
    step = max(1, len(all_probs) // 1500)
    sampled_probs = [round(float(p), 4) for p in all_probs[::step]]
    
    total_subscribers = len(segmented_df)
    churn_rate = float(segmented_df['churn'].mean() * 100)
    mrr_lost = float(segmented_df[segmented_df['churn'] == 1]['monthly_fee'].sum())
    
    CACHE['overview'] = {
        'kpis': {
            'total_subscribers': total_subscribers,
            'churn_rate_pct': round(churn_rate, 2),
            'mrr_at_risk': round(mrr_lost, 2),
            'model_auc': metrics['roc_auc'],
            'model_accuracy': metrics['accuracy'],
            'model_precision': metrics['precision'],
            'model_recall': metrics['recall'],
            'model_f1': metrics['f1']
        },
        'churn_distribution': {
            'probabilities': sampled_probs,
            'threshold': 0.50
        },
        'feature_importances': predictor.feature_importances,
        'segments': seg_summary,
        'root_causes': root_causes,
        'survival_curves': survival_data,
        'playbooks': playbooks,
        'simulation': default_sim
    }

# Initialize cache upon application import
initialize_models()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/overview", methods=["GET"])
def get_overview():
    if CACHE['overview'] is None:
        initialize_models()
    return jsonify(sanitize(CACHE['overview']))

@app.route("/api/customer/<customer_id>", methods=["GET"])
def get_customer(customer_id):
    if CACHE['predictor'] is None:
        initialize_models()
        
    predictor = CACHE['predictor']
    ret_engine = CACHE['ret_engine']
    
    profile = predictor.explain_customer(customer_id)
    if 'error' in profile:
        return jsonify(profile), 404
        
    prescription = ret_engine.prescribe_customer_strategy(profile)
    return jsonify({
        'profile': sanitize(profile),
        'prescription': sanitize(prescription)
    })

@app.route("/api/quick-subscribers", methods=["GET"])
def get_quick_subscribers():
    if CACHE['df_segmented'] is None:
        initialize_models()
        
    df = CACHE['df_segmented']
    samples = []
    
    # Pick 2 interesting samples from each segment
    for seg_name in df['segment'].unique():
        seg_rows = df[df['segment'] == seg_name]
        for _, r in seg_rows.head(2).iterrows():
            samples.append({
                'customer_id': r['customer_id'],
                'segment': seg_name,
                'risk_tier': r.get('risk_tier', 'Medium'),
                'churn_probability': round(float(r.get('churn_probability', 0.2)), 4),
                'subscription_tier': r['subscription_tier'],
                'monthly_fee': r['monthly_fee']
            })
            
    # Sort by churn probability descending
    samples.sort(key=lambda x: x['churn_probability'], reverse=True)
    return jsonify(samples)

@app.route("/api/simulate-retention", methods=["POST"])
def simulate_retention():
    if CACHE['ret_engine'] is None:
        initialize_models()
        
    data = request.get_json() or {}
    budget = float(data.get("budget", 30000.0))
    playbooks = data.get("active_playbooks", None)
    
    ret_engine = CACHE['ret_engine']
    sim_result = ret_engine.simulate_portfolio_retention(budget=budget, active_playbooks=playbooks)
    return jsonify(sanitize(sim_result))

@app.route("/api/download-pdf", methods=["GET"])
def download_pdf():
    if not os.path.exists(REPORT_PATH):
        generate_pdf(REPORT_PATH)
    return send_file(
        REPORT_PATH,
        as_attachment=True,
        download_name="Customer_Churn_Retention_Report.pdf",
        mimetype="application/pdf"
    )

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
