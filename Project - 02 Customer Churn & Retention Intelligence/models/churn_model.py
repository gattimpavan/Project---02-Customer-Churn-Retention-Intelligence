"""
Predictive Churn Modeling & Behavioral Explainability Engine
Trains Random Forest & Gradient Boosting classifiers on subscriber telemetry,
computes validation metrics (ROC-AUC, F1, Precision, Recall), extracts feature importances,
and delivers individualized churn explainability.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.preprocessing import StandardScaler

class ChurnPredictor:
    def __init__(self, df: pd.DataFrame):
        self.raw_df = df.copy()
        self.feature_cols = [
            'tenure_months', 'monthly_fee', 'num_profiles', 'weekly_watch_hours',
            'watch_hours_last_30d', 'watch_hours_prev_30d', 'usage_velocity',
            'days_since_last_watch', 'completion_rate', 'titles_started_last_30d',
            'search_to_watch_ratio', 'genre_diversity_score', 'buffering_incidents_per_month',
            'customer_support_tickets', 'price_increase_notified', 'shared_account_warning',
            'offline_downloads_used', 'avg_session_mins', 'value_per_dollar',
            'payment_retry_count', 'is_annual', 'tier_standard_ads', 'tier_premium',
            'device_mobile', 'device_smart_tv', 'friction_index', 'discovery_paralysis'
        ]
        self.model = None
        self.scaler = StandardScaler()
        self.metrics = {}
        self.feature_importances = []
        self.roc_curve_data = {}
        self.df_scored = None
        
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineers composite features and one-hot encodes categoricals."""
        feat_df = df.copy()
        
        # Binary / One-hot transforms
        feat_df['is_annual'] = (feat_df['billing_frequency'] == 'Annual').astype(float)
        feat_df['tier_standard_ads'] = (feat_df['subscription_tier'] == 'Standard with Ads').astype(float)
        feat_df['tier_premium'] = (feat_df['subscription_tier'] == 'Premium').astype(float)
        feat_df['device_mobile'] = (feat_df['primary_device'] == 'Mobile').astype(float)
        feat_df['device_smart_tv'] = (feat_df['primary_device'] == 'Smart TV').astype(float)
        
        # Composite Behavioral Risk Features
        feat_df['friction_index'] = (
            feat_df['buffering_incidents_per_month'] * 1.5 +
            feat_df['customer_support_tickets'] * 2.0 +
            feat_df['payment_retry_count'] * 3.0
        )
        feat_df['discovery_paralysis'] = feat_df['search_to_watch_ratio'] * (1.0 - feat_df['completion_rate'])
        
        return feat_df
        
    def train_and_evaluate(self) -> Dict[str, Any]:
        """Trains the ensemble classifier and computes evaluation metrics."""
        feat_df = self._engineer_features(self.raw_df)
        X = feat_df[self.feature_cols]
        y = feat_df['churn']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        # Gradient Boosting Classifier for high non-linear interaction capture
        self.model = GradientBoostingClassifier(
            n_estimators=160,
            learning_rate=0.07,
            max_depth=4,
            subsample=0.85,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Evaluation predictions
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        # Metrics
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        
        self.metrics = {
            'accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
            'precision': round(float(precision_score(y_test, y_pred)), 4),
            'recall': round(float(recall_score(y_test, y_pred)), 4),
            'f1': round(float(f1_score(y_test, y_pred)), 4),
            'roc_auc': round(float(roc_auc_score(y_test, y_prob)), 4),
            'confusion_matrix': {
                'true_negative': int(cm[0, 0]),
                'false_positive': int(cm[0, 1]),
                'false_negative': int(cm[1, 0]),
                'true_positive': int(cm[1, 1])
            }
        }
        
        # ROC Curve points (downsampled for clean payload)
        sample_indices = np.linspace(0, len(fpr) - 1, min(50, len(fpr))).astype(int)
        self.roc_curve_data = {
            'fpr': [round(float(fpr[i]), 4) for i in sample_indices],
            'tpr': [round(float(tpr[i]), 4) for i in sample_indices]
        }
        
        # Feature importances
        raw_importances = self.model.feature_importances_
        feature_imp_list = [
            {'feature': col, 'importance': round(float(imp), 4), 'displayName': self._format_feature_name(col)}
            for col, imp in zip(self.feature_cols, raw_importances)
        ]
        feature_imp_list.sort(key=lambda x: x['importance'], reverse=True)
        self.feature_importances = feature_imp_list
        
        # Score entire dataset
        all_probs = self.model.predict_proba(X)[:, 1]
        scored_df = self.raw_df.copy()
        scored_df['churn_probability'] = np.round(all_probs, 4)
        
        # Assign risk tier
        def assign_risk_tier(p):
            if p >= 0.70:
                return 'Critical'
            elif p >= 0.45:
                return 'High'
            elif p >= 0.25:
                return 'Medium'
            else:
                return 'Low'
                
        scored_df['risk_tier'] = scored_df['churn_probability'].apply(assign_risk_tier)
        self.df_scored = scored_df
        
        return self.metrics

    def _format_feature_name(self, col: str) -> str:
        name_map = {
            'usage_velocity': 'Usage Velocity (30d vs 90d)',
            'days_since_last_watch': 'Days Inactive (Recency)',
            'completion_rate': 'Content Completion Rate',
            'friction_index': 'Technical & Billing Friction',
            'discovery_paralysis': 'Discovery Paralysis Ratio',
            'buffering_incidents_per_month': 'Buffering Incidents / Mo',
            'search_to_watch_ratio': 'Search-to-Watch Ratio',
            'value_per_dollar': 'Perceived Value (Hours/$)',
            'customer_support_tickets': 'Support Tickets (60d)',
            'price_increase_notified': 'Price Hike Notification',
            'genre_diversity_score': 'Genre Diversity (Breadth)',
            'tenure_months': 'Subscriber Tenure (Months)',
            'payment_retry_count': 'Payment Retry Failures',
            'monthly_fee': 'Monthly Subscription Fee',
            'watch_hours_last_30d': 'Watch Hours (Last 30d)',
            'weekly_watch_hours': 'Weekly Watch Intensity',
            'titles_started_last_30d': 'Titles Sampled (Last 30d)',
            'tier_premium': 'Premium Tier Plan',
            'tier_standard_ads': 'Standard with Ads Plan',
            'is_annual': 'Annual Billing Cycle'
        }
        return name_map.get(col, col.replace('_', ' ').title())

    def get_scored_data(self) -> pd.DataFrame:
        if self.df_scored is None:
            self.train_and_evaluate()
        return self.df_scored

    def explain_customer(self, customer_id: str) -> Dict[str, Any]:
        """Provides individualized behavioral diagnostics for a specific subscriber."""
        if self.df_scored is None:
            self.train_and_evaluate()
            
        row = self.df_scored[self.df_scored['customer_id'] == customer_id]
        if row.empty:
            return {'error': f'Customer {customer_id} not found.'}
            
        rec = row.iloc[0].to_dict()
        p = rec['churn_probability']
        
        # Evaluate primary risk catalysts
        drivers = []
        if rec['usage_velocity'] < 0.40:
            drop_pct = round((1.0 - rec['usage_velocity']) * 100, 1)
            drivers.append({
                'category': 'Engagement Collapse',
                'description': f"Usage velocity crashed by {drop_pct}% compared to prior baseline.",
                'severity': 'Critical'
            })
            
        if rec['days_since_last_watch'] >= 14:
            drivers.append({
                'category': 'Prolonged Inactivity',
                'description': f"Subscriber has been inactive for {rec['days_since_last_watch']} consecutive days.",
                'severity': 'High' if rec['days_since_last_watch'] < 30 else 'Critical'
            })
            
        if rec['buffering_incidents_per_month'] >= 6 or rec['customer_support_tickets'] >= 2:
            drivers.append({
                'category': 'Quality of Service Friction',
                'description': f"{rec['buffering_incidents_per_month']} buffering events and {rec['customer_support_tickets']} support contacts recorded.",
                'severity': 'High'
            })
            
        if rec['search_to_watch_ratio'] >= 2.5 or rec['completion_rate'] < 0.40:
            drivers.append({
                'category': 'Content Discovery Paralysis',
                'description': f"High search ratio ({rec['search_to_watch_ratio']}x) and low completion ({rec['completion_rate']*100:.0f}%), signaling catalog fatigue.",
                'severity': 'Medium'
            })
            
        if rec['price_increase_notified'] == 1 and rec['value_per_dollar'] < 1.8:
            drivers.append({
                'category': 'Price Sensitivity & Value Mismatch',
                'description': f"Recent price increase notice with low perceived consumption value ({rec['value_per_dollar']} hrs/$).",
                'severity': 'High'
            })
            
        if rec['payment_retry_count'] >= 1:
            drivers.append({
                'category': 'Billing / Involuntary Failure',
                'description': f"{rec['payment_retry_count']} payment retry failures detected on file.",
                'severity': 'Critical' if rec['payment_retry_count'] >= 2 else 'Medium'
            })
            
        if not drivers:
            drivers.append({
                'category': 'Healthy Engagement',
                'description': 'Normal streaming telemetry with high completion and low friction.',
                'severity': 'Low'
            })
            
        rec['churn_drivers'] = drivers
        return rec
