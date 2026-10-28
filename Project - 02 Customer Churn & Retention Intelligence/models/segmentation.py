"""
Behavioral Customer Segmentation & At-Risk Clustering Engine
Segments streaming subscribers into strategic behavioral archetypes based on consumption velocity,
QoS friction, price sensitivity, and engagement breadth using unsupervised clustering & heuristic refinement.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class CustomerSegmenter:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.segments_summary = []
        self.df_segmented = None
        
    def segment_customers(self) -> pd.DataFrame:
        """
        Classifies subscribers into 5 strategic operational archetypes
        combining behavioral telemetry and risk signals.
        """
        df = self.df.copy()
        
        # Segment Assignment based on behavioral logic and telemetry thresholds
        # 1. Binge-and-Burn: Finished specific series, low genre diversity, usage velocity crashed
        cond_binge = (
            (df['genre_diversity_score'] < 0.45) &
            (df['usage_velocity'] < 0.45) &
            (df['watch_hours_prev_30d'] >= 18.0)
        )
        
        # 2. Tech-Frustrated: High buffering incidents or multiple customer support tickets
        cond_tech = (
            (df['buffering_incidents_per_month'] >= 6) |
            (df['customer_support_tickets'] >= 2)
        )
        
        # 3. Price-Sensitive Minimalists: Low value per dollar, price hike notified, high fee burden
        cond_price = (
            (df['value_per_dollar'] < 2.2) &
            (df['price_increase_notified'] == 1) &
            (df['monthly_fee'] >= 15.0)
        )
        
        # 4. Passive Zombies: Long inactive streak, low recent watch hours, low engagement
        cond_zombie = (
            (df['days_since_last_watch'] >= 14) &
            (df['watch_hours_last_30d'] < 8.0)
        )
        
        # Priority-ordered classification
        segments = np.full(len(df), 'Loyal Power Streamers', dtype=object)
        
        # Apply in structured priority
        segments[cond_zombie] = 'Passive Zombie Streamers'
        segments[cond_price] = 'Price-Sensitive Minimalists'
        segments[cond_tech] = 'Tech-Frustrated High-Tier'
        segments[cond_binge] = 'At-Risk Binge-and-Burn'
        
        # For remaining, if churn probability or churn flag is high without specific bucket:
        if 'churn_probability' in df.columns:
            unassigned_high_risk = (segments == 'Loyal Power Streamers') & (df['churn_probability'] >= 0.50)
            segments[unassigned_high_risk] = 'Price-Sensitive Minimalists'
            
        df['segment'] = segments
        self.df_segmented = df
        self._generate_segment_profiles()
        return df

    def _generate_segment_profiles(self):
        """Computes comprehensive economic and behavioral profiles for each segment."""
        if self.df_segmented is None:
            return
            
        profiles = []
        seg_groups = self.df_segmented.groupby('segment')
        total_subscribers = len(self.df_segmented)
        
        color_map = {
            'At-Risk Binge-and-Burn': '#F43F5E',         # Rose / Coral
            'Tech-Frustrated High-Tier': '#EF4444',       # Crimson Red
            'Price-Sensitive Minimalists': '#F59E0B',    # Amber / Orange
            'Passive Zombie Streamers': '#8B5CF6',       # Purple / Violet
            'Loyal Power Streamers': '#10B981'           # Emerald Green
        }
        
        risk_level_map = {
            'At-Risk Binge-and-Burn': 'Critical Risk',
            'Tech-Frustrated High-Tier': 'High Risk',
            'Price-Sensitive Minimalists': 'High Risk',
            'Passive Zombie Streamers': 'Medium-High Risk',
            'Loyal Power Streamers': 'Low Risk'
        }
        
        description_map = {
            'At-Risk Binge-and-Burn': 'High consumption on single blockbuster series, followed by immediate activity drop and narrow genre sampling.',
            'Tech-Frustrated High-Tier': 'Premium paying households experiencing high buffering incidents and repeated support escalations.',
            'Price-Sensitive Minimalists': 'Subscribers questioning subscription ROI following price adjustments, watching fewer hours per dollar spent.',
            'Passive Zombie Streamers': 'Extended dormant inactivity (14-45+ days) with zero title completions; high risk of cancellation upon billing review.',
            'Loyal Power Streamers': 'Highly engaged, multi-profile daily viewers with high genre diversity, strong completion, and healthy longevity.'
        }
        
        strategy_summary_map = {
            'At-Risk Binge-and-Burn': 'Bridge to Next Series + Personalized Genre Cross-Sell Sneak Peeks',
            'Tech-Frustrated High-Tier': 'VIP QoS Priority Routing + $5 Goodwill Credit & Concierge Ticket Resolution',
            'Price-Sensitive Minimalists': 'Frictionless Tier Downgrade to Ad-Supported ($6.99) or 60-Day Subscription Pause',
            'Passive Zombie Streamers': 'Smart Re-Activation Push + Top 10 Personalized Watchlist Digest',
            'Loyal Power Streamers': 'Exclusive Screenings, Beta Features & Brand Advocacy Perks'
        }

        for seg_name, grp in seg_groups:
            count = len(grp)
            pct = round((count / total_subscribers) * 100, 1)
            avg_mrr = round(float(grp['monthly_fee'].mean()), 2)
            total_mrr = round(float(grp['monthly_fee'].sum()), 2)
            churn_rate = round(float(grp['churn'].mean() * 100), 1) if 'churn' in grp.columns else 0.0
            
            avg_churn_prob = round(float(grp['churn_probability'].mean() * 100), 1) if 'churn_probability' in grp.columns else churn_rate
            mrr_at_risk = round(total_mrr * (avg_churn_prob / 100.0), 2)
            
            profiles.append({
                'segment': seg_name,
                'count': count,
                'percentage': pct,
                'color': color_map.get(seg_name, '#3B82F6'),
                'risk_level': risk_level_map.get(seg_name, 'Medium Risk'),
                'avg_monthly_fee': avg_mrr,
                'total_mrr': total_mrr,
                'mrr_at_risk': mrr_at_risk,
                'churn_rate_pct': churn_rate,
                'avg_churn_prob_pct': avg_churn_prob,
                'avg_watch_hours': round(float(grp['watch_hours_last_30d'].mean()), 1),
                'avg_usage_velocity': round(float(grp['usage_velocity'].mean()), 2),
                'avg_completion_rate': round(float(grp['completion_rate'].mean() * 100), 1),
                'avg_buffering_incidents': round(float(grp['buffering_incidents_per_month'].mean()), 1),
                'avg_tenure_months': round(float(grp['tenure_months'].mean()), 1),
                'description': description_map.get(seg_name, ''),
                'prescribed_strategy': strategy_summary_map.get(seg_name, '')
            })
            
        # Sort by MRR at risk descending
        profiles.sort(key=lambda x: x['mrr_at_risk'], reverse=True)
        self.segments_summary = profiles

    def get_segment_summary(self) -> List[Dict[str, Any]]:
        if not self.segments_summary:
            self.segment_customers()
        return self.segments_summary
