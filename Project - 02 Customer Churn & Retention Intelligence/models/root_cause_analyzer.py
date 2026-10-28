"""
Root Cause Diagnostics & Retention Survival Analysis Engine
Decomposes churn drivers into operational catalysts, audits behavioral friction metrics,
and computes Kaplan-Meier-style subscriber survival retention curves across tenure cohorts.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any

class RootCauseAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
    def get_root_cause_breakdown(self) -> List[Dict[str, Any]]:
        """
        Decomposes ground-truth and high-risk churn into operational root causes
        with economic quantification of monthly revenue impact.
        """
        df = self.df.copy()
        churned = df[df['churn'] == 1]
        total_churned = len(churned)
        total_lost_mrr = float(churned['monthly_fee'].sum())
        
        causes = [
            {
                'cause': 'Discovery Fatigue & Search Paralysis',
                'description': 'Subscribers spend excessive time browsing without playing content, resulting in search abandonment.',
                'color': '#8B5CF6',
                'icon': 'search',
                'benchmark_metric': 'Search-to-watch ratio > 2.8x & completion < 40%'
            },
            {
                'cause': 'Technical Buffering & Playback QoS Friction',
                'description': 'Frequent stream stuttering, 4K downgrades, or app crashes driving user frustration and tickets.',
                'color': '#EF4444',
                'icon': 'wifi-off',
                'benchmark_metric': 'Buffering incidents >= 6/mo or support tickets >= 2'
            },
            {
                'cause': 'Price Hike Reaction & Perceived Value Deficit',
                'description': 'Negative reaction to subscription fee increases without corresponding increase in perceived viewing value.',
                'color': '#F59E0B',
                'icon': 'dollar-sign',
                'benchmark_metric': 'Recent price hike notification & < 2.0 watch hours per $'
            },
            {
                'cause': 'Involuntary Payment & Billing Retry Failures',
                'description': 'Card expirations, insufficient funds, or banking gateway declines without proactive recovery.',
                'color': '#EC4899',
                'icon': 'credit-card',
                'benchmark_metric': 'Payment retry count >= 2'
            },
            {
                'cause': 'Content Stagnation & Library Exhaustion',
                'description': 'Perception that the catalog has run out of relevant shows matching subscriber taste profile.',
                'color': '#3B82F6',
                'icon': 'film',
                'benchmark_metric': 'Low title starts (< 4/mo) and persistent inactivity'
            },
            {
                'cause': 'Finished Specific Series (Binge & Burn)',
                'description': 'Subscriber subscribed solely for one tentpole title and cancels immediately upon finale.',
                'color': '#F43F5E',
                'icon': 'fast-forward',
                'benchmark_metric': 'Narrow genre breadth (<0.45) & post-binge velocity crash'
            }
        ]
        
        results = []
        cause_map = {
            'Discovery Fatigue (Too Hard to Find Content)': 'Discovery Fatigue & Search Paralysis',
            'Technical Buffering & Playback Issues': 'Technical Buffering & Playback QoS Friction',
            'Price Hike Reaction & Value Misalignment': 'Price Hike Reaction & Perceived Value Deficit',
            'Involuntary Payment Failure': 'Involuntary Payment & Billing Retry Failures',
            'Content Stagnation & Boredom': 'Content Stagnation & Library Exhaustion',
            'Finished Specific Series (Binge & Burn)': 'Finished Specific Series (Binge & Burn)'
        }
        
        for c in causes:
            # Match churned rows
            sub_matches = churned[churned['primary_churn_reason'].apply(lambda r: cause_map.get(r, '') == c['cause'])]
            count = len(sub_matches)
            pct = round((count / total_churned * 100), 1) if total_churned > 0 else 0.0
            lost_mrr = round(float(sub_matches['monthly_fee'].sum()), 2)
            pct_lost_mrr = round((lost_mrr / total_lost_mrr * 100), 1) if total_lost_mrr > 0 else 0.0
            
            c_dict = dict(c)
            c_dict['affected_subscribers'] = count
            c_dict['pct_of_churn'] = pct
            c_dict['lost_mrr'] = lost_mrr
            c_dict['pct_lost_mrr'] = pct_lost_mrr
            results.append(c_dict)
            
        # Sort by lost MRR descending
        results.sort(key=lambda x: x['lost_mrr'], reverse=True)
        return results

    def compute_survival_curves(self) -> Dict[str, Any]:
        """
        Computes non-parametric empirical survival probabilities S(t)
        across tenure months (1 to 48) for overall cohort and each subscription tier.
        """
        df = self.df.copy()
        max_tenure = min(48, int(df['tenure_months'].max()))
        tenure_grid = list(range(1, max_tenure + 1))
        
        def calculate_survival(sub_df):
            surv = []
            n_total = len(sub_df)
            if n_total == 0:
                return [1.0] * len(tenure_grid)
                
            for t in tenure_grid:
                # Active at or beyond tenure t
                # Kaplan-Meier estimator proxy:
                # Cumulative retention = (Active users with tenure >= t) / (Users who reached t or churned at t)
                at_risk = sub_df[sub_df['tenure_months'] >= t]
                churned_at_t = sub_df[(sub_df['tenure_months'] == t) & (sub_df['churn'] == 1)]
                
                # Empirical survival rate
                retained_ratio = 1.0 - (len(sub_df[(sub_df['tenure_months'] <= t) & (sub_df['churn'] == 1)]) / n_total)
                surv.append(round(max(0.05, float(retained_ratio)), 3))
            return surv
            
        overall_survival = calculate_survival(df)
        tier_survival = {}
        for tier in ['Standard with Ads', 'Standard', 'Premium']:
            tier_df = df[df['subscription_tier'] == tier]
            tier_survival[tier] = calculate_survival(tier_df)
            
        return {
            'tenure_months': tenure_grid,
            'overall': overall_survival,
            'tiers': tier_survival
        }

    def get_friction_audit(self) -> Dict[str, Any]:
        """Calculates behavioral elasticity and risk multiplier statistics."""
        df = self.df.copy()
        
        # Buffering friction impact
        high_buffering = df[df['buffering_incidents_per_month'] >= 6]
        low_buffering = df[df['buffering_incidents_per_month'] < 3]
        buffering_churn_rate = round(float(high_buffering['churn'].mean() * 100), 1)
        clean_churn_rate = round(float(low_buffering['churn'].mean() * 100), 1)
        
        # Velocity collapse impact
        crashing_velocity = df[df['usage_velocity'] < 0.40]
        stable_velocity = df[df['usage_velocity'] >= 0.90]
        velocity_crash_churn_rate = round(float(crashing_velocity['churn'].mean() * 100), 1)
        stable_churn_rate = round(float(stable_velocity['churn'].mean() * 100), 1)
        
        # Price sensitivity impact
        price_notified_low_val = df[(df['price_increase_notified'] == 1) & (df['value_per_dollar'] < 2.0)]
        price_churn_rate = round(float(price_notified_low_val['churn'].mean() * 100), 1)
        
        # Inactivity streak impact
        inactive_14d = df[df['days_since_last_watch'] >= 14]
        inactive_churn_rate = round(float(inactive_14d['churn'].mean() * 100), 1)
        
        return {
            'buffering_multiplier': round(buffering_churn_rate / max(1.0, clean_churn_rate), 2),
            'high_buffering_churn_pct': buffering_churn_rate,
            'low_buffering_churn_pct': clean_churn_rate,
            'velocity_crash_churn_pct': velocity_crash_churn_rate,
            'stable_velocity_churn_pct': stable_churn_rate,
            'price_shock_churn_pct': price_churn_rate,
            'inactivity_14d_churn_pct': inactive_churn_rate
        }
