"""
Personalized Retention Strategy & Prescriptive AI Optimization Engine
Generates targeted retention playbooks for at-risk cohorts, computes unit economics,
evaluates campaign ROI (CLV saved vs intervention cost), and runs what-if simulations.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any

class RetentionEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.playbooks = self._initialize_playbooks()
        
    def _initialize_playbooks(self) -> Dict[str, Dict[str, Any]]:
        return {
            'binge_bridge': {
                'id': 'binge_bridge',
                'name': 'Next-Series Bridge & Sneak Peek',
                'target_segment': 'At-Risk Binge-and-Burn',
                'channel': 'Mobile Push & In-App Hero Carousel',
                'intervention_cost': 0.25,
                'relative_churn_reduction': 0.28,
                'offer_type': 'Curated Content + Trailer Teaser',
                'message_copy': 'Loved your recent series? Watch an exclusive 5-minute sneak peek of the upcoming sequel and your matched thriller.',
                'description': 'Intercepts post-season binge fatigue with tailored series recommendations in complementary genres.'
            },
            'tech_concierge': {
                'id': 'tech_concierge',
                'name': 'VIP QoS Concierge & Goodwill Credit',
                'target_segment': 'Tech-Frustrated High-Tier',
                'channel': 'Proactive Email + In-App Notice + $5 Credit',
                'intervention_cost': 5.20,
                'relative_churn_reduction': 0.42,
                'offer_type': 'Network Route Fix + $5 Courtesy Discount',
                'message_copy': 'We detected minor buffering during your last stream. We have optimized your CDN delivery and credited $5.00 to your next bill.',
                'description': 'Resolves playback friction proactively before the subscriber files complaints or cancels.'
            },
            'value_pause_downgrade': {
                'id': 'value_pause_downgrade',
                'name': 'Value-Preservation Downgrade & Pause Option',
                'target_segment': 'Price-Sensitive Minimalists',
                'channel': 'Smart Intercept Modal & Email Offer',
                'intervention_cost': 3.50,
                'relative_churn_reduction': 0.36,
                'offer_type': 'Pause for 60 Days OR Switch to Ad-Tier ($6.99)',
                'message_copy': 'Need a break? Pause your subscription for up to 60 days without losing your watchlist, or switch to our $6.99 plan.',
                'description': 'Prevents hard cancellation by offering flexible downgrades or temporary billing pauses.'
            },
            'zombie_wakeup': {
                'id': 'zombie_wakeup',
                'name': 'Personalized Watchlist Re-Activation',
                'target_segment': 'Passive Zombie Streamers',
                'channel': 'Email Digest + Dynamic App Push',
                'intervention_cost': 0.35,
                'relative_churn_reduction': 0.22,
                'offer_type': 'Top 10 Curated Playlist + 1-Click Resume',
                'message_copy': 'Your favorite show has new episodes waiting! Tap here to resume where you left off in 1 tap.',
                'description': 'Awakens dormant subscribers through personalized social triggers and frictionless resume links.'
            },
            'smart_dunning': {
                'id': 'smart_dunning',
                'name': '1-Click Apple/Google Pay Dunning Recovery',
                'target_segment': 'Involuntary Payment Risk',
                'channel': 'SMS + Push Deep Link',
                'intervention_cost': 0.40,
                'relative_churn_reduction': 0.65,
                'offer_type': 'Instant 1-Click Wallet Auth + 3-Day Grace Period',
                'message_copy': 'Your payment method failed. Tap here to instantly authorize via Apple Pay or Google Pay without password entry.',
                'description': 'Automates graceful payment recovery to eradicate involuntary involuntary churn.'
            },
            'power_advocacy': {
                'id': 'power_advocacy',
                'name': 'Loyalty Perks & VIP Preview Club',
                'target_segment': 'Loyal Power Streamers',
                'channel': 'In-App VIP Badge & Private Screening Link',
                'intervention_cost': 0.80,
                'relative_churn_reduction': 0.12,
                'offer_type': 'Exclusive Early Screening Invitations',
                'message_copy': 'Thank you for being a valued subscriber. You have been selected for early access to our upcoming premiere.',
                'description': 'Deepens emotional brand loyalty among high-value champions.'
            }
        }

    def get_playbook_catalog(self) -> List[Dict[str, Any]]:
        return list(self.playbooks.values())

    def prescribe_customer_strategy(self, customer_record: Dict[str, Any]) -> Dict[str, Any]:
        """Prescribes the optimal personalized retention action plan for an individual subscriber."""
        p = customer_record.get('churn_probability', 0.20)
        segment = customer_record.get('segment', 'Loyal Power Streamers')
        payment_retries = customer_record.get('payment_retry_count', 0)
        
        # Determine playbook
        if payment_retries >= 2:
            pb = self.playbooks['smart_dunning']
        elif segment == 'At-Risk Binge-and-Burn':
            pb = self.playbooks['binge_bridge']
        elif segment == 'Tech-Frustrated High-Tier':
            pb = self.playbooks['tech_concierge']
        elif segment == 'Price-Sensitive Minimalists':
            pb = self.playbooks['value_pause_downgrade']
        elif segment == 'Passive Zombie Streamers':
            pb = self.playbooks['zombie_wakeup']
        else:
            pb = self.playbooks['power_advocacy']
            
        fee = customer_record.get('monthly_fee', 15.49)
        baseline_churn = p
        reduction = pb['relative_churn_reduction']
        new_churn = round(baseline_churn * (1.0 - reduction), 4)
        prob_saved = round(baseline_churn * reduction, 4)
        
        # Annualized revenue saved
        annual_saved_expected = round(prob_saved * fee * 12, 2)
        net_expected_profit = round(annual_saved_expected - pb['intervention_cost'], 2)
        roi_pct = round((net_expected_profit / pb['intervention_cost']) * 100, 1) if pb['intervention_cost'] > 0 else 999.0
        
        return {
            'customer_id': customer_record.get('customer_id'),
            'segment': segment,
            'current_churn_prob': round(baseline_churn, 4),
            'projected_churn_prob': new_churn,
            'prob_retention_saved': prob_saved,
            'playbook_id': pb['id'],
            'strategy_name': pb['name'],
            'recommended_channel': pb['channel'],
            'offer_type': pb['offer_type'],
            'personalized_copy': pb['message_copy'],
            'intervention_cost': pb['intervention_cost'],
            'expected_annual_saved': annual_saved_expected,
            'net_expected_value': net_expected_profit,
            'roi_pct': roi_pct
        }

    def simulate_portfolio_retention(
        self,
        budget: float = 30000.0,
        active_playbooks: List[str] = None
    ) -> Dict[str, Any]:
        """
        Simulates running AI targeted retention campaigns across the subscriber portfolio
        subject to an intervention budget constraint.
        """
        df = self.df.copy()
        if active_playbooks is None:
            active_playbooks = list(self.playbooks.keys())
            
        # Segment mapping to playbooks
        seg_pb_map = {
            'At-Risk Binge-and-Burn': 'binge_bridge',
            'Tech-Frustrated High-Tier': 'tech_concierge',
            'Price-Sensitive Minimalists': 'value_pause_downgrade',
            'Passive Zombie Streamers': 'zombie_wakeup',
            'Loyal Power Streamers': 'power_advocacy'
        }
        
        candidates = []
        for idx, row in df.iterrows():
            if row.get('payment_retry_count', 0) >= 2:
                pb_id = 'smart_dunning'
            else:
                pb_id = seg_pb_map.get(row.get('segment', ''), 'power_advocacy')
                
            if pb_id not in active_playbooks:
                continue
                
            pb = self.playbooks[pb_id]
            prob = row.get('churn_probability', 0.20)
            fee = row.get('monthly_fee', 15.49)
            
            # Prioritize candidates with higher risk
            if prob < 0.20:
                continue
                
            prob_saved = prob * pb['relative_churn_reduction']
            ann_revenue_saved = prob_saved * fee * 12
            cost = pb['intervention_cost']
            net_val = ann_revenue_saved - cost
            efficiency = net_val / cost if cost > 0 else 999.0
            
            candidates.append({
                'customer_id': row['customer_id'],
                'segment': row.get('segment', ''),
                'monthly_fee': fee,
                'baseline_churn_prob': prob,
                'prob_saved': prob_saved,
                'playbook_id': pb_id,
                'cost': cost,
                'mrr_saved': prob_saved * fee,
                'annual_revenue_saved': ann_revenue_saved,
                'net_value': net_val,
                'efficiency': efficiency
            })
            
        # Sort candidates by efficiency descending (Knapsack greedy prioritization)
        candidates.sort(key=lambda x: x['efficiency'], reverse=True)
        
        allocated_candidates = []
        spent = 0.0
        for c in candidates:
            if spent + c['cost'] <= budget:
                allocated_candidates.append(c)
                spent += c['cost']
                
        # Aggregate impact
        n_targeted = len(allocated_candidates)
        total_subscribers = len(df)
        total_mrr_baseline = float(df['monthly_fee'].sum())
        
        saved_subscribers = round(sum(c['prob_saved'] for c in allocated_candidates), 1)
        saved_mrr = round(sum(c['mrr_saved'] for c in allocated_candidates), 2)
        saved_annual = round(sum(c['annual_revenue_saved'] for c in allocated_candidates), 2)
        net_annual_profit = round(saved_annual - spent, 2)
        portfolio_roi = round((net_annual_profit / spent * 100), 1) if spent > 0 else 0.0
        
        baseline_churned_count = float(df['churn'].sum()) if 'churn' in df.columns else (total_subscribers * 0.25)
        new_projected_churned = max(0.0, baseline_churned_count - saved_subscribers)
        baseline_churn_rate = round(baseline_churned_count / total_subscribers * 100, 2)
        projected_churn_rate = round(new_projected_churned / total_subscribers * 100, 2)
        churn_reduction_relative_pct = round((baseline_churn_rate - projected_churn_rate) / baseline_churn_rate * 100, 1)
        
        # By playbook breakdown
        pb_breakdown = {}
        for c in allocated_candidates:
            pid = c['playbook_id']
            if pid not in pb_breakdown:
                pb_breakdown[pid] = {
                    'playbook_name': self.playbooks[pid]['name'],
                    'targeted_users': 0,
                    'cost_spent': 0.0,
                    'subscribers_saved': 0.0,
                    'mrr_saved': 0.0,
                    'annual_saved': 0.0
                }
            pb_breakdown[pid]['targeted_users'] += 1
            pb_breakdown[pid]['cost_spent'] += c['cost']
            pb_breakdown[pid]['subscribers_saved'] += c['prob_saved']
            pb_breakdown[pid]['mrr_saved'] += c['mrr_saved']
            pb_breakdown[pid]['annual_saved'] += c['annual_revenue_saved']
            
        for pid in pb_breakdown:
            pb_breakdown[pid]['cost_spent'] = round(pb_breakdown[pid]['cost_spent'], 2)
            pb_breakdown[pid]['subscribers_saved'] = round(pb_breakdown[pid]['subscribers_saved'], 1)
            pb_breakdown[pid]['mrr_saved'] = round(pb_breakdown[pid]['mrr_saved'], 2)
            pb_breakdown[pid]['annual_saved'] = round(pb_breakdown[pid]['annual_saved'], 2)
            net_p = pb_breakdown[pid]['annual_saved'] - pb_breakdown[pid]['cost_spent']
            pb_breakdown[pid]['roi_pct'] = round(net_p / pb_breakdown[pid]['cost_spent'] * 100, 1) if pb_breakdown[pid]['cost_spent'] > 0 else 0.0
            
        return {
            'budget_limit': budget,
            'budget_spent': round(spent, 2),
            'budget_utilization_pct': round((spent / budget * 100), 1) if budget > 0 else 0.0,
            'targeted_subscribers_count': n_targeted,
            'projected_saved_subscribers': saved_subscribers,
            'baseline_churn_rate_pct': baseline_churn_rate,
            'projected_churn_rate_pct': projected_churn_rate,
            'churn_reduction_pct': churn_reduction_relative_pct,
            'monthly_mrr_saved': saved_mrr,
            'annual_revenue_saved': saved_annual,
            'net_annual_profit': net_annual_profit,
            'campaign_roi_pct': portfolio_roi,
            'playbook_breakdown': list(pb_breakdown.values())
        }
