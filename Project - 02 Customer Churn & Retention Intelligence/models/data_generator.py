"""
Netflix Subscriber Telemetry & Behavioral Data Generator
Generates realistic, statistically correlated subscriber behavior, engagement metrics,
technical friction logs, billing telemetry, and churn events for 10,000 customers.
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_netflix_churn_data(output_dir: str = None, n_samples: int = 10000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates synthetic but highly realistic Netflix subscriber telemetry dataset.
    """
    np.random.seed(random_seed)
    
    # 1. Identifiers & Demographics
    customer_ids = [f"NFLX-{10000 + i}" for i in range(n_samples)]
    
    # Subscription Tiers & Pricing
    tiers = ['Standard with Ads', 'Standard', 'Premium']
    tier_probs = [0.28, 0.44, 0.28]
    tier_choices = np.random.choice(tiers, size=n_samples, p=tier_probs)
    
    tier_prices = {
        'Standard with Ads': 6.99,
        'Standard': 15.49,
        'Premium': 22.99
    }
    monthly_fees = np.array([tier_prices[t] for t in tier_choices])
    
    # Tenure in months (Gamma distribution: skewed toward 1-12 months, long tail up to 48)
    tenure_raw = np.random.gamma(shape=2.2, scale=7.0, size=n_samples)
    tenure_months = np.clip(np.round(tenure_raw), 1, 48).astype(int)
    
    # Billing cycle & payment methods
    billing_frequencies = np.random.choice(['Monthly', 'Annual'], size=n_samples, p=[0.88, 0.12])
    payment_methods = np.random.choice(
        ['Credit Card', 'PayPal', 'Debit Card', 'Direct Carrier Billing', 'Gift Card'],
        size=n_samples,
        p=[0.58, 0.22, 0.12, 0.05, 0.03]
    )
    
    # Profiles & Devices
    num_profiles = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.25, 0.35, 0.22, 0.12, 0.06])
    # Premium tiers tend to have more profiles
    num_profiles = np.where(tier_choices == 'Premium', np.clip(num_profiles + 1, 1, 5), num_profiles)
    
    primary_devices = np.random.choice(
        ['Smart TV', 'Mobile', 'Laptop', 'Tablet', 'Streaming Stick'],
        size=n_samples,
        p=[0.46, 0.24, 0.16, 0.08, 0.06]
    )
    
    # 2. Historical & Current Engagement Telemetry
    # Baseline weekly watch hours
    baseline_hours = np.random.gamma(shape=3.5, scale=4.0, size=n_samples)
    weekly_watch_hours = np.clip(baseline_hours, 0.5, 50.0)
    
    # Previous 30-day consumption
    watch_hours_prev_30d = np.clip(weekly_watch_hours * 4.2 + np.random.normal(0, 5, n_samples), 1.0, 220.0)
    
    # Behavioral shifts / Usage trend:
    # Some customers experience an engagement crash, some are stable, some growing
    trend_type = np.random.choice(['crashing', 'declining', 'stable', 'growing'], size=n_samples, p=[0.14, 0.22, 0.48, 0.16])
    velocity_multipliers = np.zeros(n_samples)
    for i, t in enumerate(trend_type):
        if t == 'crashing':
            velocity_multipliers[i] = np.random.uniform(0.05, 0.35)
        elif t == 'declining':
            velocity_multipliers[i] = np.random.uniform(0.40, 0.75)
        elif t == 'stable':
            velocity_multipliers[i] = np.random.uniform(0.85, 1.15)
        else:
            velocity_multipliers[i] = np.random.uniform(1.20, 1.80)
            
    watch_hours_last_30d = np.clip(watch_hours_prev_30d * velocity_multipliers + np.random.normal(0, 3, n_samples), 0.0, 240.0)
    usage_velocity = np.round(watch_hours_last_30d / (watch_hours_prev_30d + 1e-4), 3)
    
    # Days since last watch (recency)
    days_since_last_watch = np.zeros(n_samples)
    for i, t in enumerate(trend_type):
        if t == 'crashing':
            days_since_last_watch[i] = np.random.randint(14, 55)
        elif t == 'declining':
            days_since_last_watch[i] = np.random.randint(5, 25)
        elif t == 'stable':
            days_since_last_watch[i] = np.random.randint(0, 7)
        else:
            days_since_last_watch[i] = np.random.randint(0, 4)
    days_since_last_watch = days_since_last_watch.astype(int)
    
    # Content Completion Rate (percentage of titles watched to at least 80% credits)
    # Dissatisfied / disengaged users abandon titles early
    completion_rate_base = np.random.beta(a=5, b=2.5, size=n_samples)
    completion_rate = np.clip(
        completion_rate_base * np.where(trend_type == 'crashing', 0.45, np.where(trend_type == 'declining', 0.75, 1.0)),
        0.12, 0.98
    )
    completion_rate = np.round(completion_rate, 3)
    
    # Titles started in last 30 days
    titles_started_last_30d = np.clip(
        np.round(watch_hours_last_30d / np.random.uniform(2.5, 5.5, n_samples)).astype(int),
        0, 50
    )
    
    # Search to watch ratio (Browsing paralysis / search fatigue)
    # Higher search ratio means user spends 25 minutes searching and giving up
    search_to_watch_ratio = np.random.gamma(shape=2.0, scale=0.8, size=n_samples) + 0.3
    # Crashing/declining users have higher search abandonment
    search_to_watch_ratio = np.where(trend_type == 'crashing', search_to_watch_ratio * 1.8, search_to_watch_ratio)
    search_to_watch_ratio = np.round(np.clip(search_to_watch_ratio, 0.2, 6.0), 2)
    
    # Genre diversity (Shannon-style entropy proxy: 0.1 = mono-show addict, 0.95 = omnivore)
    genre_diversity = np.random.beta(a=3.5, b=2.5, size=n_samples)
    genre_diversity = np.round(np.clip(genre_diversity, 0.10, 0.96), 2)
    
    # 3. Technical Friction & Quality of Service (QoS)
    # Buffering incidents per month (Poisson with occasional severe spikes)
    buffering_base = np.random.poisson(lam=1.8, size=n_samples)
    tech_glitch_cohort = np.random.choice([0, 1], size=n_samples, p=[0.90, 0.10])
    buffering_incidents_per_month = np.clip(buffering_base + tech_glitch_cohort * np.random.randint(6, 18, size=n_samples), 0, 25)
    
    # Customer support tickets in last 60 days
    ticket_probs = [0.65, 0.22, 0.08, 0.03, 0.015, 0.005]
    support_tickets = np.random.choice([0, 1, 2, 3, 4, 5], size=n_samples, p=ticket_probs)
    # Increase tickets for tech glitch cohort
    support_tickets = np.clip(support_tickets + tech_glitch_cohort * np.random.choice([1, 2], size=n_samples), 0, 6)
    
    # 4. Economic, Price Sensitivity & Policy Catalysts
    # Price hike notification received in last 45 days
    price_increase_notified = np.random.choice([0, 1], size=n_samples, p=[0.60, 0.40])
    
    # Shared account crackdown / paid sharing warning prompt
    shared_account_warning = np.random.choice([0, 1], size=n_samples, p=[0.75, 0.25])
    
    # Involuntary billing friction (card retry failures)
    payment_retry_count = np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.82, 0.10, 0.05, 0.02, 0.01])
    
    # Offline downloads feature usage
    offline_downloads_used = np.random.choice([0, 1], size=n_samples, p=[0.62, 0.38])
    
    # Average session duration in minutes
    avg_session_mins = np.clip(np.random.normal(68, 25, size=n_samples), 12.0, 210.0)
    avg_session_mins = np.round(avg_session_mins, 1)
    
    # 5. Composite Churn Propensity & Deterministic Causal Ground Truth
    # Calculate Latent Churn Propensity Z-score
    # Features positively driving churn:
    # - Low usage velocity (drop in watch hours)
    # - High days since last watch
    # - Low completion rate
    # - High search to watch ratio (discovery paralysis)
    # - High buffering & support tickets (tech friction)
    # - Price hike notification coupled with low perceived value (hours per dollar)
    # - Payment retry count (involuntary churn)
    # - Low genre diversity after finishing a series (binge & burn)
    
    value_per_dollar = (watch_hours_last_30d + 1.0) / monthly_fees
    
    churn_score_latent = (
        - 2.80 * np.log(np.clip(usage_velocity, 0.05, 3.0))
        + 0.085 * (days_since_last_watch - 6)
        - 3.20 * (completion_rate - 0.65)
        + 0.55 * (search_to_watch_ratio - 1.5)
        + 0.22 * buffering_incidents_per_month
        + 0.45 * support_tickets
        + 0.65 * price_increase_notified * (1.0 / np.clip(value_per_dollar, 0.2, 5.0))
        + 0.35 * shared_account_warning
        + 0.95 * payment_retry_count
        - 1.40 * (genre_diversity - 0.50)
        - 0.045 * (tenure_months - 12)
        - 0.85 * (billing_frequencies == 'Annual').astype(float)
        - 4.65 # calibrated intercept for ~22.5% realistic streaming cohort churn
    )
    
    # Convert to probability via sigmoid
    churn_prob_latent = 1.0 / (1.0 + np.exp(-churn_score_latent))
    churn_prob_calibrated = np.clip(churn_prob_latent, 0.01, 0.99)
    churn = (np.random.rand(n_samples) < churn_prob_calibrated).astype(int)
    
    # Assign Deterministic Root Cause for Churned Users (and 'Active / Retained' for non-churned)
    churn_reasons = []
    for i in range(n_samples):
        if churn[i] == 0:
            churn_reasons.append('Active / Retained')
        else:
            # Evaluate dominant symptom
            if payment_retry_count[i] >= 2:
                churn_reasons.append('Involuntary Payment Failure')
            elif buffering_incidents_per_month[i] >= 7 or support_tickets[i] >= 3:
                churn_reasons.append('Technical Buffering & Playback Issues')
            elif price_increase_notified[i] == 1 and value_per_dollar[i] < 2.0:
                churn_reasons.append('Price Hike Reaction & Value Misalignment')
            elif genre_diversity[i] < 0.48 and usage_velocity[i] < 0.45 and watch_hours_prev_30d[i] > 18:
                churn_reasons.append('Finished Specific Series (Binge & Burn)')
            elif search_to_watch_ratio[i] >= 3.0 or completion_rate[i] < 0.35:
                churn_reasons.append('Discovery Fatigue (Too Hard to Find Content)')
            else:
                churn_reasons.append('Content Stagnation & Boredom')
                
    # Build DataFrame
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'subscription_tier': tier_choices,
        'monthly_fee': monthly_fees,
        'tenure_months': tenure_months,
        'billing_frequency': billing_frequencies,
        'payment_method': payment_methods,
        'payment_retry_count': payment_retry_count,
        'num_profiles': num_profiles,
        'primary_device': primary_devices,
        'weekly_watch_hours': np.round(weekly_watch_hours, 1),
        'watch_hours_last_30d': np.round(watch_hours_last_30d, 1),
        'watch_hours_prev_30d': np.round(watch_hours_prev_30d, 1),
        'usage_velocity': usage_velocity,
        'days_since_last_watch': days_since_last_watch,
        'completion_rate': completion_rate,
        'titles_started_last_30d': titles_started_last_30d,
        'search_to_watch_ratio': search_to_watch_ratio,
        'genre_diversity_score': genre_diversity,
        'buffering_incidents_per_month': buffering_incidents_per_month,
        'customer_support_tickets': support_tickets,
        'price_increase_notified': price_increase_notified,
        'shared_account_warning': shared_account_warning,
        'offline_downloads_used': offline_downloads_used,
        'avg_session_mins': avg_session_mins,
        'value_per_dollar': np.round(value_per_dollar, 2),
        'churn': churn,
        'primary_churn_reason': churn_reasons
    })
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, 'netflix_subscribers.csv')
        df.to_csv(csv_path, index=False)
        print(f"Dataset successfully generated and saved to {csv_path}")
        print(f"Total Subscribers: {len(df):,}, Churned: {df['churn'].sum():,} ({df['churn'].mean()*100:.2f}%)")
        
    return df

if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    generate_netflix_churn_data(data_dir)
