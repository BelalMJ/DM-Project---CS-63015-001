import numpy as np
import pandas as pd

def lift_curve(y_true, y_proba, n_bins=10):
    df = pd.DataFrame({'y_true': y_true, 'y_proba': y_proba})
    df = df.sort_values('y_proba', ascending=False).reset_index(drop=True)
    df['decile'] = pd.qcut(df.index, q=n_bins, labels=False, duplicates='drop')
    agg = df.groupby('decile').agg(
        n=('y_true','count'),
        positives=('y_true','sum'),
        avg_proba=('y_proba','mean')
    ).reset_index()
    total_rate = df['y_true'].sum() / len(df)
    agg['response_rate'] = agg['positives'] / agg['n']
    agg['lift'] = agg['response_rate'] / total_rate
    return agg.sort_values('decile')

def topk_capture_rate(y_true, y_proba, k=0.1):
    df = pd.DataFrame({'y_true': y_true, 'y_proba': y_proba})
    df = df.sort_values('y_proba', ascending=False)
    cutoff = max(1, int(len(df) * k))  # Ensure at least 1 sample
    top = df.head(cutoff)
    if df['y_true'].sum() == 0:
        return 0.0
    return top['y_true'].sum() / df['y_true'].sum()

def gains_table(y_true, y_proba, n_bins=10):
    agg = lift_curve(y_true, y_proba, n_bins=n_bins)
    agg['cumulative_positives'] = agg['positives'].cumsum()
    agg['cumulative_percent_positives'] = agg['cumulative_positives'] / agg['positives'].sum()
    agg['cumulative_percent_population'] = ((agg['decile']+1) * (1.0/n_bins)).values
    return agg
