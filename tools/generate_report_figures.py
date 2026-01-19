#!/usr/bin/env python3
"""
Generate report figures from the model outputs saved in output_full_t10/.
Creates:
 - confusion_matrix_counts.png
 - confusion_matrix_normalized.png
 - label_distribution.png
 - pred_vs_actual.png (A)
 - residuals_hist.png (B)
 - mean_by_timewindow.png (E)

Usage: python3 tools/generate_report_figures.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

OUT_DIR = os.path.join(os.getcwd(), 'output_full_t10')
CSV_PATH = os.path.join(OUT_DIR, 'test_predictions.csv')

if not os.path.exists(CSV_PATH):
    print('Missing CSV:', CSV_PATH)
    sys.exit(1)

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
# normalize column names
df.columns = [c.strip() for c in df.columns]

# required columns check
if 'prediction' not in df.columns or 'ArrDelay' not in df.columns:
    print('CSV missing numeric columns: prediction or ArrDelay')
    sys.exit(1)

# Confusion matrices and label distribution
if 'predicted_label' in df.columns and 'actual_label' in df.columns:
    cm = pd.crosstab(df['actual_label'], df['predicted_label'], dropna=False)
    cm_norm = pd.crosstab(df['actual_label'], df['predicted_label'], normalize='index').fillna(0)

    plt.figure(figsize=(6,4))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues')
    plt.title('Confusion matrix (normalized by true label)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    fn1 = os.path.join(OUT_DIR, 'confusion_matrix_normalized.png')
    plt.savefig(fn1, dpi=150)
    plt.close()

    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
    plt.title('Confusion matrix (counts)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    fn2 = os.path.join(OUT_DIR, 'confusion_matrix_counts.png')
    plt.savefig(fn2, dpi=150)
    plt.close()

    plt.figure(figsize=(6,3))
    order = sorted(df['actual_label'].unique())
    sns.countplot(x='actual_label', data=df, order=order)
    plt.title('Actual label distribution')
    plt.tight_layout()
    fn3 = os.path.join(OUT_DIR, 'label_distribution.png')
    plt.savefig(fn3, dpi=150)
    plt.close()
    print('Saved:', fn1, fn2, fn3)
else:
    print('predicted_label/actual_label columns not found; skipping confusion/label plots')

# A: Predicted vs Actual scatter
try:
    y = df['ArrDelay'].astype(float).to_numpy()
    yhat = df['prediction'].astype(float).to_numpy()
    mae = np.mean(np.abs(yhat - y))
    rmse = np.sqrt(np.mean((yhat - y)**2))
except Exception as e:
    print('Failed to compute numeric metrics:', e)
    sys.exit(1)

plt.figure(figsize=(6,6))
plt.scatter(y, yhat, alpha=0.8)
lims = [min(y.min(), yhat.min()), max(y.max(), yhat.max())]
plt.plot(lims, lims, 'k--', linewidth=1)
plt.xlabel('Actual ArrDelay')
plt.ylabel('Predicted')
plt.title(f'Predicted vs Actual (MAE={mae:.2f}, RMSE={rmse:.2f})')
plt.tight_layout()
fn4 = os.path.join(OUT_DIR, 'pred_vs_actual.png')
plt.savefig(fn4, dpi=150)
plt.close()
print('Saved:', fn4)

# B: Residual histogram + KDE
res = yhat - y
plt.figure(figsize=(6,4))
try:
    sns.histplot(res, kde=True, bins=30)
except Exception:
    plt.hist(res, bins=30)
plt.axvline(res.mean(), color='red', linestyle='--', label=f'mean={res.mean():.2f}')
plt.axvline(np.median(res), color='orange', linestyle=':', label=f'median={np.median(res):.2f}')
plt.legend()
plt.title('Residuals (prediction - actual)')
plt.tight_layout()
fn5 = os.path.join(OUT_DIR, 'residuals_hist.png')
plt.savefig(fn5, dpi=150)
plt.close()
print('Saved:', fn5)

# E: Mean actual vs predicted by time window
agg = None
if 'DepTime_TOD' in df.columns and df['DepTime_TOD'].notnull().any():
    group_col = 'DepTime_TOD'
    agg = df.groupby(group_col).agg(actual=('ArrDelay', 'mean'), pred=('prediction', 'mean')).reset_index()
    order = sorted(agg[group_col].unique(), key=lambda x: str(x))
    agg = agg.set_index(group_col).reindex(order).reset_index()
else:
    # fallback: extract hour from DepTime
    def extract_hour(x):
        try:
            xi = int(x)
            return xi//100
        except Exception:
            return np.nan
    if 'DepTime' in df.columns:
        df['DepHour'] = df['DepTime'].apply(extract_hour)
        group_col = 'DepHour'
        agg = df.dropna(subset=[group_col]).groupby(group_col).agg(actual=('ArrDelay','mean'), pred=('prediction','mean')).reset_index()
        agg = agg.sort_values(group_col)

if agg is not None and len(agg)>0:
    plt.figure(figsize=(8,4))
    sns.lineplot(x=agg[group_col], y=agg['actual'], label='Actual', marker='o')
    sns.lineplot(x=agg[group_col], y=agg['pred'], label='Predicted', marker='o')
    plt.title('Mean Actual vs Predicted by Time Window')
    plt.xlabel(group_col)
    plt.legend()
    plt.tight_layout()
    fn6 = os.path.join(OUT_DIR, 'mean_by_timewindow.png')
    plt.savefig(fn6, dpi=150)
    plt.close()
    print('Saved:', fn6)
else:
    print('No suitable time-of-day information; skipped mean-by-timewindow plot')

print('All done')
