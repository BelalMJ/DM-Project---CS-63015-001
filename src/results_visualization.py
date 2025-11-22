import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve

def plot_model_metrics_table(results_df, outpath=None):
    fig, ax = plt.subplots(figsize=(10, 1+0.5*len(results_df)))
    sns.heatmap(results_df, annot=True, fmt=".3f", cmap="YlGnBu", cbar=False, ax=ax)
    ax.set_title("Model Performance Metrics")
    if outpath:
        os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
        fig.savefig(outpath, bbox_inches='tight')
    plt.show()

def plot_roc(models_results, y_test, outpath=None):
    plt.figure(figsize=(8,6))
    for name, res in models_results.items():
        fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
    plt.plot([0,1],[0,1],'k--', alpha=0.3)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curves'); plt.legend(loc='lower right')
    if outpath:
        os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
        plt.savefig(outpath, bbox_inches='tight')
    plt.show()

def plot_lift(agg, outpath=None):
    plt.figure(figsize=(8,5))
    plt.plot(agg['decile']+1, agg['lift'], marker='o')
    plt.xlabel('Decile (1=top)'); plt.ylabel('Lift'); plt.title('Lift by Decile')
    if outpath:
        os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
        plt.savefig(outpath, bbox_inches='tight')
    plt.show()
