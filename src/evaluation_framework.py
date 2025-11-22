import os, pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, precision_recall_curve, auc)

class ModelEvaluator:
    def __init__(self, models, X_test, y_test, feature_names=None):
        self.models = models
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names or X_test.columns.tolist()
        self.results = {}

    def evaluate_single_model(self, model, model_name):
        y_pred = model.predict(self.X_test)
        try:
            y_proba = model.predict_proba(self.X_test)[:, 1]
        except Exception:
            if hasattr(model, "decision_function"):
                y_proba = model.decision_function(self.X_test)
            else:
                y_proba = y_pred
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1': f1_score(self.y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(self.y_test, y_proba) if len(np.unique(self.y_test))>1 else np.nan,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        self.results[model_name] = metrics
        return metrics

    def compare_models(self):
        rows = {}
        for name, model in self.models.items():
            m = self.evaluate_single_model(model, name)
            rows[name] = {k: v for k, v in m.items() if k in ['accuracy','precision','recall','f1','roc_auc']}
        return pd.DataFrame(rows).T

    def plot_roc_curves(self, outpath=None):
        plt.figure(figsize=(8,6))
        for name, res in self.results.items():
            y_proba = res['y_proba']
            fpr, tpr, _ = roc_curve(self.y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
        plt.plot([0,1],[0,1],'k--', alpha=0.3)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc='lower right')
        if outpath:
            os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
            plt.savefig(outpath, bbox_inches='tight')
        plt.show()

    def plot_pr_curves(self, outpath=None):
        plt.figure(figsize=(8,6))
        for name, res in self.results.items():
            y_proba = res['y_proba']
            precision, recall, _ = precision_recall_curve(self.y_test, y_proba)
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, label=f"{name} (AUC = {pr_auc:.3f})")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc='lower left')
        if outpath:
            os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
            plt.savefig(outpath, bbox_inches='tight')
        plt.show()

    def save_confusion_matrices(self, outdir="output/plots"):
        os.makedirs(outdir, exist_ok=True)
        for name, res in self.results.items():
            cm = confusion_matrix(self.y_test, res['y_pred'])
            fig, ax = plt.subplots(figsize=(4,3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
            ax.set_title(f'Confusion Matrix: {name}')
            fig.savefig(os.path.join(outdir, f"confusion_{name}.png"), bbox_inches='tight')
            plt.close(fig)

    def feature_importance(self, model_name, outpath=None):
        model = self.models[model_name]
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).ravel()
        else:
            raise ValueError("Model has no feature_importances_ or coef_")
        fi = pd.DataFrame({'feature': self.feature_names, 'importance': importances})
        fi = fi.sort_values('importance', ascending=False)
        if outpath:
            os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
            fi.to_csv(outpath, index=False)
        return fi
