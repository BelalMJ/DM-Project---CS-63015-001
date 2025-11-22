import os, argparse, joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def load_data(X_path, y_path):
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path).squeeze()
    return X, y

def train_and_save(X_train, y_train, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    models = {}

    # Logistic Regression
    pipe_lr = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(max_iter=500, random_state=42))])
    pipe_lr.fit(X_train, y_train)
    models['logistic'] = pipe_lr
    joblib.dump(pipe_lr, os.path.join(output_dir, 'logistic.pkl'))

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models['random_forest'] = rf
    joblib.dump(rf, os.path.join(output_dir, 'random_forest.pkl'))

    # XGBoost - FIXED: Added missing comma
    xgb = XGBClassifier(
        n_estimators=200,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False,  # ADDED COMMA HERE
        subsample=1.0,          
        colsample_bytree=1.0    
    )
    xgb.fit(X_train, y_train)
    models['xgboost'] = xgb
    joblib.dump(xgb, os.path.join(output_dir, 'xgboost.pkl'))

    print("✅ Trained and saved models:", list(models.keys()))
    return models

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("X_train")
    parser.add_argument("y_train")
    parser.add_argument("output_dir", default="output/models")
    args = parser.parse_args()
    X_train, y_train = load_data(args.X_train, args.y_train)
    train_and_save(X_train, y_train, args.output_dir)

if __name__ == "__main__":
    main()
