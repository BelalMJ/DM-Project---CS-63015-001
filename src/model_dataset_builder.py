import os, argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def build_dataset(rfm_csv, labels_csv, output_dir, test_size=0.3, random_state=42):
    os.makedirs(output_dir, exist_ok=True)
    rfm = pd.read_csv(rfm_csv)
    labels = pd.read_csv(labels_csv)
    df = pd.merge(rfm, labels, on='CustomerID', how='left')
    df['WillPurchaseNext'] = df['WillPurchaseNext'].fillna(0).astype(int)
    df['Monetary_log'] = (df['Monetary'] + 1).apply(lambda x: np.log(x))
    features = ['Recency', 'Frequency', 'Monetary_log']
    X = df[features]
    y = df['WillPurchaseNext']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    df.to_csv(os.path.join(output_dir, "model_dataset_full.csv"), index=False)
    print("✅ Saved dataset files to", output_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rfm_csv")
    parser.add_argument("labels_csv")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    build_dataset(args.rfm_csv, args.labels_csv, args.output_dir)

if __name__ == "__main__":
    main()
