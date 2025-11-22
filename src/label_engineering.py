import pandas as pd
import argparse
import os

def create_labels(period_a_csv, period_b_csv, output_csv):
    df_a = pd.read_csv(period_a_csv)
    df_b = pd.read_csv(period_b_csv)
    
    customers_a = df_a['CustomerID'].unique()
    customers_b = df_b['CustomerID'].unique()
    
    labels = pd.DataFrame({'CustomerID': customers_a})
    labels['WillPurchaseNext'] = labels['CustomerID'].isin(customers_b).astype(int)
    
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    labels.to_csv(output_csv, index=False)
    print("✅ Saved labels to", output_csv)

def main():  # ADD THIS MISSING FUNCTION
    parser = argparse.ArgumentParser()
    parser.add_argument("period_a_csv")
    parser.add_argument("period_b_csv")
    parser.add_argument("output_csv")
    args = parser.parse_args()
    create_labels(args.period_a_csv, args.period_b_csv, args.output_csv)

if __name__ == "__main__":
    main()
