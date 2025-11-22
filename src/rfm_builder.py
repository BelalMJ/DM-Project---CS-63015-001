import os, argparse
import pandas as pd

def compute_rfm(df, ref_date=None):
    df = df.copy()
    if ref_date is None:
        ref_date = df['InvoiceDate'].max()
    else:
        ref_date = pd.to_datetime(ref_date)
    grouped = df.groupby('CustomerID').agg(
        Recency = ('InvoiceDate', lambda x: (ref_date - x.max()).days),
        Frequency = ('Invoice', pd.Series.nunique),
        Monetary = ('TotalPrice', 'sum')
    ).reset_index()
    grouped['Monetary'] = grouped['Monetary'].fillna(0.0)
    return grouped

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    parser.add_argument("--ref_date", default=None)
    args = parser.parse_args()
    df = pd.read_csv(args.input_csv, parse_dates=['InvoiceDate'])
    rfm = compute_rfm(df, ref_date=args.ref_date)
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    rfm.to_csv(args.output_csv, index=False)
    print("✅ Saved RFM to", args.output_csv)

if __name__ == "__main__":
    main()
