import os
import pandas as pd

INPUT_XLSX = "data/online_retail_II.xlsx"
SHEET_1 = "Year 2009-2010"
SHEET_2 = "Year 2010-2011"
OUTPUT_DIR = "data"

def clean_transactions(df, keep_country="United Kingdom"):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    
    # Normalize column names
    if 'Customer ID' in df.columns:
        df.rename(columns={'Customer ID': 'CustomerID'}, inplace=True)
    if 'InvoiceNo' in df.columns:
        df.rename(columns={'InvoiceNo': 'Invoice'}, inplace=True)
    if 'Invoice Number' in df.columns:
        df.rename(columns={'Invoice Number': 'Invoice'}, inplace=True)
    
    # Using normalized names
    if 'CustomerID' in df.columns:
        df = df.dropna(subset=['CustomerID'])
        df['CustomerID'] = pd.to_numeric(df['CustomerID'], errors='coerce').dropna().astype(int)
    if 'Invoice' in df.columns:
        df = df[~df['Invoice'].astype(str).str.startswith('C', na=False)]
    if 'Quantity' in df.columns:
        df = df[df['Quantity'] > 0]
    if 'InvoiceDate' in df.columns:
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    if 'Quantity' in df.columns and 'Price' in df.columns:
        df['TotalPrice'] = df['Quantity'] * df['Price']
    if 'Country' in df.columns and keep_country:
        df = df[df['Country'] == keep_country]
    
    keep_cols = ['Invoice', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'Price', 'CustomerID', 'Country', 'TotalPrice']
    existing = [c for c in keep_cols if c in df.columns]
    return df[existing]

def main():
    xlsx_path = os.path.join("data", "online_retail_II.xlsx")
    xls = pd.ExcelFile(xlsx_path)
    df1 = pd.read_excel(xlsx_path, sheet_name=SHEET_1)
    df2 = pd.read_excel(xlsx_path, sheet_name=SHEET_2)
    clean1 = clean_transactions(df1)
    clean2 = clean_transactions(df2)
    os.makedirs("data", exist_ok=True)
    clean1.to_csv(os.path.join("data","cleaned_2009_2010.csv"), index=False)
    clean2.to_csv(os.path.join("data","cleaned_2010_2011.csv"), index=False)
    print("✅ Saved cleaned CSVs to data/")

if __name__ == "__main__":
    main()
