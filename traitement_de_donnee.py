# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 21:40:17 2026

@author: grabe
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

BASE_PATH = "data_base"



START_DATE = "2020-10-22"

all_dates = pd.date_range(
    start=START_DATE,
    end=pd.Timestamp.today(),
    freq="D"
)


print("📂 Loading all data...\n")


# ─── DATAFRAME ────────────────────────────────────────────────────────────────────  

def finalize_dataframe(df, min_period=71):
    df = df.iloc[min_period:]
    df = df.reset_index().rename(columns={"index": "Date"})
    return df

def prepare_dates(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    df = df.reindex(all_dates)
    df["is_trading_day"] = (~df["Close"].isna()).astype(int)
    return df

def fill_data(df):
    price_cols = ["Open", "High", "Low", "Close"]
    
    indicator_cols = [
        "MA20", "MA50", "Momentum", "RSI",
        "MACD", "MACD_Signal", "MACD_Hist",
        "Volatility_20d",
        "BB_Upper", "BB_Lower", "BB_Mid",
        "Return_1d","Return_5d",
        "Normalized","VIX_Close"
    ]

    df[price_cols] = df[price_cols].ffill()
    df["Volume"] = df["Volume"].fillna(0)

    existing = [col for col in indicator_cols if col in df.columns]
    df[existing] = df[existing].ffill().bfill()

    return df

def clean_dataframe(df):
    df = prepare_dates(df)
    df = fill_data(df)
    df = finalize_dataframe(df)
    return df

# ─── loadding ────────────────────────────────────────────────────────────────────
def load_indices(base_path):
    liste=[]
    for file in os.listdir(base_path):
        path = os.path.join(base_path, file)
        
        if not file.endswith(".csv"):
            continue
        
        df = pd.read_csv(path)
        
        df=clean_dataframe(df)

        df["indice"] = file.replace(".csv", "")
        liste.append(df)
    return liste

def load_actions(base_path):
    liste=[]

    for market in os.listdir(base_path):
        market_path = os.path.join(base_path, market)

        if not os.path.isdir(market_path):
            continue

        for sector in os.listdir(market_path):
            sector_path = os.path.join(market_path, sector)

            if not os.path.isdir(sector_path) or sector.lower() == "unknown":
                continue

            for file in os.listdir(sector_path):

                if not file.endswith(".csv") :
                    continue

                file_path = os.path.join(sector_path, file)

                try:
                    df = pd.read_csv(file_path)
                    
                    df = clean_dataframe(df)
                    df["Ticker"] = file.replace(".csv", "")
                    df["Market"] = market
                    df["Sector"] = sector
                    cols_to_check = ["Open", "High"]
                    if df[cols_to_check].isna().any().any(): 
                        continue 
                    liste.append(df)

                except Exception as e:
                    print(f"❌ Error {file}: {e}")

    return liste
# ─── MAIN ────────────────────────────────────────────────────────────────────
def dataframe_info(df):
    print(df.head())
    print(df.info())
    print(df.select_dtypes(include=['object', 'string', 'int', 'float', 'bool']).isnull().sum())

def label_encoder(df,colonne,labelencoder):
    df[colonne]= labelencoder.fit_transform(df[colonne])

def clean_all_dataframe(fill_actions,fill_sectors,fill_bourses):
    
    fill_actions["Market"] = LabelEncoder().fit_transform(fill_actions["Market"])
    
    fill_actions["Sector"] = LabelEncoder().fit_transform(fill_actions["Sector"])
    fill_actions['Ticker'] = LabelEncoder().fit_transform(fill_actions['Ticker'])
    fill_sectors['indice'] = LabelEncoder().fit_transform(fill_sectors['indice'])
    fill_bourses['indice'] = LabelEncoder().fit_transform(fill_bourses['indice'])
    return fill_actions,fill_sectors,fill_bourses

def fusion(liste):
    df=pd.concat(liste, ignore_index=True)
    return df

def load_all():
    sectors = load_indices(f"{BASE_PATH}/sector_indices")
    bourses=load_indices(f"{BASE_PATH}/bourse_indices")
    # actions
    actions = load_actions(BASE_PATH)
    print(f"\n✅ Loaded {len(actions)+len(sectors)+len(bourses)} files")
    return actions,sectors,bourses

def concat_dataframe():
    
    actions,sectors,bourses=load_all()
    
    # ── Fusion ──
    fill_sectors= fusion(sectors)
    fill_action = fusion(actions)
    fill_bourse = fusion(bourses)
    return fill_action,fill_sectors,fill_bourse

def prepare_external_df(df, key_name, suffix):
    
    df = df.rename(columns={"indice": key_name})

    cols_to_rename = {
        col: f"{col}_{suffix}"
        for col in df.columns
        if col not in ["Date", key_name,"is_trading_day"]
    }

    df = df.rename(columns=cols_to_rename)

    return df

def merge_dataset(left_df, right_df, keys):
    
    return left_df.merge(
        right_df,
        on=keys,
        how="left"
    )
    
def merge_all_datasets(actions, sectors, bourses):

    sectors = prepare_external_df(
        sectors,
        key_name="Sector",
        suffix="sector"
    )

    bourses = prepare_external_df(
        bourses,
        key_name="Market",
        suffix="bourse"
    )

    df = merge_dataset(
        actions,
        sectors.drop(columns=["is_trading_day"]),
        ["Date", "Sector"]
    )

    df = merge_dataset(
        df,
        bourses.drop(columns=["is_trading_day"]),
        ["Date", "Market"]
    )

    return df

def prepare_datasets():
    indices = load_indices(os.path.join(BASE_PATH, "sector_indices"))
    bourses = load_indices(os.path.join(BASE_PATH, "bourse_indices"))
    actions = load_actions(BASE_PATH)

    fill_indice = pd.concat(indices, ignore_index=True)
    fill_bourse = pd.concat(bourses, ignore_index=True)
    fill_df = pd.concat(actions, ignore_index=True)

    fill_df = pd.get_dummies(fill_df, columns=["Market", "Sector"])

    ticker_encoder = LabelEncoder()
    indice_encoder = LabelEncoder()
    bourse_encoder = LabelEncoder()

    fill_df["Ticker"] = ticker_encoder.fit_transform(fill_df["Ticker"])
    fill_indice["indice"] = indice_encoder.fit_transform(fill_indice["indice"])
    fill_bourse["indice"] = bourse_encoder.fit_transform(fill_bourse["indice"])

    return fill_indice, fill_df, fill_bourse, ticker_encoder

def main():  
    
    fill_actions,fill_sectors,fill_bourses = concat_dataframe()
    
    fill_actions,fill_sectors,fill_bourses= clean_all_dataframe(fill_actions,fill_sectors,fill_bourses)
    
    dataframe_info(fill_actions)
    dataframe_info(fill_bourses)
    dataframe_info(fill_sectors)
    
    final_df = merge_all_datasets(
    fill_actions,
    fill_sectors,
    fill_bourses
    )
    dataframe_info(final_df)
    
    return fill_actions,fill_sectors,fill_bourses,final_df

if __name__ == "__main__":
    fill_actions,fill_sectors,fill_bourses, final_df=main()
