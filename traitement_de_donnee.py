# -*- coding: utf-8 -*-
"""
Prétraitement des données financières
"""

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(BASE_DIR, "data_base")
START_DATE = "2021-01-01"

all_dates = pd.date_range(
    start=START_DATE,
    end=pd.Timestamp.today(),
    freq="D"
)

def clean_indice_dataframe(df):

    df.columns = df.columns.str.strip()

    if "Date" not in df.columns:
        raise ValueError("Date manquante")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date")

    df = df.reindex(all_dates)

    df["is_trading_day"] = (~df["Close"].isna()).astype(int)

    price_cols = ["Open", "High", "Low", "Close"]

    indicator_cols = [
        "MA20", "MA50", "Momentum", "RSI",
        "MACD", "MACD_Signal", "MACD_Hist",
        "Volatility_20d", "BB_Upper", "BB_Lower",
        "BB_Mid", "Return_1d", "Return_5d", "Normalized"
    ]

    for col in price_cols:
        if col in df.columns:
            df[col] = df[col].ffill()

    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].fillna(0)

    existing = [c for c in indicator_cols if c in df.columns]
    df[existing] = df[existing].ffill()

    df = df.iloc[54:].reset_index().rename(columns={"index": "Date"})
    df[existing] = df[existing].bfill()

    return df


def clean_action_dataframe(df, market, sector, ticker):

    df.columns = df.columns.str.strip()

    if "Date" not in df.columns:
        raise ValueError(f"Date manquante {ticker}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date")

    df = df.reindex(all_dates)

    df["is_trading_day"] = (~df["Close"].isna()).astype(int)

    df["Market"] = market
    df["Sector"] = sector

    # 🔥 IMPORTANT FIX ICI
    df["Ticker"] = str(ticker).strip().upper()

    price_cols = ["Open", "High", "Low", "Close"]
    volume_cols = [c for c in df.columns if "Volume" in c]

    indicator_cols = [
        "MA20", "MA50", "Momentum", "RSI",
        "MACD", "MACD_Signal", "MACD_Hist",
        "Volatility_20d", "BB_Upper", "BB_Lower",
        "BB_Mid", "VIX_Close"
    ]

    for col in price_cols:
        if col in df.columns:
            df[col] = df[col].ffill()

    df[volume_cols] = df[volume_cols].fillna(0)

    existing = [c for c in indicator_cols if c in df.columns]
    df[existing] = df[existing].ffill()

    df = df.iloc[4:].reset_index().rename(columns={"index": "Date"})
    df[existing] = df[existing].bfill()

    return df


def load_indices(path):

    data = []

    for file in os.listdir(path):

        if file.endswith(".csv"):

            df = pd.read_csv(os.path.join(path, file), sep=None, engine="python")
            df = clean_indice_dataframe(df)

            df["indice"] = file.replace(".csv", "")

            data.append(df)

    return data


def load_actions(base_path):

    data = []

    for market in os.listdir(base_path):

        market_path = os.path.join(base_path, market)
        if not os.path.isdir(market_path):
            continue

        for sector in os.listdir(market_path):

            sector_path = os.path.join(market_path, sector)
            if not os.path.isdir(sector_path):
                continue

            for file in os.listdir(sector_path):

                if file.endswith(".csv"):

                    try:
                        ticker = file.replace(".csv", "").upper()

                        df = pd.read_csv(
                            os.path.join(sector_path, file),
                            sep=None,
                            engine="python"
                        )

                        df = clean_action_dataframe(df, market, sector, ticker)

                        data.append(df)

                    except Exception as e:
                        print(f"Error {file}: {e}")

    return data

def prepare_datasets():

    indices = load_indices(os.path.join(BASE_PATH, "sector_indices"))
    bourses = load_indices(os.path.join(BASE_PATH, "bourse_indices"))
    actions = load_actions(BASE_PATH)

    fill_indice = pd.concat(indices, ignore_index=True)
    fill_bourse = pd.concat(bourses, ignore_index=True)
    fill_df = pd.concat(actions, ignore_index=True)

    fill_df["Ticker_name"] = fill_df["Ticker"].astype(str)

    ticker_encoder = LabelEncoder()
    sector_encoder = LabelEncoder()
    market_encoder = LabelEncoder()

    fill_df["Ticker"] = ticker_encoder.fit_transform(fill_df["Ticker"].astype(str))
    fill_df["Sector"] = sector_encoder.fit_transform(fill_df["Sector"].astype(str))
    fill_df["Market"] = market_encoder.fit_transform(fill_df["Market"].astype(str))

    return fill_indice, fill_df, fill_bourse, ticker_encoder
