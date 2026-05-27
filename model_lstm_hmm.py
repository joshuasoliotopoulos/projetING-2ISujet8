# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
from scipy import stats

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from hmmlearn.hmm import GaussianHMM

from traitement_de_donnee import prepare_datasets

# =========================
# FINANCIAL METRICS
# =========================

def sharpe_ratio(returns, rf=0.0):
    returns = np.array(returns)
    if returns.std() == 0:
        return 0
    return (returns.mean() - rf) / returns.std() * np.sqrt(252)


def max_drawdown(cum_returns):
    cum_returns = np.array(cum_returns)
    cum_returns = np.where(cum_returns == 0, 1e-10, cum_returns)
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - peak) / peak
    return drawdown.min()


def hit_rate(returns):
    returns = np.array(returns)
    return np.mean(returns > 0)

# =========================
# CONFIG
# =========================

fill_indice, fill_df, fill_bourse, ticker_encoder = prepare_datasets()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQUENCE_LEN = 30
HORIZON = 5
EPOCHS = 15
BATCH_SIZE = 64

FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "MA20", "MA50", "RSI", "MACD", "Volatility_20d"
]

# =========================
# DATA CLEAN
# =========================

df_all = fill_df.copy()
df_all["Ticker"] = df_all["Ticker"].astype(str).str.upper().str.strip()


tickers = sorted(df_all["Ticker"].unique())

print(f"✅ Tickers: {len(tickers)}")
print("Exemple tickers:", tickers[:10])


# =========================
# GLOBAL STORAGE (IMPORTANT)
# =========================

all_preds = []
all_true = []

all_strategy_returns = []
all_market_returns = []


# =========================
# MODEL
# =========================

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden=128, layers=2, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0
        )

        self.fc = nn.Linear(hidden, HORIZON)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


# =========================
# SEQUENCES
# =========================

def make_sequences(data):
    X, y = [], []

    for i in range(len(data) - SEQUENCE_LEN - HORIZON):
        X.append(data[i:i+SEQUENCE_LEN])
        y.append(data[i+SEQUENCE_LEN:i+SEQUENCE_LEN+HORIZON, 3])

    return np.array(X), np.array(y)


# =========================
# HMM
# =========================

def train_hmm(df):
    try:
        close = df["Close"].ffill().bfill().fillna(0)
        returns = close.pct_change().fillna(0).values.reshape(-1, 1)

        if len(returns) < 100 or returns.std() < 1e-6:
            return None

        returns = StandardScaler().fit_transform(returns)

        hmm = GaussianHMM(
            n_components=2,
            covariance_type="diag",
            n_iter=50,
            random_state=42
        )

        hmm.fit(returns)
        return hmm

    except:
        return None


# =========================
# TRAIN ONE
# =========================

def train_one(ticker):

    global all_preds, all_true
    global all_strategy_returns, all_market_returns

    df = df_all[df_all["Ticker"] == ticker].copy().sort_values("Date")

    if len(df) < SEQUENCE_LEN + HORIZON + 10:
        return None

    feat = df[FEATURES].ffill().bfill().fillna(0)

    scaler = MinMaxScaler()
    data = scaler.fit_transform(feat.values)

    X, y = make_sequences(data)

    if len(X) < 50:
        return None

    split = int(len(X) * 0.85)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = LSTMModel(input_size=len(FEATURES)).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # =========================
    # TRAIN
    # =========================

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"{ticker} | epoch {epoch+1}/{EPOCHS} | loss={total_loss:.4f}")

    # =========================
    # GLOBAL METRICS (PER TICKER)
    # =========================

    model.eval()

    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32).to(DEVICE)).cpu().numpy()
        true = y_test
        
    # =========================
    # STRATEGY RETURNS
    # =========================
    
    strategy_returns = (preds - true) / (np.abs(true) + 1e-8)
    market_returns = np.diff(true, axis=0)
    market_returns = np.pad(market_returns, ((1,0),(0,0)), mode="constant")

    # store GLOBAL benchmark data
    all_preds.append(preds)
    all_true.append(true)
    
    all_strategy_returns.append(strategy_returns.flatten())
    all_market_returns.append(market_returns.flatten())

    rmse = np.sqrt(mean_squared_error(true, preds))
    mae = mean_absolute_error(true, preds)
    r2 = r2_score(true, preds)

    print("\n📊 METRICS TICKER")
    print(f"{ticker}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R2  : {r2:.4f}")
    
    strat_flat = strategy_returns.mean(axis=1)
    
    sharpe = sharpe_ratio(strat_flat)
    mdd    = max_drawdown(np.cumprod(1 + strat_flat))
    hit    = hit_rate(strat_flat)
    
    print("\n📊 FINANCIAL METRICS")
    print(f"Sharpe Ratio : {sharpe:.4f}")
    print(f"Max Drawdown : {mdd:.4f}")
    print(f"Hit Rate     : {hit:.4f}")

    hmm = train_hmm(df)

    return model, scaler, hmm


# =========================
# SAVE
# =========================

def save(model, scaler, hmm, ticker):

    os.makedirs("models", exist_ok=True)

    torch.save(model.state_dict(), f"models/{ticker}_lstm.pt")

    with open(f"models/{ticker}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(f"models/{ticker}_hmm.pkl", "wb") as f:
        pickle.dump(hmm, f)


# =========================
# TRAIN ALL
# =========================

if __name__ == "__main__":

    print("Training LSTM + HMM (GLOBAL METRICS)...")

    for t in tickers:

        print("\n📊", t)

        res = train_one(t)

        if res is None:
            continue

        save(*res, t)

        print("💾 Saved")

    # =========================
    # GLOBAL LIGHTGBM-STYLE METRICS
    # =========================

    all_preds = np.concatenate(all_preds, axis=0)
    all_true = np.concatenate(all_true, axis=0)

    rmse_global = np.sqrt(mean_squared_error(all_true, all_preds))
    mae_global = mean_absolute_error(all_true, all_preds)
    r2_global = r2_score(all_true, all_preds)
    
    # =========================
    # 📊 GLOBAL FINANCIAL METRICS
    # =========================
    
    all_strategy_returns = np.concatenate(all_strategy_returns)
    all_market_returns = np.concatenate(all_market_returns)
    
    sharpe_global = sharpe_ratio(all_strategy_returns)
    
    mdd_global = max_drawdown(
        np.cumprod(1 + all_strategy_returns)
    )
    
    hit_global = hit_rate(all_strategy_returns)
    
    print("\n==============================")
    print("📊 GLOBAL STRATEGY PERFORMANCE")
    print("==============================")
    print(f"Sharpe Ratio GLOBAL : {sharpe_global:.4f}")
    print(f"Max Drawdown GLOBAL : {mdd_global:.4f}")
    print(f"Hit Rate GLOBAL     : {hit_global:.4f}")
    print("==============================")

    print("\n==============================")
    print("📊 GLOBAL MODEL PERFORMANCE")
    print("==============================")
    print(f"RMSE GLOBAL: {rmse_global:.4f}")
    print(f"MAE  GLOBAL: {mae_global:.4f}")
    print(f"R2   GLOBAL: {r2_global:.4f}")
    print("==============================")

    print("✅ DONE")
