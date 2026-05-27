import os
import pickle
import torch
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from model_lstm_hmm import FEATURES, DEVICE, SEQUENCE_LEN, HORIZON, LSTMModel, df_all

MODEL_DIR = "./models"

st.set_page_config(page_title="Dashboard Actions", layout="wide")
st.title("📈 Dashboard Actions — LSTM + HMM")

# ----- Sidebar ----------------------------------------------------

with st.sidebar:
    st.header("⚙️ Paramètres")
    tickers = sorted(df_all["Ticker_name"].unique())
    ticker = st.selectbox("Choisir une action", tickers)
    st.caption(f"Données disponibles jusqu'au {pd.to_datetime(df_all['Date'].max()).strftime('%d/%m/%Y')}")

# ----- LOADERS ---------------------------------------------------

def load_model(ticker: str):
    path = os.path.join(MODEL_DIR, f"{ticker}_lstm.pt")
    model = LSTMModel(input_size=len(FEATURES)).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model

def load_scaler(ticker: str):
    path = os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

def load_hmm(ticker: str):
    path = os.path.join(MODEL_DIR, f"{ticker}_hmm.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

# ----- PREDICT ------------------------------------------------

def predict(ticker: str):
    model  = load_model(ticker)
    scaler = load_scaler(ticker)
    hmm    = load_hmm(ticker)

    if model is None or scaler is None:
        return None
    
    data = df_all[df_all["Ticker_name"] == ticker].copy().sort_values("Date")

    if len(data) < SEQUENCE_LEN:
        return None

    missing = [f for f in FEATURES if f not in data.columns]
    if missing:
        st.error(f"Features manquantes : {missing}")
        return None

    feat = (
        data[FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .bfill()
        .fillna(0)
    )

    scaled = scaler.transform(feat.values)
    seq    = scaled[-SEQUENCE_LEN:]
    X      = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        forecast = model(X).cpu().numpy()[0]

    close_min = scaler.data_min_[3]
    close_max = scaler.data_max_[3]
    forecast  = forecast * (close_max - close_min) + close_min
    forecast  = np.clip(forecast, 0, None)

    current = float(data["Close"].iloc[-1])

    # HMM regime
    regime = None
    if hmm is not None and len(data) > 50:
        try:
            returns = data["Close"].pct_change().fillna(0).values.reshape(-1, 1)
            regime  = int(hmm.predict(returns)[-1])
        except Exception:
            regime = None

    return current, forecast, data, regime

# ------ RUN -----------------------------------------------------

with st.spinner("Chargement..."):
    res = predict(ticker)

if res is None:
    st.warning("Modèle ou données insuffisantes pour ce ticker.")
    st.stop()

current, forecast, hist, regime = res

variation      = (forecast[0] - current) / current * 100
last_data_date = pd.to_datetime(hist["Date"].iloc[-1]).strftime("%d/%m/%Y")

# ----- MÉTRIQUES -----------------------------------------------

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Close actuel",    f"{current:.2f}")
c2.metric("J+1 prédit",      f"{forecast[0]:.2f}", delta=f"{variation:.2f}%")
c3.metric("J+5 prédit",      f"{forecast[-1]:.2f}")
c4.metric("Régime HMM",      str(regime) if regime is not None else "N/A")
c5.metric("Dernière donnée", last_data_date)

st.divider()

# ----- GRAPHE --------------------------------------------------

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=hist["Date"].tail(90),
    y=hist["Close"].tail(90),
    name="Historique",
    line=dict(color="#00b4d8", width=2)
))

future_dates = pd.date_range(
    start=hist["Date"].iloc[-1],
    periods=HORIZON + 1
)[1:]

fig.add_trace(go.Scatter(
    x=future_dates,
    y=forecast,
    mode="lines+markers",
    name="Forecast",
    line=dict(color="#f77f00", width=2, dash="dot"),
    marker=dict(size=7)
))

last_date_str = str(hist["Date"].iloc[-1])[:10]

fig.add_shape(
    type="line",
    x0=last_date_str, x1=last_date_str,
    y0=0, y1=1,
    xref="x", yref="paper",
    line=dict(color="gray", dash="dash", width=1)
)

fig.add_annotation(
    x=last_date_str,
    y=1,
    xref="x", yref="paper",
    text="Aujourd'hui",
    showarrow=False,
    font=dict(color="gray"),
    xanchor="left"
)

fig.update_layout(
    template="plotly_dark",
    height=600,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis_title="Date",
    yaxis_title="Prix (Close)",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# ----- TABLEAU ---------------------------------------------------------

st.subheader("📋 Prévisions détaillées")

st.dataframe(pd.DataFrame({
    "Date":         future_dates.strftime("%d/%m/%Y"),
    "Close prédit": [f"{v:.2f}" for v in forecast],
    "Variation":    [f"{(v - current) / current * 100:.2f}%" for v in forecast]
}), use_container_width=True, hide_index=True)
