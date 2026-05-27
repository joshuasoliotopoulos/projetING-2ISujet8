#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 14:02:42 2026

@author: joshuasoliotopoulos
"""

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ── Chargement ─────────────────────────────
df = pd.read_csv("./future_results.csv")  # ou ton dataset enrichi
model = joblib.load("./model_final.pkl")
scaler = joblib.load("./scaler.pkl")

st.title("📈 Dashboard Actions — Prédiction J+1")

# ── Sélection ticker ───────────────────────
tickers = df["Ticker"].unique()
ticker = st.selectbox("Choisir une action", tickers)

data = df[df["Ticker"] == ticker]

# ── Affichage valeur actuelle / prédite ────
st.subheader(f"📊 {ticker}")

col1, col2 = st.columns(2)

col1.metric("Close actuel", round(data["Close_actuel"].values[0], 2))
col2.metric("Close prédit J+1", round(data["Close_prédit_J+1"].values[0], 2))

st.metric("Variation (%)", round(data["Variation_%"].values[0], 2))

# ── Graphique simple ────────────────────────
fig = px.line(
    data,
    x="Ticker",
    y=["Close_actuel", "Close_prédit_J+1"],
    markers=True,
    title="Comparaison réel vs prédiction"
)

st.plotly_chart(fig, use_container_width=True)

# ── Tableau complet ─────────────────────────
st.subheader("📋 Détails")
st.dataframe(data)
