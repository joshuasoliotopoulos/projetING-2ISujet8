# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import RobustScaler
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

def sharpe_ratio(returns, rf=0.0):
    returns = np.array(returns)
    if returns.std() == 0:
        return 0
    return (returns.mean() - rf) / returns.std() * np.sqrt(252)


def max_drawdown(cum_returns):
    cum_returns = np.array(cum_returns)
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - peak) / peak
    return drawdown.min()


def hit_rate(returns):
    returns = np.array(returns)
    return np.mean(returns > 0)

# ── 1. Chargement ─────────────────────────────────────────────────────────────
from traitement_de_donnee import main, prepare_datasets


fill_actions,fill_sectors,fill_bourses=main()

df = fill_actions.copy()

fill_sector, fill_action, fill_bourse, ticker_encoder = prepare_datasets()

# ── 2. Filtre jours de trading ────────────────────────────────────────────────
print(f"Lignes avant filtre is_trading_day : {len(df)}")
df = df[df["is_trading_day"] == 1].copy()
print(f"Lignes après filtre is_trading_day : {len(df)}\n")
df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

# ── 3. Target + lags ──────────────────────────────────────────────────────────
df["target_close"] = df.groupby("Ticker")["Close"].shift(-1)

for lag in [1, 3, 5, 10, 20, 30]:
    df[f"Close_lag_{lag}"]  = df.groupby("Ticker")["Close"].shift(lag)
    df[f"Volume_lag_{lag}"] = df.groupby("Ticker")["Volume"].shift(lag)
    df[f"Return_lag_{lag}"] = df.groupby("Ticker")["Close"].pct_change(lag)

df["Close_roll_mean_5"]  = df.groupby("Ticker")["Close"].transform(
    lambda x: x.shift(1).rolling(5).mean()
)
df["Close_roll_std_5"]   = df.groupby("Ticker")["Close"].transform(
    lambda x: x.shift(1).rolling(5).std()
)
df["Close_roll_mean_20"] = df.groupby("Ticker")["Close"].transform(
    lambda x: x.shift(1).rolling(20).mean()
)
df["Close_roll_std_20"]  = df.groupby("Ticker")["Close"].transform(
    lambda x: x.shift(1).rolling(20).std()
)
df["Volume_roll_mean_5"] = df.groupby("Ticker")["Volume"].transform(
    lambda x: x.shift(1).rolling(5).mean()
)
df["Volume_roll_std_5"]  = df.groupby("Ticker")["Volume"].transform(
    lambda x: x.shift(1).rolling(5).std()
)

df["day_of_week"]  = pd.to_datetime(df["Date"]).dt.dayofweek
df["week_of_year"] = pd.to_datetime(df["Date"]).dt.isocalendar().week.astype(int)
df["month"]        = pd.to_datetime(df["Date"]).dt.month
df["quarter"]      = pd.to_datetime(df["Date"]).dt.quarter

df = df.dropna(subset=["target_close"])

for file in ["scaler.pkl", "important_features.pkl", "model_final.pkl", "best_params.pkl"]:
    if os.path.exists(file):
        os.remove(file)

# ── 4. Boîte à moustaches AVANT normalisation ─────────────────────────────────
print("Génération boîtes à moustaches avant normalisation...\n")
numeric_cols = [c for c in ["Close", "Volume", "RSI", "MACD", "Momentum",
                            "Volatility_20d", "BB_Upper", "BB_Lower",
                            "MA20", "MA50"] if c in df.columns]
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    if i < len(axes):
        axes[i].boxplot(df[col].dropna(), patch_artist=True,
                        boxprops=dict(facecolor="#4C72B0", color="white"),
                        medianprops=dict(color="orange", linewidth=2),
                        whiskerprops=dict(color="white"),
                        capprops=dict(color="white"),
                        flierprops=dict(marker="o", color="red", alpha=0.3))
        axes[i].set_title(col, color="white", fontsize=11)
        axes[i].set_facecolor("#1e1e1e")
        axes[i].tick_params(colors="white")
        for spine in axes[i].spines.values():
            spine.set_edgecolor("#444")
for j in range(len(numeric_cols), len(axes)):
    axes[j].set_visible(False)
fig.patch.set_facecolor("#1e1e1e")
fig.suptitle("Boîtes à moustaches AVANT RobustScaler",
             color="white", fontsize=14)
plt.tight_layout()
plt.savefig("boxplots.png", dpi=150, bbox_inches="tight",
                facecolor="#1e1e1e")
plt.show()

# ── 5. Matrice de corrélation ─────────────────────────────────────────────────
print("Génération matrice de corrélation...\n")
corr_cols = [c for c in ["Close", "Volume", "RSI", "MACD", "Momentum",
                              "Volatility_20d", "MA20", "MA50", "BB_Upper",
                              "BB_Lower", "VIX_Close", "target_close"] if c in df.columns]
corr_matrix = df[corr_cols].corr()
fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, vmin=-1, vmax=1,
            linewidths=0.5, ax=ax, annot_kws={"size": 8})
ax.set_title("Matrice de corrélation", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig("correlation_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 6. Features / Target ──────────────────────────────────────────────────────
X = df.drop(columns=["Date", "target_close"])
X = X.select_dtypes(include=[np.number])
y = np.log1p(df["target_close"])

# ── 7. Normalisation RobustScaler ─────────────────────────────────────────────
print("Calcul du RobustScaler...\n")
scaler   = RobustScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")

X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
print(f"Données normalisées — shape : {X_scaled.shape}\n")

# ── 8. Boîte à moustaches APRÈS normalisation ─────────────────────────────────
print("Génération boîtes à moustaches après normalisation...\n")
numeric_cols_scaled = [c for c in ["Close", "Volume", "RSI", "MACD",
                                        "Momentum", "Volatility_20d",
                                        "BB_Upper", "BB_Lower", "MA20", "MA50"] if c in X_scaled.columns]
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()
for i, col in enumerate(numeric_cols_scaled):
    if i < len(axes):
        data = X_scaled[col].dropna()
        axes[i].boxplot(data, patch_artist=True,
                            boxprops=dict(facecolor="#55a868", color="white"),
                            medianprops=dict(color="orange", linewidth=2),
                            whiskerprops=dict(color="white"),
                            capprops=dict(color="white"),
                            flierprops=dict(marker="o", color="red", alpha=0.3))
        axes[i].axhline(y=0, color="orange", linestyle="--",
                            linewidth=1, alpha=0.5)
        axes[i].set_title(
                f"{col}\nmed={data.median():.2f} | iqr={data.quantile(0.75) - data.quantile(0.25):.2f}",
                color="white", fontsize=9
            )
        axes[i].set_facecolor("#1e1e1e")
        axes[i].tick_params(colors="white")
        for spine in axes[i].spines.values():
            spine.set_edgecolor("#444")
for j in range(len(numeric_cols_scaled), len(axes)):
    axes[j].set_visible(False)
fig.patch.set_facecolor("#1e1e1e")
fig.suptitle("Boîtes à moustaches APRÈS RobustScaler",
                 color="white", fontsize=14)
plt.tight_layout()
plt.savefig("boxplots_scaled.png", dpi=150, bbox_inches="tight",
                facecolor="#1e1e1e")
plt.show()

# ── 9. Split temporel ─────────────────────────────────────────────────────────
tscv = TimeSeriesSplit(n_splits=3)

# ── 10. Entraînement initial (feature selection) ──────────────────────────────
print(" Entraînement initial pour sélection des features...\n")

model_init = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )

scores_mae_init, scores_r2_init, scores_rmse_init = [], [], []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
    X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

model_init.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )

preds      = np.expm1(model_init.predict(X_val))
y_val_real = np.expm1(y_val)

scores_mae_init.append(mean_absolute_error(y_val_real, preds))
scores_r2_init.append(r2_score(y_val_real, preds))
scores_rmse_init.append(np.sqrt(mean_squared_error(y_val_real, preds)))
        
print(f"Fold {fold+1} — MAE: {scores_mae_init[-1]:.2f} | RMSE: {scores_rmse_init[-1]:.2f} | R²: {scores_r2_init[-1]:.4f}")

print(f"RMSE moyen (initial) : {np.mean(scores_rmse_init):.2f}")
print(f"\n MAE moyen (initial) : {np.mean(scores_mae_init):.2f}")
print(f"R² moyen  (initial) : {np.mean(scores_r2_init):.4f}")

importance = pd.DataFrame({
    "feature"    : X_scaled.columns,
    "importance" : model_init.feature_importances_
    }).sort_values("importance", ascending=False)

print("\n Top 20 features :")
print(importance.head(20))

threshold = importance["importance"].sum() * 0.001
important_features = importance[importance["importance"] > threshold]["feature"].tolist()

print(f"\nFeatures droppées   : {len(X_scaled.columns) - len(important_features)}")
print(f" Features conservées : {len(important_features)}")

joblib.dump(important_features, "important_features.pkl")

X_reduced = X_scaled[important_features]

# ── 11. RandomizedSearchCV ────────────────────────────────────────────────────
print(" Recherche des meilleurs hyperparamètres...\n")

param_distributions = {
        "n_estimators"      : [200, 300, 500, 700, 1000],
        "learning_rate"     : [0.01, 0.03, 0.05, 0.08, 0.1],
        "num_leaves"        : [31, 63, 95, 127, 255],
        "max_depth"         : [4, 6, 8, 10, 12],
        "min_child_samples" : [10, 20, 50, 100],
        "subsample"         : [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree"  : [0.6, 0.7, 0.8, 0.9, 1.0],
        "reg_alpha"         : [0.0, 0.01, 0.1, 1.0],
        "reg_lambda"        : [0.0, 0.01, 0.1, 1.0],
    }

tscv_search = TimeSeriesSplit(n_splits=3)

search = RandomizedSearchCV(
        estimator=lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbosity=-1),
        param_distributions=param_distributions,
        n_iter=20,
        scoring="neg_mean_absolute_error",
        cv=tscv_search,
        random_state=42,
        n_jobs=-1,
        verbose=2
    )

search.fit(X_reduced, y)

best_params = search.best_params_
best_params["random_state"] = 42
best_params["n_jobs"]       = -1
best_params["verbosity"]    = -1

print(f"\n Meilleurs hyperparamètres : {best_params}")
print(f" Meilleur MAE : {-search.best_score_:.4f}")

joblib.dump(best_params, "best_params.pkl")

# ── 12. Entraînement final ─────────────────────────────────────────────────────
print("\n Entraînement final...\n")
model_final = lgb.LGBMRegressor(**best_params)

scores_mae_final, scores_r2_final, scores_rmse_final = [], [], []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_reduced)):
    X_train, X_val = X_reduced.iloc[train_idx], X_reduced.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model_final.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )

    preds      = np.expm1(model_final.predict(X_val))
    y_val_real = np.expm1(y_val)

    scores_mae_final.append(mean_absolute_error(y_val_real, preds))
    scores_r2_final.append(r2_score(y_val_real, preds))
    scores_rmse_final.append(np.sqrt(mean_squared_error(y_val_real, preds)))
        
    print(f"Fold {fold+1} — MAE: {scores_mae_final[-1]:.2f} | RMSE: {scores_rmse_final[-1]:.2f} | R²: {scores_r2_final[-1]:.4f}")

print(f"RMSE moyen (final) : {np.mean(scores_rmse_final):.2f}")
print(f"\n MAE moyen (final) : {np.mean(scores_mae_final):.2f}")
print(f" R² moyen  (final) : {np.mean(scores_r2_final):.4f}")

joblib.dump(model_final, "model_final.pkl")

# ── 13. Graphiques R² et MAE ──────────────────────────────────────────────────
folds = list(range(1, len(scores_mae_final) + 1))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#1e1e1e")

axes[0].plot(folds, scores_mae_final, color="#4C72B0", marker="o",
             linewidth=2, markersize=8, label="MAE par fold")
axes[0].axhline(y=np.mean(scores_mae_final), color="orange",
                linestyle="--", linewidth=2,
                label=f"MAE moyen : {np.mean(scores_mae_final):.2f}")
axes[0].fill_between(folds, scores_mae_final,
                     np.mean(scores_mae_final), alpha=0.15, color="#4C72B0")
axes[0].set_title("MAE par fold", color="white", fontsize=13)
axes[0].set_xlabel("Fold", color="white")
axes[0].set_ylabel("MAE", color="white")
axes[0].set_xticks(folds)
axes[0].set_facecolor("#1e1e1e")
axes[0].tick_params(colors="white")
axes[0].legend(facecolor="#2c2c2c", labelcolor="white")
for spine in axes[0].spines.values():
    spine.set_edgecolor("#444")

axes[1].plot(folds, scores_r2_final, color="#55a868", marker="o",
             linewidth=2, markersize=8, label="R² par fold")
axes[1].axhline(y=np.mean(scores_r2_final), color="orange",
                linestyle="--", linewidth=2,
                label=f"R² moyen : {np.mean(scores_r2_final):.4f}")
axes[1].fill_between(folds, scores_r2_final,
                     np.mean(scores_r2_final), alpha=0.15, color="#55a868")
axes[1].set_title("R² par fold", color="white", fontsize=13)
axes[1].set_xlabel("Fold", color="white")
axes[1].set_ylabel("R²", color="white")
axes[1].set_xticks(folds)
axes[1].set_ylim(0, 1)
axes[1].set_facecolor("#1e1e1e")
axes[1].tick_params(colors="white")
axes[1].legend(facecolor="#2c2c2c", labelcolor="white")
for spine in axes[1].spines.values():
    spine.set_edgecolor("#444")

fig.suptitle("Performance du modèle par fold — LightGBM",
             color="white", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig("performance_folds.png", dpi=150, bbox_inches="tight",
            facecolor="#1e1e1e")
plt.show()

# ── 14. Prédictions sur le dernier fold ───────────────────────────────────────
train_idx, val_idx = list(tscv.split(X_reduced))[-1]

y_val_real = np.expm1(y.iloc[val_idx])
preds_real = np.expm1(model_final.predict(X_reduced.iloc[val_idx]))

market_returns = np.diff(y_val_real.values, prepend=y_val_real.values[0]) / y_val_real.values

threshold = 0.005
signal = (preds_real - y_val_real) / y_val_real

strategy_returns = np.where(
    signal > threshold,
    market_returns,
    np.where(signal < -threshold, -market_returns, 0)
)
strategy_returns = np.clip(strategy_returns, -0.1, 0.1)

tickers_encoded = df.iloc[val_idx]["Ticker"].astype(int).values

tickers = ticker_encoder.inverse_transform(
    np.clip(tickers_encoded, 0, len(ticker_encoder.classes_) - 1)
)

results = pd.DataFrame({
    "Date": df.iloc[val_idx]["Date"].values,
    "Ticker": tickers,
    "Close_réel": y_val_real.values,
    "Close_prédit": preds_real,
    "Erreur_%": np.abs(y_val_real.values - preds_real) / y_val_real.values * 100
})

print("\n Résultats sur le dernier fold :")
print(results.head(20))

sharpe = sharpe_ratio(strategy_returns)
mdd    = max_drawdown(np.cumprod(1 + strategy_returns))
hit    = hit_rate(strategy_returns)

print("\n==============================")
print(" STRATEGY PERFORMANCE")
print("==============================")
print(f"Sharpe Ratio : {sharpe:.4f}")
print(f"Max Drawdown : {mdd:.4f}")
print(f"Hit Rate     : {hit:.4f}")
print("==============================")


# ── 15. Graphique Close réel vs prédit ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
fig.patch.set_facecolor("#1e1e1e")

ax.plot(results["Date"], results["Close_réel"],
        color="#4C72B0", linewidth=1.5, label="Close réel")
ax.plot(results["Date"], results["Close_prédit"],
        color="orange", linewidth=1.5, linestyle="--", label="Close prédit")
ax.set_title("Close réel vs Close prédit — Dernier fold",
             color="white", fontsize=13)
ax.set_xlabel("Date", color="white")
ax.set_ylabel("Prix (€/$)", color="white")
ax.set_facecolor("#1e1e1e")
ax.tick_params(colors="white", axis="both")
ax.legend(facecolor="#2c2c2c", labelcolor="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#444")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("close_reel_vs_predit.png", dpi=150, bbox_inches="tight",
            facecolor="#1e1e1e")
plt.show()

# ── 16. Graphique erreur % ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
fig.patch.set_facecolor("#1e1e1e")

ax.bar(results["Date"], results["Erreur_%"],
       color=["#ff4444" if e > 5 else "#55a868" for e in results["Erreur_%"]],
       width=0.8, alpha=0.8)
ax.axhline(y=results["Erreur_%"].mean(), color="orange",
           linestyle="--", linewidth=1.5,
           label=f"Erreur moyenne : {results['Erreur_%'].mean():.2f}%")
ax.set_title("Erreur % par jour — Dernier fold",
             color="white", fontsize=13)
ax.set_xlabel("Date", color="white")
ax.set_ylabel("Erreur %", color="white")
ax.set_facecolor("#1e1e1e")
ax.tick_params(colors="white")
ax.legend(facecolor="#2c2c2c", labelcolor="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#444")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("erreur_par_jour.png", dpi=150, bbox_inches="tight",
            facecolor="#1e1e1e")
plt.show()

# ── 17. Prédiction J+1 ────────────────────────────────────────────────────────
last_known        = df.sort_values("Date").groupby("Ticker").last().reset_index()
last_known_scaled = scaler.transform(last_known[X.columns])
last_known_scaled = pd.DataFrame(last_known_scaled, columns=X.columns)

missing = [f for f in important_features if f not in last_known_scaled.columns]
if missing:
    print(f" Features manquantes pour J+1 : {missing}")
else:
    X_future = last_known_scaled[important_features]
    
    codes = last_known["Ticker"].astype(int).values

    valid_max = len(ticker_encoder.classes_) - 1
    
    codes = np.clip(codes, 0, valid_max)
    
    tickers = ticker_encoder.inverse_transform(codes)

    future_results = pd.DataFrame({
        "Ticker"           : tickers,
        "Close_actuel"     : last_known["Close"].values,
        "Close_prédit_J+1" : np.expm1(model_final.predict(X_future)),
    })

    future_results["Variation_%"] = (
        (future_results["Close_prédit_J+1"] - future_results["Close_actuel"])
        / future_results["Close_actuel"] * 100
    ).round(2)

    # Filtre variations irréalistes
    future_results = future_results[
        (future_results["Variation_%"] > -20) &
        (future_results["Variation_%"] <  20)
    ].sort_values("Variation_%", ascending=False).reset_index(drop=True)

    print("\n Prédictions Close J+1 :")
    print(future_results)

    future_results.to_csv("future_results.csv", index=False)

# ── 18. Sauvegarde finale ─────────────────────────────────────────────────────
joblib.dump(model_final,        "model_final.pkl")
joblib.dump(important_features, "important_features.pkl")
joblib.dump(ticker_encoder,     "ticker_encoder.pkl")
joblib.dump(scaler,             "scaler.pkl")

print("\n Modèle, scaler et encodeur sauvegardés.")
