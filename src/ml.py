from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

@dataclass
class MLResults:
    metrics: pd.DataFrame
    predictions: pd.DataFrame


def make_next_season_dataset(
    df_season: pd.DataFrame,
    player_col: str = "player_id",
    season_col: str = "Season",
    target_col: str = "relative_volatility",
    feature_cols: Optional[list[str]] = None,
) -> pd.DataFrame:

    """
    Features from season t -> target is target_col in season t+1.
    Output includes: features + target_next + player_id + Season (season t).
    """
    if feature_cols is None:
        feature_cols = ["relative_volatility",
            "net_mean",
            "net_volatility",
            "games_played",
            "minutes_avg",
            "minutes_volatility",
            "fga_per_min",
            "ast_per_min",
            "three_rate",
            "fg_pct_volatility",
            "age_avg",
            "years_pro_avg",
            "home_rate",
        ]

    # target names
    target_next_col = f"{target_col}_next"

    needed = [player_col, season_col, target_col] + feature_cols
    missing = [c for c in needed if c not in df_season.columns]
    if missing:
        raise ValueError(f"Missing columns for ML dataset: {missing}")

    d = df_season.copy()
    d[season_col] = pd.to_numeric(d[season_col], errors="coerce")
    d = d.dropna(subset=[season_col]).copy()
    d[season_col] = d[season_col].astype(int)

    # sécurité: si la colonne next existe déjà, on la drop pour éviter _x/_y au merge
    if target_next_col in d.columns:
        d = d.drop(columns=[target_next_col])


    # Next-season target: shift season back by 1 so season t holds target(t+1)
    # --- Next-season target: align season t row with target from season t+1
    nxt = d[[player_col, season_col, target_col]].copy()
    nxt[season_col] = nxt[season_col] - 1
    nxt = nxt.rename(columns={target_col: target_next_col})

    out = d.merge(
        nxt,
        on=[player_col, season_col],
        how="inner",
    )

    if target_next_col not in out.columns:
        raise ValueError(
            f"{target_next_col} not created. "
            f"Check target_col='{target_col}' and columns in df_season. "
            f"out cols sample: {list(out.columns)[:30]}"
        )

    # convert numeric where possible (features + target_next)
    for c in feature_cols + [target_next_col]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # drop uniquement si le label futur est manquant
    out = out.dropna(subset=[target_next_col]).copy()
    return out

def split_train_test(
    df_ml: pd.DataFrame,
    season_col: str = "Season",
    train_years: list[int] = None,
    test_year: int = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if train_years is None or test_year is None:
        raise ValueError("Provide train_years and test_year.")
    train = df_ml[df_ml[season_col].isin(train_years)].copy()
    test = df_ml[df_ml[season_col] == test_year].copy()
    return train, test


def _eval(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def eval_regression(y_true, y_pred, model_name: str) -> dict:
    out = _eval(y_true, y_pred)          # réutilise la fonction existante
    out["model"] = model_name
    return out

def run_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    target_next_col: str = "relative_volatility_next",
    random_state: int = 42,
) -> MLResults:
    # (NEW) clean + impute (fit medians on train only to avoid leakage)
    train = train.dropna(subset=[target_next_col]).copy()
    test  = test.dropna(subset=[target_next_col]).copy()

    # force numeric + compute medians on train
    train_medians = {}
    for c in feature_cols:
        if c in train.columns:
            train[c] = pd.to_numeric(train[c], errors="coerce")
            med = train[c].median()
            train_medians[c] = med
            train[c] = train[c].fillna(med)

            # apply SAME median to test
            if c in test.columns:
                test[c] = pd.to_numeric(test[c], errors="coerce")
                test[c] = test[c].fillna(med)

    # drop rows where any feature is still NaN (can happen if column missing entirely)
    train = train.dropna(subset=feature_cols).copy()
    test  = test.dropna(subset=feature_cols).copy()

    X_train = train[feature_cols].to_numpy()
    y_train = train[target_next_col].to_numpy()
    X_test  = test[feature_cols].to_numpy()
    y_test  = test[target_next_col].to_numpy()


    models = {
        "Linear": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "Ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0, random_state=random_state))]),
        "RandomForest": RandomForestRegressor(
            n_estimators=400,
            random_state=random_state,
            n_jobs=-1,
            min_samples_leaf=2,
        ),
    }

    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
        )

    rows = []
    preds_all = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        m = _eval(y_test, pred)
        rows.append({"model": name, **m})

        p = test[["player_id", "Season"]].copy()
        p["y_true"] = y_test
        p["y_pred"] = pred
        p["model"] = name
        preds_all.append(p)

    metrics_df = pd.DataFrame(rows).sort_values("RMSE")
    preds_df = pd.concat(preds_all, ignore_index=True)
    return MLResults(metrics=metrics_df, predictions=preds_df)


def save_pred_vs_actual_plots(preds_df: pd.DataFrame, save_dir: str = "results/ml") -> None:
    os.makedirs(save_dir, exist_ok=True)

    for model_name in preds_df["model"].unique():
        d = preds_df[preds_df["model"] == model_name].copy()

        plt.figure(figsize=(8, 6))
        plt.scatter(d["y_true"], d["y_pred"], alpha=0.6, s=25)

        lo = float(np.nanmin([d["y_true"].min(), d["y_pred"].min()]))
        hi = float(np.nanmax([d["y_true"].max(), d["y_pred"].max()]))
        plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=2)

        plt.xlabel("Actual next-season relative volatility")
        plt.ylabel("Predicted next-season relative volatility")
        plt.title(f"Predicted vs Actual — {model_name}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"pred_vs_actual_{model_name}.png"), dpi=200)
        plt.close()

if __name__ == "__main__":

    df = pd.read_csv("data/processed/nba_player_seasons.csv")

    ml_df = make_next_season_dataset(df)
    
    print("RAW df seasons min/max:", df["Season"].min(), df["Season"].max())
    print("RAW rows per season (tail):")
    print(df["Season"].value_counts().sort_index().tail(12))

    print("ML df seasons min/max:", ml_df["Season"].min(), ml_df["Season"].max())
    print("ML rows per season (tail):")
    print(ml_df["Season"].value_counts().sort_index().tail(12))
    print("Available seasons in ml_df:", sorted(ml_df["Season"].unique())[-7:])

    target = "relative_volatility_next"

    features = ["relative_volatility",
            "net_mean",
            "net_volatility",
            "games_played",
            "minutes_avg",
            "minutes_volatility",
            "fga_per_min",
            "ast_per_min",
            "three_rate",
            "fg_pct_volatility",
            "age_avg",
            "years_pro_avg",
            "home_rate",
    ]

    # We predict next-season volatility (t+1),
    # so the last usable input season is 2018 (predicting 2019)
    train_start, train_end = 2015, 2017
    test_season = 2018

    train_df = ml_df[
        (ml_df["Season"] >= train_start) &
        (ml_df["Season"] <= train_end)
    ].copy()

    test_df = ml_df[
        ml_df["Season"] == test_season
    ].copy()

    print("Train seasons:", train_start, "->", train_end, "rows:", len(train_df))
    print("Test season:", test_season, "rows:", len(test_df))

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("Train or test split is empty. Check available seasons in ml_df.")

    X_train, y_train = train_df[features], train_df[target]
    X_test,  y_test  = test_df[features],  test_df[target]

    lin = LinearRegression()
    lin.fit(X_train, y_train)

    y_pred_lin = lin.predict(X_test)

    lin_metrics = {
        "model": "Linear",
        "MAE": mean_absolute_error(y_test, y_pred_lin),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_lin)),
        "R2": r2_score(y_test, y_pred_lin),
    }

    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])

    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)

    ridge_metrics = {
        "model": "Ridge",
        "MAE": mean_absolute_error(y_test, y_pred_ridge),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
        "R2": r2_score(y_test, y_pred_ridge),
    }

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        random_state=42,
    )

    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    rf_metrics = {
        "model": "RandomForest",
        "MAE": mean_absolute_error(y_test, y_pred_rf),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        "R2": r2_score(y_test, y_pred_rf),
    }

    if not HAS_XGB:
        print("XGBoost not installed — skipping.")

    if HAS_XGB:
        xgb = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective="reg:squarederror",
        )

        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)

        xgb_metrics = {
            "model": "XGBoost",
            "MAE": mean_absolute_error(y_test, y_pred_xgb),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
            "R2": r2_score(y_test, y_pred_xgb),
        }



    print("ML dataset shape:", ml_df.shape)
    print("Columns:", ml_df.columns.tolist())
    print(ml_df.head())

    # ---------- TRAIN / TEST SPLIT (time-based) ----------
    target = "relative_volatility_next"
    features = ["relative_volatility",
            "net_mean",
            "net_volatility",
            "games_played",
            "minutes_avg",
            "minutes_volatility",
            "fga_per_min",
            "ast_per_min",
            "three_rate",
            "fg_pct_volatility",
            "age_avg",
            "years_pro_avg",
            "home_rate", 
    ]

    # =========================
    # WALK-FORWARD (limited folds + better display)
    # =========================

    # --- SETTINGS (tu modifies ici uniquement)
    N_FOLDS = 3                # <- tu veux ~3 folds      
    ROLLING_K = 3             # si MODE="rolling": nombre de saisons train (ex: 2 saisons avant ts)
    MIN_TRAIN_ROWS = 50        # sécurité: skip si train trop petit

    # target/features doivent déjà exister au-dessus
    # target = "relative_volatility_next"
    # features = [...]

    # (Optionnel mais recommandé) on drop seulement si target manque
    ml_df = ml_df.dropna(subset=[target]).copy()

    # saisons disponibles
    seasons = sorted(ml_df["Season"].dropna().unique().astype(int))
    # On prend les N dernières saisons comme test (ex: ...2016,2017,2018 si N_FOLDS=3)
    test_seasons = seasons[-N_FOLDS:]

    print("test_seasons:", test_seasons)
    print("rolling windows:", [(ts-ROLLING_K, ts-1) for ts in test_seasons])

    all_metrics: list[dict] = []
    all_preds: list[pd.DataFrame] = []

    print("\n=== WALK-FORWARD SETTINGS ===")
    print("ROLLING_K:", ROLLING_K)
    print("TEST SEASONS:", test_seasons)

    for ts in test_seasons:
        # -------- split train / test --------
        # -------- split train / test (ROLLING ONLY) --------
        train_df = ml_df[(ml_df["Season"] >= ts - ROLLING_K) & (ml_df["Season"] < ts)].copy()
        test_df  = ml_df[ml_df["Season"] == ts].copy()


        if len(train_df) < MIN_TRAIN_ROWS or len(test_df) == 0:
            print(f"[SKIP] ts={ts} train_rows={len(train_df)} test_rows={len(test_df)}")
            continue

        # -------- numeric coercion + imputation (fit on train only) --------
        for c in features:
            train_df[c] = pd.to_numeric(train_df[c], errors="coerce")
            test_df[c] = pd.to_numeric(test_df[c], errors="coerce")

        y_train = pd.to_numeric(train_df[target], errors="coerce")
        y_test = pd.to_numeric(test_df[target], errors="coerce")

        med = train_df[features].median()
        X_train = train_df[features].fillna(med)
        X_test = test_df[features].fillna(med)

        # sécurité si target contient NaN après coercion
        keep_train = y_train.notna()
        keep_test = y_test.notna()
        X_train, y_train = X_train.loc[keep_train], y_train.loc[keep_train]
        X_test, y_test = X_test.loc[keep_test], y_test.loc[keep_test]

        if len(X_train) < MIN_TRAIN_ROWS or len(X_test) == 0:
            print(f"[SKIP-after-clean] ts={ts} train={len(X_train)} test={len(X_test)}")
            continue

        # fit/predict models (tu gardes tes modèles actuels)
        lin = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
        lin.fit(X_train, y_train)
        pred_lin = lin.predict(X_test)

        ridge = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0, random_state=42))])
        ridge.fit(X_train, y_train)
        pred_ridge = ridge.predict(X_test)

        rf = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)
        rf.fit(X_train, y_train)
        pred_rf = rf.predict(X_test)

        pred_xgb = None
        if HAS_XGB:
            xgb = XGBRegressor(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                objective="reg:squarederror",
            )
            xgb.fit(X_train, y_train)
            pred_xgb = xgb.predict(X_test)

        # metrics (IMPORTANT: train seasons logged + test season)
        # eval_regression(y_true, y_pred, model_name) doit déjà exister dans ton fichier
        train_start = int(train_df["Season"].min())
        train_end = int(train_df["Season"].max())

        all_metrics.append({**eval_regression(y_test, pred_lin, "Linear"), "test_season": int(ts), "train_start": train_start, "train_end": train_end, "train_rows": int(len(X_train))})
        all_metrics.append({**eval_regression(y_test, pred_ridge, "Ridge"), "test_season": int(ts), "train_start": train_start, "train_end": train_end, "train_rows": int(len(X_train))})
        all_metrics.append({**eval_regression(y_test, pred_rf, "RandomForest"), "test_season": int(ts), "train_start": train_start, "train_end": train_end, "train_rows": int(len(X_train))})
        if HAS_XGB and pred_xgb is not None:
            all_metrics.append({**eval_regression(y_test, pred_xgb, "XGBoost"), "test_season": int(ts), "train_start": train_start, "train_end": train_end, "train_rows": int(len(X_train))})

        # predictions df (lisible)
        preds_out = test_df[["player_id", "Season"]].copy()
        preds_out["y_true"] = y_test.values
        preds_out["pred_linear"] = pred_lin
        preds_out["pred_ridge"] = pred_ridge
        preds_out["pred_rf"] = pred_rf
        if HAS_XGB and pred_xgb is not None:
            preds_out["pred_xgb"] = pred_xgb
        preds_out["test_season"] = int(ts)

        all_preds.append(preds_out)

    # aggregate + print pretty tables
    metrics_df = pd.DataFrame(all_metrics)
    preds_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()

    os.makedirs("results/ml", exist_ok=True)
    metrics_df.to_csv("results/ml/rolling_metrics_detailed.csv", index=False)
    preds_df.to_csv("results/ml/rolling_predictions_detailed.csv", index=False)

    # résumé lisible: mean/std par modèle
    summary = (
        metrics_df.groupby("model")[["MAE", "RMSE", "R2"]]
        .agg(["mean", "std"])
        .round(4)
    )

    print("\n=== rolling SUMMARY (mean ± std across folds) ===")
    print(summary)

    # résumé par saison (pratique pour voir stabilité)
    by_season = metrics_df.pivot_table(index="test_season", columns="model", values="RMSE", aggfunc="mean").round(4)
    print("\n=== RMSE by test season ===")
    print(by_season)

    # sauvegarde des résumés
    summary.to_csv("results/ml/rolling_summary_mean_std.csv")
    by_season.to_csv("results/ml/rolling_rmse_by_season.csv")

    print("\nSaved:")
    print(" - results/ml/rolling_metrics_detailed.csv")
    print(" - results/ml/rolling_predictions_detailed.csv")
    print(" - results/ml/rolling_summary_mean_std.csv")
    print(" - results/ml/rolling_rmse_by_season.csv")

    # drop uniquement si la target est manquante (sinon pas de label)
    ml_df = ml_df.dropna(subset=[target]).copy()

    test_season = int(ml_df["Season"].max())   # on teste sur la dernière saison dispo (t)
    train_df = ml_df[ml_df["Season"] < test_season].copy()
    test_df  = ml_df[ml_df["Season"] == test_season].copy()

    # imputation (fit sur train uniquement -> pas de leakage)
    for c in features:
        train_df[c] = pd.to_numeric(train_df[c], errors="coerce")
        test_df[c] = pd.to_numeric(test_df[c], errors="coerce")

    medians = train_df[features].median()

    train_df[features] = train_df[features].fillna(medians)
    test_df[features] = test_df[features].fillna(medians)

    # sécurité: si une colonne est 100% NaN même dans train, median sera NaN -> on drop ce qui reste
    train_df = train_df.dropna(subset=features + [target]).copy()
    test_df  = test_df.dropna(subset=features + [target]).copy()


    print("Train seasons:", train_df["Season"].min(), "->", train_df["Season"].max(), "rows:", len(train_df))
    print("Test season:", test_season, "rows:", len(test_df))

    X_train, y_train = train_df[features], train_df[target]
    X_test,  y_test  = test_df[features],  test_df[target]

    metrics_rows = []
    preds_out = test_df[["player_id", "Season"]].copy()
    preds_out["y_true"] = y_test.values

    # --- Linear
    lin = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    lin.fit(X_train, y_train)
    pred_lin = lin.predict(X_test)
    metrics_rows.append(eval_regression(y_test, pred_lin, "Linear"))
    preds_out["pred_linear"] = pred_lin

    # --- Ridge
    ridge = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0, random_state=42))])
    ridge.fit(X_train, y_train)
    pred_ridge = ridge.predict(X_test)
    metrics_rows.append(eval_regression(y_test, pred_ridge, "Ridge"))
    preds_out["pred_ridge"] = pred_ridge

    # --- Random Forest
    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    metrics_rows.append(eval_regression(y_test, pred_rf, "RandomForest"))
    preds_out["pred_rf"] = pred_rf

    # --- XGBoost (si installé)
    if HAS_XGB:
        xgb = XGBRegressor(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective="reg:squarederror",
        )
        xgb.fit(X_train, y_train)
        pred_xgb = xgb.predict(X_test)
        metrics_rows.append(eval_regression(y_test, pred_xgb, "XGBoost"))
        preds_out["pred_xgb"] = pred_xgb
    else:
        print("XGBoost not available -> skipping.")

    os.makedirs("results/ml", exist_ok=True)

    metrics_df = pd.DataFrame(metrics_rows).sort_values("RMSE")
    metrics_df.to_csv("results/ml/model_comparison.csv", index=False)
    preds_out.to_csv("results/ml/predictions_test_season.csv", index=False)

    print("\n=== Model comparison (sorted by RMSE) ===")
    print(metrics_df)
    print("\nSaved: results/ml/model_comparison.csv")
    print("Saved: results/ml/predictions_test_season.csv")

    # convert predictions to long format for plotting (needs 'model' column)
    pred_cols = [c for c in preds_out.columns if c.startswith("pred_")]

    preds_long = preds_out.melt(
        id_vars=["player_id", "Season", "y_true"],
        value_vars=pred_cols,
        var_name="model",
        value_name="y_pred",
    )

    preds_long["model"] = preds_long["model"].str.replace("pred_", "", regex=False)

    save_pred_vs_actual_plots(preds_long, save_dir="results/ml")

    print("Saved pred-vs-actual plots in results/ml/")
