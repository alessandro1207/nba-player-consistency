from __future__ import annotations

import os
import pandas as pd
import numpy as np

from src.metrics import (
    mp_to_minutes,
    add_noi_ndi,
    add_net_impact_per36,
    season_consistency,
)
from src.evaluation import plot_validation, plot_volatility_vs_minutes

RAW_PATH = "data/raw/NBA_game_logs.csv"
PROCESSED_GAMES_PATH = "data/processed/nba_game_logs_processed.csv"
PROCESSED_SEASONS_PATH = "data/processed/nba_player_seasons.csv"


def build_season_from_date(df: pd.DataFrame, date_col: str = "Date") -> pd.Series:
    d = pd.to_datetime(df[date_col], errors="coerce")
    season_start = d.dt.year.where(d.dt.month >= 7, d.dt.year - 1)
    return season_start.astype("Int64").astype(str)


def main() -> None:
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 1) Load raw
    df = pd.read_csv(RAW_PATH)

    # 2) Ensure player_id
    if "player_id" not in df.columns:
        if "Player" in df.columns:
            df["player_id"] = pd.factorize(df["Player"])[0].astype(int)
        else:
            raise ValueError("Missing 'player_id' and 'Player' columns.")

    # 3) Ensure Season
    if "Season" not in df.columns:
        if "Date" in df.columns:
            df["Season"] = build_season_from_date(df, "Date")
        else:
            raise ValueError("Missing 'Season' and 'Date' columns.")
    
    # Home/Away from Location: empty = home, '@' = away
    if "Location" in df.columns:
        loc = df["Location"].fillna("").astype(str).str.strip()
        df["is_home"] = (loc == "").astype(int)   # 1 if home, 0 if away
    else:
        df["is_home"] = np.nan

    # 4) MP → minutes
    if "MP" not in df.columns:
        raise ValueError("Missing 'MP' column in raw data.")
    df["MP_min"] = mp_to_minutes(df, mp_col="MP")
    
    # 4b) Filter: remove ultra-short stints
    before = len(df)
    n_short = (df["MP_min"] < 5).sum()
    print("games with MP_min < 5:", int(n_short), "/", before)

    df = df[df["MP_min"] >= 5].copy()
    print("after MP>=5 filter:", len(df), "(-", before - len(df), ")")
  
    # 5) NOI / NDI
    df = add_noi_ndi(df)

    # 6) Net Impact per-36
    df = add_net_impact_per36(df, mp_min_col="MP_min")

    # 7) Season-level table
    df_season = season_consistency(
        df,
        player_col="player_id",
        season_col="Season",
        ni36_col="Net_Impact_36",
    )
    
    # 7b) Filter: min games per season (si tu veux l'appliquer)
    min_games_season = 40
    df_season = df_season[df_season["games_played"] >= min_games_season].copy()

    # s'assurer que Season est triable correctement
    df_season["Season"] = pd.to_numeric(df_season["Season"], errors="coerce")

    # 7c) Target next-season (DOIT être avant tout print/ML qui l'utilise)
    df_season = df_season.sort_values(["player_id", "Season"]).copy()
    df_season["relative_volatility_next"] = (
        df_season.groupby("player_id")["relative_volatility"].shift(-1)
    )

    # prints de debug
    print("rel_vol_next percentiles:", df_season["relative_volatility_next"].quantile([0.5,0.9,0.95,0.99]))

    # Option (souvent utile): drop NA target pour le dataset ML
    df_season_ml = df_season.dropna(subset=["relative_volatility_next"]).copy()

    # 8) Save outputs
    df.to_csv(PROCESSED_GAMES_PATH, index=False)
    df_season.to_csv(PROCESSED_SEASONS_PATH, index=False)

    print(f"Saved: {PROCESSED_GAMES_PATH}")
    print(f"Saved: {PROCESSED_SEASONS_PATH}")
    print("Nb player-seasons:", len(df_season))

    # 9) Graph 1 — TON graphe (inchangé), + régression + couleur minutes
    plot_validation(
        df_season,
        x_col="net_volatility",
        y_col="relative_volatility",
        color_col="minutes_avg",
        add_regression=True,
        clip_q=0.95,
        save_path="results/consistency_validation.png",
        title="Consistency Validation: Net Impact",
    )

    # 10) Graph 2 — volatilité vs minutes, coloré par ROLE (quartiles minutes)
    plot_volatility_vs_minutes(
        df_season,
        x_col="minutes_avg",
        y_col="relative_volatility",
        mode="role",
        add_regression=True,
        save_path="results/volatility_vs_minutes_roles.png",
        title="Consistency vs Playing Time (role proxy)",
    )

    # 11) Graph 3 — volatilité vs minutes, STAR vs NON-STAR (top 10% net_mean)
    plot_volatility_vs_minutes(
        df_season,
        x_col="minutes_avg",
        y_col="relative_volatility",
        mode="star",
        star_score_col="net_mean",
        star_top_pct=0.10,
        add_regression=True,
        clip_q=0.95,
        save_path="results/volatility_vs_minutes_star.png",
        title="Consistency vs Playing Time (star vs non-star)",
    )

# va être principalement utulisé dans ml.py
if __name__ == "__main__":
    main()

