import numpy as np
import pandas as pd

def ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def mp_to_minutes(df: pd.DataFrame, mp_col: str = "MP") -> pd.Series:
    """
    Convertit MP en minutes float.
    Supporte:
      - "MM:SS" (ex: "34:12")
      - timedelta string (ex: "0 days 00:44:55.000000000")
      - déjà numérique
    """
    s = df[mp_col]

    # déjà numérique
    if np.issubdtype(s.dtype, np.number):
        return s.astype(float)

    # tentative timedelta
    td = pd.to_timedelta(s, errors="coerce")
    out = td.dt.total_seconds() / 60.0

    # fallback "MM:SS"
    mask = out.isna()
    if mask.any():
        # split "MM:SS"
        parts = s[mask].astype(str).str.split(":", expand=True)
        if parts.shape[1] >= 2:
            mm = pd.to_numeric(parts[0], errors="coerce")
            ss = pd.to_numeric(parts[1], errors="coerce")
            out.loc[mask] = mm + ss / 60.0

    return out

def add_noi_ndi(df: pd.DataFrame) -> pd.DataFrame:
    """
    NOI (Columbia-like) from your doc:
      NOI = PTS + 1.5*AST - 1.25*TOV - 0.75*(FGA-FG) - 0.5*(FTA-FT)

    NDI (your defensive impact variant, replacing DOI name):
      NDI = 1.2*STL + 1.0*BLK + 0.8*DRB - 0.6*PF

    IMPORTANT:
      - ORB is offensive -> not included in NDI
    """
    df = df.copy()

    needed = ["PTS", "AST", "TOV", "FG", "FGA", "FT", "FTA", "STL", "BLK", "DRB", "PF"]
    df = ensure_numeric(df, needed)

    # NOI
    df["NOI"] = (
        df["PTS"]
        + 1.5 * df["AST"]
        - 1.25 * df["TOV"]
        - 0.75 * (df["FGA"] - df["FG"])
        - 0.5 * (df["FTA"] - df["FT"])
    )

    # NDI
    df["NDI"] = (
        1.2 * df["STL"]
        + 1.0 * df["BLK"]
        + 0.8 * df["DRB"]
        - 0.6 * df["PF"]
    )

    return df

def add_net_impact_per36(df: pd.DataFrame, mp_min_col: str = "MP_min") -> pd.DataFrame:
    df = df.copy()

    if mp_min_col not in df.columns:
        raise ValueError(f"Missing {mp_min_col}. Compute it first (MP -> minutes).")

    df = ensure_numeric(df, ["NOI", "NDI", mp_min_col])
    df["Net_Impact"] = df["NOI"] - df["NDI"]

    # juste avant df["Net_Impact_36"] = ...
    df.loc[df[mp_min_col] <= 0, mp_min_col] = np.nan

    # per-36
    df["Net_Impact_36"] = df["Net_Impact"] * (36.0 / df[mp_min_col])

    # clean inf/nan
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Net_Impact_36"])

    return df

def season_consistency(
    df: pd.DataFrame,
    player_col: str = "player_id",
    season_col: str = "Season",
    ni36_col: str = "Net_Impact_36",
) -> pd.DataFrame:
    """
    Season-level features (per player-season):
    """

    df = df.copy()

    # ---- required columns checks
    required = [player_col, season_col, ni36_col, "MP_min"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing {c}")

    # ---- make sure numeric where needed (safe even if already numeric)
    num_cols = [
        ni36_col, "MP_min", "FG", "FGA", "AST", "3PA", "Age", "Years_pro",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    def weighted_std(x: np.ndarray, w: np.ndarray) -> float:
        mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
        x = x[mask]
        w = w[mask]
        if x.size == 0:
            return np.nan
        m = np.average(x, weights=w)
        v = np.average((x - m) ** 2, weights=w)
        return float(np.sqrt(v))

    def safe_div(a: float, b: float) -> float:
        return float(a / b) if (b is not None and np.isfinite(b) and b > 0) else np.nan

    # ---- groupby
    g = df.groupby([player_col, season_col], sort=False)

    def _one_season(d: pd.DataFrame) -> pd.Series:
        # basics
        ni = d[ni36_col].to_numpy(dtype=float)
        mp = d["MP_min"].to_numpy(dtype=float)

        net_mean = np.nanmean(ni)
        net_vol = weighted_std(ni, mp)  # minutes-weighted

        # robust relative volatility (avoid explosion when mean ~ 0)
        # (we'll compute floor later from agg distribution)
        games_played = int(len(d))
        minutes_avg = float(np.nanmean(mp))
        minutes_vol = float(np.nanstd(mp, ddof=0))
        home_rate = float(pd.to_numeric(d.get("is_home"), errors="coerce").mean()) if "is_home" in d.columns else np.nan
        
        # usage / shot profile
        fga_sum = float(np.nansum(d["FGA"])) if "FGA" in d.columns else np.nan
        ast_sum = float(np.nansum(d["AST"])) if "AST" in d.columns else np.nan
        threepa_sum = float(np.nansum(d["3PA"])) if "3PA" in d.columns else np.nan
        fg_sum = float(np.nansum(d["FG"])) if "FG" in d.columns else np.nan
        mp_sum = float(np.nansum(mp))

        fga_per_min = safe_div(fga_sum, mp_sum)
        ast_per_min = safe_div(ast_sum, mp_sum)
        three_rate = safe_div(threepa_sum, fga_sum)

        # FG% volatility (game-level FG% = FG/FGA), weighted by attempts
        fg_pct_vol = np.nan
        if ("FG" in d.columns) and ("FGA" in d.columns):
            fga = d["FGA"].to_numpy(dtype=float)
            fg = d["FG"].to_numpy(dtype=float)
            fg_pct = np.divide(fg, fga, out=np.full_like(fg, np.nan), where=(fga > 0))
            fg_pct_vol = weighted_std(fg_pct, fga)

        # age / experience
        age_avg = float(np.nanmean(d["Age"])) if "Age" in d.columns else np.nan
        years_pro_avg = float(np.nanmean(d["Years_pro"])) if "Years_pro" in d.columns else np.nan

        # context sensitivity (optional)
        home_away_diff = np.nan
        if "Location" in d.columns:
            # try to interpret home/away robustly
            loc = d["Location"].astype(str).str.lower()
            is_home = loc.isin(["h", "home", "1"])  # adjust if your encoding differs
            is_away = loc.isin(["a", "away", "0"])
            if is_home.any() and is_away.any():
                home_away_diff = float(np.nanmean(ni[is_home.to_numpy()]) - np.nanmean(ni[is_away.to_numpy()]))

        return pd.Series({
            "net_mean": net_mean,
            "net_volatility": net_vol,
            "games_played": games_played,
            "minutes_avg": minutes_avg,
            "minutes_volatility": minutes_vol,
            "fga_per_min": fga_per_min,
            "ast_per_min": ast_per_min,
            "three_rate": three_rate,
            "fg_pct_volatility": fg_pct_vol,
            "age_avg": age_avg,
            "years_pro_avg": years_pro_avg,
            "home_away_diff": home_away_diff,
            "home_rate": home_rate,
        })

    # pandas warning fix (include_groups may not exist depending on pandas version)
    try:
        agg = g.apply(_one_season, include_groups=False).reset_index()
    except TypeError:
        agg = g.apply(_one_season).reset_index()

    # robust rel_vol after we have distribution of net_mean
    abs_mean = agg["net_mean"].abs()
    mean_floor = abs_mean.quantile(0.25)
    mean_floor = float(mean_floor) if pd.notna(mean_floor) and mean_floor > 0 else 1.0

    agg["relative_volatility"] = agg["net_volatility"] / (abs_mean + mean_floor)

    return agg
