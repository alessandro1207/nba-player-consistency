import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def clip_middle_quantiles(d: pd.DataFrame, cols: list[str], clip_q: float = 0.95) -> pd.DataFrame:
    """
    Garde seulement les valeurs dans l'intervalle [ (1-clip_q)/2 , 1-(1-clip_q)/2 ] 
    pour chaque colonne.
    clip_q=0.95 -> garde le "middle 95%" et enlève 2.5% en bas + 2.5% en haut.
    """
    if clip_q is None or clip_q >= 1.0:
        return d

    low_q = (1.0 - clip_q) / 2.0
    high_q = 1.0 - low_q

    out = d.copy()
    mask = pd.Series(True, index=out.index)
    for c in cols:
        lo = out[c].quantile(low_q)
        hi = out[c].quantile(high_q)
        mask &= out[c].between(lo, hi)
    return out.loc[mask].copy()


def plot_validation(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str | None = None,
    add_regression: bool = False,
    save_path: str | None = None,
    title: str | None = None,
    clip_q: float | None = None,
) -> None:
    d = df.copy()
    cols = [x_col, y_col] + ([color_col] if color_col else [])
    d = d.dropna(subset=cols)
    # Clip pour rendre l'échelle lisible (uniquement pour le plot)
    d = clip_middle_quantiles(d, [x_col, y_col], clip_q=clip_q or 1.0)


    plt.figure(figsize=(10, 7))

    sc = plt.scatter(
        d[x_col],
        d[y_col],
        c=d[color_col] if color_col else None,
        cmap="viridis" if color_col else None,
        alpha=0.55,
        s=25,
    )

    if color_col:
        plt.colorbar(sc, label="Average minutes played")

    if add_regression and len(d) >= 2:
        x = d[x_col].to_numpy()
        y = d[y_col].to_numpy()
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        plt.plot(xs, m * xs + b, "k--", linewidth=2, label="Linear trend")
        plt.legend()

    plt.xlabel(x_col.replace("_", " ").title())
    plt.ylabel(y_col.replace("_", " ").title())
    plt.title(title or "")
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()



def plot_volatility_vs_minutes(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    mode: str = "role",  # "role" or "star"
    star_score_col: str | None = None,
    star_top_pct: float = 0.10,
    add_regression: bool = False,
    save_path: str | None = None,
    title: str | None = None,
    clip_q: float | None = None,
) -> None:
    d = df.copy()
    d = d.dropna(subset=[x_col, y_col])
    # Clip pour rendre l'échelle lisible (uniquement pour le plot)
    d = clip_middle_quantiles(d, [x_col, y_col], clip_q=clip_q or 1.0)


    if mode == "role":
        d["role"] = pd.qcut(
            d[x_col],
            q=4,
            labels=["Low minutes", "Rotation", "Starter", "Heavy minutes"],
        )
        groups = d.groupby("role", observed=True)
        legend_title = "Role (minutes proxy)"

    elif mode == "star":
        if star_score_col is None:
            raise ValueError("star_score_col must be provided when mode='star'")
        d = d.dropna(subset=[star_score_col])
        thr = d[star_score_col].quantile(1 - star_top_pct)
        d["group"] = np.where(d[star_score_col] >= thr, "Star", "Non-star")
        groups = d.groupby("group")
        legend_title = "Player type"

    else:
        raise ValueError("mode must be 'role' or 'star'")

    plt.figure(figsize=(10, 7))

    for name, g in groups:
        plt.scatter(
            g[x_col],
            g[y_col],
            label=str(name),
            alpha=0.6,
            s=25,
        )

    if add_regression and len(d) >= 2:
        x = d[x_col].to_numpy()
        y = d[y_col].to_numpy()
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        plt.plot(xs, m * xs + b, "k--", linewidth=2, label="Linear trend")

    plt.xlabel("Average minutes played")
    plt.ylabel(y_col.replace("_", " ").title())
    plt.title(title or "")
    plt.legend(title=legend_title)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
