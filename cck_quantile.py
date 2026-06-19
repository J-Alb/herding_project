"""
cck_quantile.py
---------------
Quantile regression version of the CCK herding test.

Instead of estimating the relationship at the mean (OLS), quantile
regression estimates it at each quantile tau of CSAD, allowing the
test to detect herding that concentrates in stress periods (high CSAD).

Model:
    Q_tau(CSAD_t) = alpha(tau) + beta1(tau)*|Rm_t| + beta2(tau)*Rm_t^2 + e_t

Key hypothesis:
    beta2(tau) becomes increasingly negative at higher quantiles
    -> herding intensifies during market stress

This connects directly to Essay 3 (volatility regime analysis):
the quantile regression captures regime heterogeneity continuously
rather than through discrete cutoffs.

Functions:
    fit_quantile_cck(csad_df, quantiles)   -> DataFrame of coefficients
    plot_quantile_betas(results_df)        -> coefficient paths across quantiles
    quantile_report(csad_df, market)       -> full table + plot

Usage in Jupyter:
    from cck_quantile import quantile_report
    from csad import compute_csad
    import numpy as np

    prices  = load_or_fetch_market("sp500", start="2015-01-01")
    returns = np.log(prices / prices.shift(1)).dropna(how="all")
    csad    = compute_csad(returns)
    quantile_report(csad, market="sp500")

Dependencies:
    pip install statsmodels matplotlib
"""

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=FutureWarning)

QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

def fit_quantile_cck(
    csad_df: pd.DataFrame,
    quantiles: list[float] = QUANTILES,
) -> pd.DataFrame:
    """
    Estimates CCK at each quantile of CSAD via statsmodels QuantReg.

    Parameters
    ----------
    csad_df   : DataFrame from compute_csad() — needs CSAD, abs_Rm, Rm_sq
    quantiles : list of quantiles to estimate (default: 0.10 to 0.90)

    Returns
    -------
    DataFrame with columns: quantile, param, coef, lower_ci, upper_ci, p_value
    """
    df = csad_df[["CSAD", "abs_Rm", "Rm_sq"]].dropna().copy()

    rows = []
    for q in quantiles:
        model  = smf.quantreg("CSAD ~ abs_Rm + Rm_sq", data=df)
        result = model.fit(q=q, max_iter=2000)

        for param in ["Intercept", "abs_Rm", "Rm_sq"]:
            name = {"Intercept": "alpha", "abs_Rm": "beta1", "Rm_sq": "beta2"}[param]
            ci   = result.conf_int().loc[param]
            rows.append({
                "quantile": q,
                "param":    name,
                "coef":     result.params[param],
                "lower_ci": ci[0],
                "upper_ci": ci[1],
                "p_value":  result.pvalues[param],
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_quantile_betas(
    results_df: pd.DataFrame,
    market: str = "",
    ols_coefs: dict | None = None,
) -> None:
    """
    Plots coefficient paths across quantiles with confidence bands.

    Parameters
    ----------
    results_df : DataFrame from fit_quantile_cck()
    market     : label for the plot title
    ols_coefs  : optional dict {"beta1": (coef, ci_lo, ci_hi),
                                "beta2": (coef, ci_lo, ci_hi)}
                 to overlay OLS estimates as horizontal reference lines
    """
    params = ["alpha", "beta1", "beta2"]
    labels = {
        "alpha": "alpha (intercept)",
        "beta1": "beta1  [ |Rm| ]",
        "beta2": "beta2  [ Rm^2 ]  <- herding test",
    }
    colors = {"alpha": "#888888", "beta1": "#1f77b4", "beta2": "#d62728"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    title = f"CCK Quantile Regression — {market.upper()}" if market else "CCK Quantile Regression"
    fig.suptitle(title, fontsize=13, y=1.01)

    for ax, param in zip(axes, params):
        sub = results_df[results_df["param"] == param].sort_values("quantile")
        q   = sub["quantile"].values
        c   = sub["coef"].values
        lo  = sub["lower_ci"].values
        hi  = sub["upper_ci"].values

        color = colors[param]
        ax.plot(q, c, "o-", color=color, linewidth=2, markersize=6, label="QR coef")
        ax.fill_between(q, lo, hi, alpha=0.2, color=color, label="95% CI")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

        # Overlay OLS as dashed horizontal line
        if ols_coefs and param in ols_coefs:
            oc, ol, oh = ols_coefs[param]
            ax.axhline(oc, color="gray", linewidth=1.2, linestyle=":", label="OLS")
            ax.axhspan(ol, oh, alpha=0.08, color="gray")

        ax.set_title(labels[param], fontsize=10)
        ax.set_xlabel("Quantile (tau)")
        ax.set_ylabel("Coefficient")
        ax.set_xticks(QUANTILES)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Annotate significance
        for i, row in sub.iterrows():
            stars = "***" if row["p_value"] < 0.01 else ("**" if row["p_value"] < 0.05 else ("*" if row["p_value"] < 0.10 else ""))
            if stars:
                ax.text(row["quantile"], row["coef"], stars,
                        ha="center", va="bottom", fontsize=8, color=color)

    fig.tight_layout()
    fname = f"cck_quantile_{market}.png" if market else "cck_quantile.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Saved: {fname}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def quantile_report(
    csad_df: pd.DataFrame,
    market: str = "",
    quantiles: list[float] = QUANTILES,
    ols_results=None,
) -> pd.DataFrame:
    """
    Estimates CCK at each quantile and prints a formatted table.

    Parameters
    ----------
    csad_df     : DataFrame from compute_csad()
    market      : label for display
    quantiles   : quantiles to estimate
    ols_results : optional OLS RegressionResults from cck_symmetric.fit_symmetric()
                  used to overlay OLS coefficients on the plot

    Returns
    -------
    DataFrame with all quantile estimates
    """
    label = market.upper() or "MARKET"
    n     = len(csad_df.dropna())
    sep   = "=" * 65

    print(f"\n{sep}")
    print(f"  CCK Quantile Regression — {label}  |  N={n}")
    print(f"  Model: Q_tau(CSAD) = alpha + beta1*|Rm| + beta2*Rm^2")
    print(sep)

    results = fit_quantile_cck(csad_df, quantiles=quantiles)

    # Print table: params as rows, quantiles as columns
    pivot = results.pivot(index="param", columns="quantile", values="coef")
    pvals = results.pivot(index="param", columns="quantile", values="p_value")

    print(f"\n  Coefficients (*** p<0.01  ** p<0.05  * p<0.10)\n")
    header = f"  {'param':<10s}" + "".join(f"  tau={q:.2f}" for q in quantiles)
    print(header)
    print(f"  {'-'*60}")

    for param in ["alpha", "beta1", "beta2"]:
        row = f"  {param:<10s}"
        for q in quantiles:
            coef = pivot.loc[param, q]
            p    = pvals.loc[param, q]
            stars = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else "  "))
            row += f"  {coef:>+8.5f}{stars:<3s}"
        print(row)

    # Highlight beta2 trend
    b2 = pivot.loc["beta2"]
    print(f"\n  beta2 path: {' -> '.join(f'{v:+.5f}' for v in b2.values)}")
    if b2.iloc[-1] < b2.iloc[0]:
        print(f"  >> beta2 decreases at higher quantiles — herding intensifies under stress")
    else:
        print(f"  >> beta2 does not decrease at higher quantiles — no quantile herding pattern")

    print(f"\n{sep}\n")

    # Build OLS overlay dict if provided
    ols_coefs = None
    if ols_results is not None:
        ci = ols_results.conf_int()
        ols_coefs = {
            "alpha": (ols_results.params["alpha"], ci.loc["alpha", 0], ci.loc["alpha", 1]),
            "beta1": (ols_results.params["beta1"], ci.loc["beta1", 0], ci.loc["beta1", 1]),
            "beta2": (ols_results.params["beta2"], ci.loc["beta2", 0], ci.loc["beta2", 1]),
        }

    plot_quantile_betas(results, market=market, ols_coefs=ols_coefs)

    return results


# ---------------------------------------------------------------------------
# Asymmetric quantile CCK (Tan et al. 2008 + quantile regression)
# ---------------------------------------------------------------------------

def fit_quantile_asymmetric(
    csad_df: pd.DataFrame,
    quantiles: list[float] = QUANTILES,
) -> pd.DataFrame:
    """
    Estimates the asymmetric CCK at each quantile of CSAD.

    Model:
        Q_tau(CSAD) = alpha
                    + beta1_up * D_up  * |Rm|
                    + beta1_dn * D_dn  * |Rm|
                    + beta2_up * D_up  * Rm^2
                    + beta2_dn * D_dn  * Rm^2

    Key comparison: beta2_dn(tau) vs beta2_up(tau) across quantiles.
    If beta2_dn turns negative at high quantiles -> herding on down days
    under stress, the theoretically expected pattern in emerging markets.

    Returns
    -------
    DataFrame with columns: quantile, param, coef, lower_ci, upper_ci, p_value
    """
    df = csad_df[["CSAD", "abs_Rm", "Rm_sq", "D_up", "D_down"]].dropna().copy()
    df["b1_up"] = df["D_up"]   * df["abs_Rm"]
    df["b1_dn"] = df["D_down"] * df["abs_Rm"]
    df["b2_up"] = df["D_up"]   * df["Rm_sq"]
    df["b2_dn"] = df["D_down"] * df["Rm_sq"]

    rows = []
    for q in quantiles:
        model  = smf.quantreg("CSAD ~ b1_up + b1_dn + b2_up + b2_dn", data=df)
        result = model.fit(q=q, max_iter=2000)

        rename = {
            "Intercept": "alpha",
            "b1_up": "beta1_up", "b1_dn": "beta1_dn",
            "b2_up": "beta2_up", "b2_dn": "beta2_dn",
        }
        for raw, name in rename.items():
            ci = result.conf_int().loc[raw]
            rows.append({
                "quantile": q,
                "param":    name,
                "coef":     result.params[raw],
                "lower_ci": ci[0],
                "upper_ci": ci[1],
                "p_value":  result.pvalues[raw],
            })

    return pd.DataFrame(rows)


def plot_quantile_asymmetric(
    results_df: pd.DataFrame,
    market: str = "",
) -> None:
    """
    Plots beta2_up and beta2_dn paths across quantiles on the same axes.
    The gap between the two lines shows how asymmetry evolves under stress.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    title = f"CCK Asymmetric Quantile Regression — {market.upper()}" if market else "CCK Asymmetric QR"
    fig.suptitle(title, fontsize=13, y=1.01)

    # Left: beta1_up vs beta1_dn
    ax = axes[0]
    for param, label, color in [("beta1_up", "beta1 UP", "#2ca02c"),
                                  ("beta1_dn", "beta1 DOWN", "#d62728")]:
        sub = results_df[results_df["param"] == param].sort_values("quantile")
        q, c = sub["quantile"].values, sub["coef"].values
        lo,hi = sub["lower_ci"].values, sub["upper_ci"].values
        ax.plot(q, c, "o-", color=color, linewidth=2, markersize=6, label=label)
        ax.fill_between(q, lo, hi, alpha=0.15, color=color)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.set_title("beta1  [ D*|Rm| ]  — magnitude response", fontsize=10)
    ax.set_xlabel("Quantile (tau)")
    ax.set_ylabel("Coefficient")
    ax.set_xticks(QUANTILES)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Right: beta2_up vs beta2_dn  (the herding test)
    ax = axes[1]
    for param, label, color in [("beta2_up", "beta2 UP", "#2ca02c"),
                                  ("beta2_dn", "beta2 DOWN", "#d62728")]:
        sub = results_df[results_df["param"] == param].sort_values("quantile")
        q, c = sub["quantile"].values, sub["coef"].values
        lo,hi = sub["lower_ci"].values, sub["upper_ci"].values
        ax.plot(q, c, "o-", color=color, linewidth=2, markersize=6, label=label)
        ax.fill_between(q, lo, hi, alpha=0.15, color=color)

        # Annotate significance
        for _, row in sub.iterrows():
            stars = "***" if row["p_value"] < 0.01 else ("**" if row["p_value"] < 0.05 else ("*" if row["p_value"] < 0.10 else ""))
            if stars:
                ax.text(row["quantile"], row["coef"], stars,
                        ha="center", va="bottom", fontsize=8, color=color)

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.set_title("beta2  [ D*Rm^2 ]  — herding test (negative = herding)", fontsize=10)
    ax.set_xlabel("Quantile (tau)")
    ax.set_ylabel("Coefficient")
    ax.set_xticks(QUANTILES)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fname = f"cck_quantile_asymmetric_{market}.png" if market else "cck_quantile_asymmetric.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Saved: {fname}")


def asymmetric_quantile_report(
    csad_df: pd.DataFrame,
    market: str = "",
    quantiles: list[float] = QUANTILES,
) -> pd.DataFrame:
    """
    Estimates asymmetric CCK at each quantile and prints a formatted table.

    Returns
    -------
    DataFrame with all quantile estimates
    """
    label = market.upper() or "MARKET"
    n     = len(csad_df.dropna())
    sep   = "=" * 70

    print(f"\n{sep}")
    print(f"  CCK Asymmetric Quantile Regression — {label}  |  N={n}")
    print(f"  Model: Q_tau(CSAD) = alpha + beta1_up*D+|Rm| + beta1_dn*D-|Rm|")
    print(f"                             + beta2_up*D+*Rm^2 + beta2_dn*D-*Rm^2")
    print(sep)

    results = fit_quantile_asymmetric(csad_df, quantiles=quantiles)
    pivot   = results.pivot(index="param", columns="quantile", values="coef")
    pvals   = results.pivot(index="param", columns="quantile", values="p_value")

    print(f"\n  Coefficients (*** p<0.01  ** p<0.05  * p<0.10)\n")
    header = f"  {'param':<12s}" + "".join(f"  tau={q:.2f}" for q in quantiles)
    print(header)
    print(f"  {'-'*65}")

    for param in ["alpha", "beta1_up", "beta1_dn", "beta2_up", "beta2_dn"]:
        row = f"  {param:<12s}"
        for q in quantiles:
            coef  = pivot.loc[param, q]
            p     = pvals.loc[param, q]
            stars = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else "  "))
            row  += f"  {coef:>+8.5f}{stars:<3s}"
        print(row)

    # Key finding: beta2_dn vs beta2_up across quantiles
    b2_up = pivot.loc["beta2_up"]
    b2_dn = pivot.loc["beta2_dn"]

    print(f"\n  beta2_up path: {' -> '.join(f'{v:+.5f}' for v in b2_up.values)}")
    print(f"  beta2_dn path: {' -> '.join(f'{v:+.5f}' for v in b2_dn.values)}")

    herding_dn = any(b2_dn.iloc[i] < 0 and pvals.loc["beta2_dn", q] < 0.05
                     for i, q in enumerate(quantiles))
    asym = b2_dn.iloc[-1] < b2_up.iloc[-1]

    if herding_dn:
        print(f"\n  >> Herding on DOWN days detected at high quantiles (beta2_dn < 0, p<0.05)")
    if asym:
        print(f"  >> Asymmetry confirmed: beta2_dn < beta2_up at tau=0.90 — stronger herding on down days")
    if not herding_dn and not asym:
        print(f"\n  >> No asymmetric herding pattern detected")

    print(f"\n{sep}\n")

    plot_quantile_asymmetric(results, market=market)

    return results


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    from pathlib import Path
    from fetch_market import load_or_fetch_market
    from csad import compute_csad
    from cck_symmetric import fit_symmetric

    DATA_DIR = str(Path(__file__).parent / "data" / "raw")
    START    = "2015-01-01"
    os.chdir(Path(__file__).parent)

    for market in ["sp500", "ibovespa"]:
        prices  = load_or_fetch_market(market, start=START, data_dir=DATA_DIR)
        returns = np.log(prices / prices.shift(1)).dropna(how="all")
        csad    = compute_csad(returns)
        ols     = fit_symmetric(csad)

        print(f"\n{'#'*65}")
        print(f"  {market.upper()}")
        print(f"{'#'*65}")

        res_sym = quantile_report(csad, market=market, ols_results=ols)
        res_sym.to_csv(f"{DATA_DIR}/cck_quantile_{market}_{START[:4]}.csv", index=False)

        res_asy = asymmetric_quantile_report(csad, market=market)
        res_asy.to_csv(f"{DATA_DIR}/cck_quantile_asymmetric_{market}_{START[:4]}.csv", index=False)
