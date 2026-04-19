"""
curve.py
========
Zero-Coupon Curve Bootstrapping from EUR Swap Rates
----------------------------------------------------
Module 1 of the rates-derivatives-pricer project.

Theory:
    A par swap of maturity T_N with fixed rate S_N is fairly priced when:
        S_N * sum_{i=1}^{N} delta_i * P(0, T_i) + P(0, T_N) = 1
    where:
        - delta_i  : day count fraction for period i (act/365)
        - P(0, T_i): discount factor from today to T_i

    Bootstrapping solves iteratively for P(0, T_N), given all P(0, T_i) for i < N.
    Once discount factors are known:
        r_zc(T) = -ln(P(0,T)) / T           [continuously compounded ZC rate]
        f(0,T)  = -d ln(P(0,T)) / dT        [instantaneous forward rate]

Author: Yanael Zohou
"""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ---------------------------------------------------------------------------
# 1. Market Data
# ---------------------------------------------------------------------------

def get_eur_swap_rates() -> pd.DataFrame:
    """
    Returns realistic EUR swap rates (annual fixed leg, Act/365).
    Representative of mid-2024 EUR IRS market.
    """
    data = {
        "maturity_years": [1, 2, 3, 5, 7, 10],
        "swap_rate":      [0.0370, 0.0340, 0.0320, 0.0310, 0.0308, 0.0310],
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# 2. Core Bootstrapping Engine
# ---------------------------------------------------------------------------

class ZeroCourveCurve:
    """
    Bootstraps a zero-coupon curve from par swap rates.

    Parameters
    ----------
    swap_df : pd.DataFrame
        Must contain columns 'maturity_years' and 'swap_rate'.
    day_count_convention : float
        Default: 1.0 (annual Act/365).
    interpolation : str
        'log_linear'   : log-linear on discount factors (guarantees f(t) > 0)
        'cubic_spline' : cubic spline on ZC rates (smoother forward curve)
    """

    def __init__(
        self,
        swap_df: pd.DataFrame,
        day_count_convention: float = 1.0,
        interpolation: str = "log_linear",
    ):
        self.swap_df       = swap_df.sort_values("maturity_years").reset_index(drop=True)
        self.delta         = day_count_convention
        self.interpolation = interpolation

        self.maturities       = self.swap_df["maturity_years"].values.astype(float)
        self.discount_factors = np.zeros(len(self.maturities))
        self.zc_rates         = np.zeros(len(self.maturities))

        self._interp_log_df = None
        self._interp_zc_cs  = None

        self._bootstrap()
        self._build_interpolator()

    # ------------------------------------------------------------------
    # 2a. Bootstrapping Loop
    # ------------------------------------------------------------------

    def _bootstrap(self) -> None:
        swap_rates = self.swap_df["swap_rate"].values

        # First pillar: P(0,1) = 1 / (1 + S_1 * delta)
        self.discount_factors[0] = 1.0 / (1.0 + swap_rates[0] * self.delta)

        for n in range(1, len(self.maturities)):
            T_n          = self.maturities[n]
            S_n          = swap_rates[n]
            annual_dates = np.arange(1, T_n + 1)

            df_at_annual = self._interpolate_intermediate_dfs(annual_dates, n)

            # Annuity = sum of discounted coupons (all except last)
            annuity = np.sum(self.delta * df_at_annual[:-1])

            # Solve for P(0, T_n)
            self.discount_factors[n] = (1.0 - S_n * annuity) / (1.0 + S_n * self.delta)

            if self.discount_factors[n] <= 0:
                raise ValueError(
                    f"Bootstrapping failure at {T_n}Y: "
                    f"negative discount factor {self.discount_factors[n]:.6f}."
                )

        # ZC rates from discount factors
        self.zc_rates = -np.log(self.discount_factors) / self.maturities

    def _interpolate_intermediate_dfs(
        self, annual_dates: np.ndarray, current_pillar_idx: int
    ) -> np.ndarray:
        df_out   = np.zeros(len(annual_dates))
        known_T  = self.maturities[:current_pillar_idx]
        known_df = self.discount_factors[:current_pillar_idx]

        for k, t in enumerate(annual_dates):
            if t in known_T:
                idx        = np.where(known_T == t)[0][0]
                df_out[k]  = known_df[idx]
            else:
                if t < known_T[0] or len(known_T) < 2:
                    df_out[k] = np.exp(-self.zc_rates[0] * t) if self.zc_rates[0] != 0 else 1.0
                elif t > known_T[-1]:
                    r_last    = -np.log(known_df[-1]) / known_T[-1]
                    df_out[k] = np.exp(-r_last * t)
                else:
                    i1         = np.searchsorted(known_T, t) - 1
                    i2         = i1 + 1
                    t1, t2     = known_T[i1], known_T[i2]
                    df1, df2   = known_df[i1], known_df[i2]
                    alpha      = (t - t1) / (t2 - t1)
                    log_df     = (1 - alpha) * np.log(df1) + alpha * np.log(df2)
                    df_out[k]  = np.exp(log_df)

        return df_out

    # ------------------------------------------------------------------
    # 2b. Interpolator
    # ------------------------------------------------------------------

    def _build_interpolator(self) -> None:
        log_dfs = np.log(self.discount_factors)
        if self.interpolation == "log_linear":
            self._log_dfs_pillars = log_dfs
        elif self.interpolation == "cubic_spline":
            self._interp_zc_cs = CubicSpline(
                self.maturities, self.zc_rates, bc_type="not-a-knot"
            )

    # ------------------------------------------------------------------
    # 2c. Public Interface
    # ------------------------------------------------------------------

    def get_discount_factor(self, T: float) -> float:
        """Returns interpolated discount factor P(0, T)."""
        if T <= 0:
            return 1.0
        if self.interpolation == "log_linear":
            log_df = np.interp(T, self.maturities, self._log_dfs_pillars)
            return float(np.exp(log_df))
        elif self.interpolation == "cubic_spline":
            r = float(self._interp_zc_cs(T))
            return float(np.exp(-r * T))
        raise ValueError(f"Unknown interpolation: {self.interpolation}")

    def get_zc_rate(self, T: float) -> float:
        """Returns continuously compounded ZC rate r_zc(T)."""
        df = self.get_discount_factor(T)
        if df <= 0 or T <= 0:
            return 0.0
        return float(-np.log(df) / T)

    def get_forward_rate(self, T1: float, T2: float) -> float:
        """
        Simply-compounded forward rate f(T1, T2).
        Used to price each Caplet in Module 2.
        f(T1,T2) = [P(0,T1)/P(0,T2) - 1] / (T2 - T1)
        """
        if T2 <= T1:
            raise ValueError("T2 must be strictly greater than T1.")
        df1 = self.get_discount_factor(T1)
        df2 = self.get_discount_factor(T2)
        return (df1 / df2 - 1.0) / (T2 - T1)

    def get_instantaneous_forward_rate(self, T: float, h: float = 1e-4) -> float:
        """
        Instantaneous forward f(0,T) by central finite differences:
        f(0,T) ≈ -[ln P(0,T+h) - ln P(0,T-h)] / (2h)
        """
        T_lo      = max(T - h, 1e-6)
        T_hi      = T + h
        log_df_lo = np.log(self.get_discount_factor(T_lo))
        log_df_hi = np.log(self.get_discount_factor(T_hi))
        return float(-(log_df_hi - log_df_lo) / (T_hi - T_lo))

    def to_dataframe(self, maturities: np.ndarray = None) -> pd.DataFrame:
        """Returns a summary DataFrame of the curve."""
        if maturities is None:
            maturities = self.maturities
        rows = []
        for T in maturities:
            rows.append({
                "Maturity (Y)"     : T,
                "Discount Factor"  : round(self.get_discount_factor(T), 8),
                "ZC Rate (%)"      : round(self.get_zc_rate(T) * 100, 4),
                "Inst. Forward (%)": round(self.get_instantaneous_forward_rate(T) * 100, 4),
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Visualisation
# ---------------------------------------------------------------------------

def plot_curves(curve: ZeroCourveCurve, fine_grid: np.ndarray = None) -> None:
    """2-panel figure: ZC rates vs swap rates + instantaneous forward curve."""
    if fine_grid is None:
        fine_grid = np.linspace(0.25, 10, 200)

    zc_fine   = np.array([curve.get_zc_rate(T) * 100 for T in fine_grid])
    fwd_fine  = np.array([curve.get_instantaneous_forward_rate(T) * 100 for T in fine_grid])
    zc_pil    = curve.zc_rates * 100
    swap_pil  = curve.swap_df["swap_rate"].values * 100
    T_pil     = curve.maturities

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("EUR Zero-Coupon Curve — Bootstrapped from Swap Rates",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(fine_grid, zc_fine, color="#1f77b4", lw=2, label="ZC Rate (continuous)")
    ax.scatter(T_pil, zc_pil, color="#1f77b4", zorder=5, s=60, label="ZC Pillars")
    ax.plot(T_pil, swap_pil, color="#d62728", lw=1.5,
            linestyle="--", marker="s", markersize=5, label="Par Swap Rate")
    ax.set_xlabel("Maturity (years)"); ax.set_ylabel("Rate (%)")
    ax.set_title("Zero-Coupon vs Par Swap Rates")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f%%"))
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xlim(0, 11)

    ax = axes[1]
    ax.plot(fine_grid, fwd_fine, color="#2ca02c", lw=2, label="Instantaneous Forward f(0,T)")
    ax.fill_between(fine_grid, fwd_fine, alpha=0.08, color="#2ca02c")
    ax.axhline(y=curve.zc_rates[-1]*100, color="grey", lw=1, linestyle=":",
               label=f"10Y ZC = {curve.zc_rates[-1]*100:.2f}%")
    ax.set_xlabel("Maturity (years)"); ax.set_ylabel("Rate (%)")
    ax.set_title("Instantaneous Forward Rate Curve")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f%%"))
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xlim(0, 11)

    plt.tight_layout()
    plt.savefig("../outputs/curve_output.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Figure saved to outputs/curve_output.png")


# ---------------------------------------------------------------------------
# 4. Standalone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    swap_df = get_eur_swap_rates()
    curve   = ZeroCourveCurve(swap_df, interpolation="log_linear")

    print("=== Bootstrapped Curve — Pillar Points ===")
    print(curve.to_dataframe().to_string(index=False))

    fine_grid = np.array([0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print("\n=== Interpolated Curve — Fine Grid ===")
    print(curve.to_dataframe(maturities=fine_grid).to_string(index=False))

    plot_curves(curve)