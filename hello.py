"""
Synthetic Bitcoin price paths around the log-log power law, for backtesting.

Model (log10 space):
    log10(P_t) = A + B * log10(days_since_genesis_t) + cycle_t + x_t

    trend  : the power law (A, B)
    cycle_t: optional ~4-year halving-cycle wave
    x_t    : mean-reverting residual (AR(1) / Ornstein-Uhlenbeck) with
             fat-tailed Student-t shocks

Why not plain white noise? Real residuals around the power law are strongly
autocorrelated: price stays above or below trend for months at a time.
White noise would produce absurd day-to-day jumps and no bull/bear regimes.
The AR(1) residual gives multi-month swings that pull back toward the trend.
"""

import argparse
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd

GENESIS = pd.Timestamp("2009-01-03")

# Commonly cited fit: P = 1.0117e-17 * days^5.82  (log10 A ≈ -16.995)
DEFAULT_A = -16.995
DEFAULT_B = 5.82
HISTORY_URL = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"
HISTORY_CACHE = Path(__file__).with_name("btc_daily.csv")


def load_bitcoin_history(cache_path=HISTORY_CACHE, refresh=False):
    """Load daily BTC/USD prices from Coin Metrics, caching Date/Close locally.

    Source: https://github.com/coinmetrics/data (CC BY-NC 4.0).
    Pass refresh=True to replace the cache with the latest available history.
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        frame = pd.read_csv(cache_path, usecols=["Date", "Close"])
    else:
        try:
            with urlopen(HISTORY_URL, timeout=60) as response:
                frame = pd.read_csv(response, usecols=["time", "PriceUSD"])
        except (URLError, OSError) as exc:
            raise RuntimeError(
                "Could not download Bitcoin history. Check your connection or "
                f"provide a Date,Close CSV at {cache_path}."
            ) from exc
        frame = frame.rename(columns={"time": "Date", "PriceUSD": "Close"})

    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
    frame = frame.dropna(subset=["Date", "Close"])
    frame = frame.loc[(frame["Date"] > GENESIS) & (frame["Close"] > 0)
                      & np.isfinite(frame["Close"])]
    frame = frame.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)
    if frame.empty:
        raise ValueError("Bitcoin history contains no valid positive daily prices.")
    frame.to_csv(cache_path, index=False)
    return frame


def days_since_genesis(dates):
    return (pd.DatetimeIndex(dates) - GENESIS).days.to_numpy().astype(float)


def power_law_log10(days, A=DEFAULT_A, B=DEFAULT_B):
    return A + B * np.log10(days)


# --------------------------------------------------------------------------
# Calibration from real data (recommended)
# --------------------------------------------------------------------------
def fit_from_prices(dates, prices, min_days=365):
    """
    Fit power law + AR(1) residual parameters from a real daily price history.
    Expects one price per day. Returns a dict you can pass to simulate_paths().
    """
    d = days_since_genesis(dates)
    p = np.asarray(prices, dtype=float)
    mask = (d >= min_days) & (p > 0)          # skip the very noisy early era
    d, p = d[mask], p[mask]

    lx, ly = np.log10(d), np.log10(p)
    B, A = np.polyfit(lx, ly, 1)
    r = np.corrcoef(lx, ly)[0, 1]

    resid = ly - (A + B * lx)
    phi = np.corrcoef(resid[:-1], resid[1:])[0, 1]          # daily AR(1) coef
    innovations = resid[1:] - phi * resid[:-1]
    sigma_stationary = resid.std()
    half_life = np.log(2) / -np.log(phi) if 0 < phi < 1 else np.inf

    # Fat-tail estimate from excess kurtosis of innovations: k = 6/(df-4)
    k = pd.Series(innovations).kurt()
    df = float(np.clip(4 + 6 / k, 2.5, 30)) if k > 0 else 30.0

    return dict(A=A, B=B, r=r, phi=phi, sigma_stationary=sigma_stationary,
                half_life_days=half_life, t_df=df, last_resid=resid[-1])


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
def simulate_paths(
    start_date="2026-10-21",
    n_days=365 * 4,
    n_paths=500,
    A=DEFAULT_A,
    B=DEFAULT_B,
    sigma_stationary=0.25,   # long-run std of log10 residual (0.25 ≈ ×/÷ 1.8)
    half_life_days=600,      # how fast deviations decay back toward trend
    t_df=4.0,                # Student-t degrees of freedom (lower = fatter tails)
    start_resid=0.0,         # where today's price sits vs trend, in log10
    cycle_amp=0.0,           # optional halving-cycle amplitude in log10 (e.g. 0.15)
    cycle_period=1461,       # ~4 years in days
    cycle_phase_date="2024-04-20",  # a halving date anchors the wave
    cycle_peak_lag=500,      # days after halving that the wave peaks
    seed=None,
    **_ignored,
):
    """
    Returns a DataFrame (index = dates, columns = path_0..path_{n-1}) of prices.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, periods=n_days, freq="D")
    days = days_since_genesis(dates)

    trend = power_law_log10(days, A, B)

    if cycle_amp:
        t = (dates - pd.Timestamp(cycle_phase_date)).days.to_numpy()
        cycle = cycle_amp * np.cos(2 * np.pi * (t - cycle_peak_lag) / cycle_period)
    else:
        cycle = np.zeros(n_days)

    phi = np.exp(-np.log(2) / half_life_days)
    innov_sd = sigma_stationary * np.sqrt(1 - phi**2)

    # Student-t shocks scaled to unit variance
    shocks = rng.standard_t(t_df, size=(n_days, n_paths))
    if t_df > 2:
        shocks *= np.sqrt((t_df - 2) / t_df)
    shocks *= innov_sd

    x = np.empty((n_days, n_paths))
    x[0] = start_resid
    for i in range(1, n_days):
        x[i] = phi * x[i - 1] + shocks[i]

    log_price = trend[:, None] + cycle[:, None] + x
    cols = [f"path_{j}" for j in range(n_paths)]
    return pd.DataFrame(10 ** log_price, index=dates, columns=cols)


def summarize(paths):
    """Quick sanity checks so you can compare against real BTC stats."""
    rets = np.log(paths).diff().dropna()
    daily_vol = rets.std().mean()
    dd = (paths / paths.cummax() - 1).min()
    print(f"Mean annualized vol : {daily_vol * np.sqrt(365):.1%}")
    print(f"Median max drawdown : {dd.median():.1%}")
    print(f"Worst max drawdown  : {dd.min():.1%}")
    final = paths.iloc[-1]
    print(f"Final price  5/50/95 pct: "
          f"${final.quantile(.05):,.0f} / ${final.median():,.0f} / ${final.quantile(.95):,.0f}")


def plot_paths(paths, n_show=20, A=DEFAULT_A, B=DEFAULT_B, history=None, fname=None):
    import matplotlib
    if fname:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    d_sim = days_since_genesis(paths.index)

    # Left: full log-log view
    for c in paths.columns[:n_show]:
        ax1.plot(d_sim, paths[c], lw=0.7, alpha=0.6)
    if history is not None:                       # (dates, prices) tuple
        ax1.plot(days_since_genesis(history[0]), history[1], "k", lw=1, label="Actual (Coin Metrics)")
    d_all = np.logspace(np.log10(365), np.log10(d_sim.max()), 500)
    ax1.plot(d_all, 10 ** power_law_log10(d_all, A, B), "r--", lw=1.5, label="Power law")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("Days since genesis (log)"); ax1.set_ylabel("Price USD (log)")
    ax1.set_title("Log-log view"); ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)

    # Right: zoom on the simulated window with percentile bands
    if history is not None:
        history_dates = pd.DatetimeIndex(history[0])
        recent = history_dates >= paths.index[0] - pd.Timedelta(days=365)
        ax2.plot(history_dates[recent], np.asarray(history[1])[recent], "k", lw=1,
                 label="Actual (Coin Metrics)")
    q = paths.quantile([0.05, 0.25, 0.5, 0.75, 0.95], axis=1).T
    ax2.fill_between(paths.index, q[0.05], q[0.95], alpha=0.15, color="C0", label="5–95%")
    ax2.fill_between(paths.index, q[0.25], q[0.75], alpha=0.3, color="C0", label="25–75%")
    for c in paths.columns[:5]:
        ax2.plot(paths.index, paths[c], lw=0.8, alpha=0.8)
    ax2.plot(paths.index, 10 ** power_law_log10(d_sim, A, B), "r--", lw=1.5, label="Power law")
    ax2.set_yscale("log"); ax2.set_ylabel("Price USD (log)")
    ax2.set_title("Recent history and simulated paths"); ax2.legend(loc="upper left")
    ax2.grid(True, which="both", alpha=0.3)

    fig.suptitle("Synthetic BTC paths around power law")
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=120)
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot historical BTC/USD prices and synthetic paths.")
    parser.add_argument("--refresh", action="store_true", help="Download fresh Bitcoin history.")
    args = parser.parse_args()
    history = load_bitcoin_history(refresh=args.refresh)
    last_date = history["Date"].iloc[-1]
    last_price = history["Close"].iloc[-1]
    print(f"Loaded {len(history):,} daily BTC/USD prices: "
          f"{history['Date'].iloc[0]:%Y-%m-%d} to {last_date:%Y-%m-%d} (Coin Metrics)")

    # Anchor the first simulated price to the latest observation, including the cycle.
    cycle_amp = 0.15
    cycle_at_start = cycle_amp * np.cos(
        2 * np.pi * ((last_date - pd.Timestamp("2024-04-20")).days - 500) / 1461
    )
    start_resid = (np.log10(last_price)
                   - power_law_log10(days_since_genesis([last_date]))[0] - cycle_at_start)
    paths = simulate_paths(start_date=last_date, start_resid=start_resid,
                           n_paths=500, cycle_amp=cycle_amp, seed=42)
    summarize(paths)
    plot_paths(paths, history=(history["Date"], history["Close"]),
               fname="btc_powerlaw_paths.png")

    # Feed each column into your backtester, e.g.:
    # results = [my_strategy(paths[c]) for c in paths.columns]
