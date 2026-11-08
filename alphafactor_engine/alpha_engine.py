#!/usr/bin/env python3
"""Vectorized multi-factor backtesting and risk attribution engine.

Evaluates cross-sectional equity factors (Momentum, Value, Quality, Low
Volatility) on a curated Nifty 50/100 universe with monthly quintile
rebalancing, a turnover-based friction model, Information Coefficient analysis
and an interactive Plotly report.

Usage:
    python alpha_engine.py
    python alpha_engine.py --weighting inverse_vol --cost-bps 30
    python alpha_engine.py --factor-weights 0.4 0.3 0.2 0.1 --years 5

Point-in-time rules (see ``FactorEngine``):
    * Fundamentals become visible only ``fundamental_lag_days`` after fiscal
      year-end. Periods Yahoo does not supply stay missing rather than being
      back-filled from later data, and the composite re-weights across the
      factors available at each rebalance.
    * A signal computed at the close of date t trades at the close of
      t + ``execution_lag`` and earns returns only after that.
"""

from __future__ import annotations

import argparse
import html
import logging
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore", message="The inplace keyword")

LOGGER = logging.getLogger("alpha_engine")

Weighting = Literal["equal", "inverse_vol"]

TRADING_DAYS: int = 252
CALENDAR_DAYS: float = 365.25
FACTOR_NAMES: tuple[str, ...] = ("momentum", "quality", "value", "low_vol")
MARKET_FACTOR: str = "market"
SPECIFIC_SOURCE: str = "specific"
COST_SOURCE: str = "costs"

DEFAULT_UNIVERSE: tuple[str, ...] = (
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HCLTECH.NS",
    "WIPRO.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "KOTAKBANK.NS",
    "AXISBANK.NS",
    "SBIN.NS",
    "BAJFINANCE.NS",
    "LT.NS",
    "ITC.NS",
    "HINDUNILVR.NS",
    "NESTLEIND.NS",
    "BHARTIARTL.NS",
    "MARUTI.NS",
    "M&M.NS",
    "SUNPHARMA.NS",
    "DRREDDY.NS",
    "CIPLA.NS",
    "TITAN.NS",
    "TRENT.NS",
    "ASIANPAINT.NS",
    "ULTRACEMCO.NS",
    "NTPC.NS",
    "POWERGRID.NS",
    "TATASTEEL.NS",
    "JSWSTEEL.NS",
    "COALINDIA.NS",
)

# Debt-to-equity is not a meaningful quality signal for lenders.
DEFAULT_FINANCIALS: frozenset[str] = frozenset(
    {
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "KOTAKBANK.NS",
        "AXISBANK.NS",
        "SBIN.NS",
        "BAJFINANCE.NS",
    }
)

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
COLOR_Q1 = "#2a78d6"
COLOR_Q5 = "#eb6834"
COLOR_LONG_SHORT = "#1baf7a"
COLOR_BENCHMARK = INK_MUTED
COLOR_NEGATIVE = "#e34948"


@dataclass(frozen=True)
class EngineConfig:
    """Immutable parameters for one pipeline run."""

    universe: tuple[str, ...] = DEFAULT_UNIVERSE
    financials: frozenset[str] = DEFAULT_FINANCIALS
    benchmark: str = "^NSEI"
    years: int = 5
    end_date: date = field(default_factory=date.today)
    warmup_calendar_days: int = 420

    factor_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "momentum": 0.40,
            "quality": 0.30,
            "value": 0.20,
            "low_vol": 0.10,
        }
    )
    min_composite_weight: float = 0.50
    long_window: int = 252
    skip_window: int = 21
    inverse_vol_window: int = 63
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99

    fundamental_lag_days: int = 90
    fundamental_max_staleness: int = 378
    ffill_limit: int = 5
    min_ticker_coverage: float = 0.80
    min_names: int = 10
    min_median_traded_value_inr: float = 1e8

    quintiles: int = 5
    cost_bps: float = 15.0
    weighting: Weighting = "equal"
    execution_lag: int = 1
    risk_free_rate: float = 0.067
    rolling_sharpe_window: int = 126
    risk_half_life_days: int = 90
    output: Path = Path("factor_backtest_report.html")

    @property
    def backtest_start(self) -> pd.Timestamp:
        """First calendar date eligible for a rebalance."""
        return pd.Timestamp(self.end_date) - pd.DateOffset(years=self.years)

    @property
    def download_start(self) -> pd.Timestamp:
        """First calendar date downloaded, including the factor warm-up."""
        return self.backtest_start - pd.Timedelta(
            days=self.warmup_calendar_days
        )


@dataclass(frozen=True)
class MarketData:
    """Aligned market panels plus sparse point-in-time fundamentals."""

    adj_close: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    benchmark: pd.Series
    net_income: pd.DataFrame
    equity: pd.DataFrame
    total_debt: pd.DataFrame
    shares: pd.Series
    repairs: tuple[str, ...] = ()


@dataclass(frozen=True)
class FactorSet:
    """Normalized factor scores sampled at rebalance dates."""

    scores: dict[str, pd.DataFrame]
    composite: pd.DataFrame
    rebalance_dates: pd.DatetimeIndex


@dataclass(frozen=True)
class LegResult:
    """Daily results for one quintile portfolio.

    ``holdings`` are the start-of-day weights that earn each day's return.
    """

    gross: pd.Series
    cost: pd.Series
    turnover: pd.Series
    holdings: pd.DataFrame

    @property
    def net(self) -> pd.Series:
        """Daily return after transaction costs."""
        return self.gross - self.cost


@dataclass(frozen=True)
class BacktestResult:
    """Output of a quintile backtest."""

    legs: dict[int, LegResult]
    long_short_net: pd.Series
    long_short_gross: pd.Series
    benchmark: pd.Series
    quintiles: pd.DataFrame


class DataIngestion:
    """Downloads and cleans prices, volumes and annual fundamentals."""

    INCOME_LABELS: tuple[str, ...] = (
        "Net Income Common Stockholders",
        "Net Income",
    )
    EQUITY_LABELS: tuple[str, ...] = (
        "Common Stock Equity",
        "Stockholders Equity",
    )
    DEBT_LABELS: tuple[str, ...] = ("Total Debt",)
    SHARES_LABELS: tuple[str, ...] = ("Ordinary Shares Number",)
    MIN_SPLIT_DEVIATION: float = 0.20
    SPLIT_GAP_TOLERANCE: float = 0.02
    MAX_PLAUSIBLE_EARNINGS_YIELD: float = 0.50
    SUSPECT_DAILY_MOVE: float = 0.25

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.repairs: list[str] = []

    def run(self) -> MarketData:
        """Execute the full ingestion pipeline."""
        raw = self._download_prices()
        adj, close, volume, benchmark = self._clean(raw)
        tickers = list(adj.columns)
        LOGGER.info("Fetching fundamentals for %d tickers", len(tickers))
        net_income, equity, debt, shares = self._download_fundamentals(
            tickers,
            close,
        )
        return MarketData(
            adj,
            close,
            volume,
            benchmark,
            net_income,
            equity,
            debt,
            shares,
            tuple(self.repairs),
        )

    def _download_prices(self) -> pd.DataFrame:
        cfg = self.config
        tickers = [*cfg.universe, cfg.benchmark]
        LOGGER.info(
            "Downloading %d tickers from %s to %s",
            len(tickers),
            cfg.download_start.date(),
            cfg.end_date,
        )
        raw = yf.download(
            tickers,
            start=cfg.download_start.strftime("%Y-%m-%d"),
            end=(cfg.end_date + timedelta(days=1)).isoformat(),
            auto_adjust=False,
            actions=True,
            progress=False,
            threads=True,
        )
        if raw is None or raw.empty:
            raise RuntimeError("Yahoo Finance returned no price data.")
        raw.index = pd.DatetimeIndex(raw.index).tz_localize(None).normalize()
        return raw[~raw.index.duplicated(keep="last")].sort_index()

    def _clean(
        self,
        raw: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
        """Align to the benchmark calendar, fill gaps, drop bad names."""
        cfg = self.config
        adj, close, volume = self._repair_split_breaks(
            raw["Adj Close"].where(raw["Adj Close"] > 0),
            raw["Close"].where(raw["Close"] > 0),
            raw["Volume"],
            raw["Open"].where(raw["Open"] > 0),
            raw["Stock Splits"],
        )

        if cfg.benchmark not in adj.columns or adj[cfg.benchmark].isna().all():
            raise RuntimeError(
                f"Benchmark {cfg.benchmark} failed to download."
            )

        stocks = [t for t in cfg.universe if t in adj.columns]
        has_quorum = adj[stocks].notna().mean(axis=1) >= 0.5
        calendar = adj.index[adj[cfg.benchmark].notna() & has_quorum]

        coverage = adj.loc[calendar, stocks].notna().mean()
        kept = coverage.index[coverage >= cfg.min_ticker_coverage].tolist()
        dropped = sorted(set(cfg.universe) - set(kept))
        if dropped:
            LOGGER.warning("Dropped for insufficient history: %s", dropped)
        if len(kept) < cfg.min_names:
            raise RuntimeError(
                f"Only {len(kept)} tickers usable; need {cfg.min_names}."
            )

        filled_cells = int(
            adj.loc[calendar, kept]
            .ffill(limit=cfg.ffill_limit)
            .notna()
            .sum()
            .sum()
            - adj.loc[calendar, kept].notna().sum().sum()
        )
        LOGGER.info(
            "Clean panel: %d sessions, %d tickers, %d forward-filled cells",
            len(calendar),
            len(kept),
            filled_cells,
        )
        moves = adj.loc[calendar, kept].pct_change(fill_method=None).stack()
        suspects = moves[moves.abs() >= self.SUSPECT_DAILY_MOVE]
        for (day, ticker), move in suspects.items():
            LOGGER.warning(
                "Suspect daily move: %s %+.1f%% on %s",
                ticker,
                move * 100,
                day.date(),
            )
        return (
            adj.loc[calendar, kept].ffill(limit=cfg.ffill_limit),
            close.loc[calendar, kept].ffill(limit=cfg.ffill_limit),
            volume.reindex(index=calendar, columns=kept).fillna(0.0),
            adj.loc[calendar, cfg.benchmark].ffill(limit=cfg.ffill_limit),
        )

    def _repair_split_breaks(
        self,
        adj: pd.DataFrame,
        close: pd.DataFrame,
        volume: pd.DataFrame,
        open_: pd.DataFrame,
        splits: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Undo partial split adjustment in Yahoo's price history.

        Yahoo sometimes applies a split factor only part-way back, leaving a
        false price cliff. A break is an overnight gap matching a recorded
        split factor, up to a year before that split. Earlier history is
        rescaled by the factor so the series is continuous.
        """
        adj, close, volume = adj.copy(), close.copy(), volume.astype(float)
        events = splits.where(splits > 0).stack()
        events = events[(events - 1.0).abs() >= self.MIN_SPLIT_DEVIATION]
        gap = open_ / close.ffill().shift(1)
        for (split_date, ticker), factor in events.items():
            search = gap.loc[
                split_date - pd.DateOffset(years=1) : split_date, ticker
            ]
            matched = (search * factor - 1.0).abs() < self.SPLIT_GAP_TOLERANCE
            for break_date in search.index[matched]:
                earlier = adj.index < break_date
                adj.loc[earlier, ticker] /= factor
                close.loc[earlier, ticker] /= factor
                volume.loc[earlier, ticker] *= factor
                message = (
                    f"{ticker}: rescaled prices before {break_date.date()} "
                    f"by {factor:g}x for a partial split adjustment"
                )
                LOGGER.warning("Repaired price history. %s", message)
                self.repairs.append(message)
        return adj, close, volume

    def _download_fundamentals(
        self,
        tickers: Sequence[str],
        close: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
        with ThreadPoolExecutor(max_workers=4) as pool:
            records = dict(zip(tickers, pool.map(self._fetch_one, tickers)))

        def panel(item: str) -> pd.DataFrame:
            frame = pd.DataFrame(
                {ticker: rec[item] for ticker, rec in records.items()},
            ).sort_index()
            frame.index = pd.DatetimeIndex(frame.index)
            return frame.reindex(columns=list(tickers))

        shares = pd.Series(
            {ticker: rec["shares"] for ticker, rec in records.items()},
            dtype=float,
        )
        missing = shares.index[shares.isna()].tolist()
        if missing:
            LOGGER.warning(
                "No share count, value unavailable for: %s", missing
            )
        net_income, equity, debt = (
            panel("net_income"),
            panel("equity"),
            panel("debt"),
        )
        currencies = {
            ticker: str(rec["currency"])
            for ticker, rec in records.items()
            if rec["currency"] not in (None, "INR")
        }
        market_cap = close.iloc[-1].mul(shares)
        for ticker, currency in currencies.items():
            self._convert_to_inr(
                ticker,
                currency,
                (net_income, equity, debt),
                float(market_cap[ticker]),
            )
        return net_income, equity, debt, shares

    def _convert_to_inr(
        self,
        ticker: str,
        currency: str,
        panels: Sequence[pd.DataFrame],
        market_cap: float,
    ) -> None:
        """Convert one ticker's statements to INR at availability-date FX.

        Yahoo's reporting-currency flag is unreliable for Indian listings: it
        can say USD for statements actually in INR. Conversion therefore runs
        only if the resulting earnings yield is plausible.
        """
        rates = self._fx_rates(currency)
        if rates.empty:
            for frame in panels:
                frame[ticker] = np.nan
            LOGGER.warning(
                "No %sINR rate; dropped %s fundamentals", currency, ticker
            )
            return
        latest_income = panels[0][ticker].dropna()
        if latest_income.empty:
            return
        converted_yield = (
            abs(float(latest_income.iloc[-1]))
            * float(rates.iloc[-1])
            / market_cap
        )
        if converted_yield > self.MAX_PLAUSIBLE_EARNINGS_YIELD:
            LOGGER.info(
                "%s flagged %s but statements are already in INR",
                ticker,
                currency,
            )
            return
        for frame in panels:
            frame[ticker] *= rates.reindex(frame.index, method="ffill")
        message = f"{ticker}: converted {currency} statements to INR"
        LOGGER.warning("Fundamentals adjusted. %s", message)
        self.repairs.append(message)

    def _fx_rates(self, currency: str) -> pd.Series:
        """Daily INR per unit of ``currency``; empty if unavailable."""
        cfg = self.config
        frame = yf.download(
            f"{currency}INR=X",
            start=cfg.download_start.strftime("%Y-%m-%d"),
            end=(cfg.end_date + timedelta(days=1)).isoformat(),
            auto_adjust=False,
            progress=False,
        )
        if frame is None or frame.empty:
            return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        rates = frame["Close"].squeeze("columns").dropna()
        rates.index = (
            pd.DatetimeIndex(rates.index).tz_localize(None).normalize()
        )
        return rates[~rates.index.duplicated(keep="last")].sort_index()

    def _fetch_one(
        self, ticker: str
    ) -> dict[str, pd.Series | float | str | None]:
        """Fetch one ticker's statements, keyed by availability date."""
        empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        record: dict[str, pd.Series | float | str | None] = {
            "net_income": empty,
            "equity": empty,
            "debt": empty,
            "shares": np.nan,
            "currency": None,
        }
        try:
            handle = yf.Ticker(ticker)
            income, balance = handle.income_stmt, handle.balance_sheet
            record["net_income"] = self._line_item(income, self.INCOME_LABELS)
            record["equity"] = self._line_item(balance, self.EQUITY_LABELS)
            record["debt"] = self._line_item(balance, self.DEBT_LABELS)
            history = self._line_item(balance, self.SHARES_LABELS)
            info = handle.info
            record["currency"] = info.get("financialCurrency")
            shares = info.get("sharesOutstanding")
            if shares:
                record["shares"] = float(shares)
            elif not history.empty:
                record["shares"] = float(history.sort_index().iloc[-1])
        except Exception as exc:  # yfinance raises many undocumented types
            LOGGER.warning("Fundamentals failed for %s: %s", ticker, exc)
        return record

    def _line_item(
        self,
        statement: pd.DataFrame,
        labels: Sequence[str],
    ) -> pd.Series:
        """Return the first populated label, keyed by availability date."""
        lag = pd.Timedelta(days=self.config.fundamental_lag_days)
        for label in labels:
            if statement is not None and label in statement.index:
                series = pd.to_numeric(
                    statement.loc[label],
                    errors="coerce",
                ).dropna()
                if not series.empty:
                    series.index = pd.DatetimeIndex(series.index) + lag
                    return series[~series.index.duplicated()].sort_index()
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))


class FactorEngine:
    """Builds, normalizes and blends cross-sectional alpha factors.

    Every score is oriented so that a higher value is more attractive.
    """

    def __init__(self, data: MarketData, config: EngineConfig) -> None:
        self.data = data
        self.config = config
        self.returns: pd.DataFrame = data.adj_close.pct_change(
            fill_method=None
        )
        self.benchmark_returns: pd.Series = data.benchmark.pct_change(
            fill_method=None,
        )

    def rebalance_dates(self) -> pd.DatetimeIndex:
        """Last session of each completed month inside the backtest window."""
        index = self.data.adj_close.index
        dates = pd.DatetimeIndex(
            pd.Series(index, index=index).groupby(index.to_period("M")).last(),
        )
        if index[-1] < index[-1] + pd.offsets.BMonthEnd(0):
            dates = dates[:-1]  # the final month has not finished trading
        return dates[dates >= self.config.backtest_start]

    def momentum(self) -> pd.DataFrame:
        """12M-1M price return, skipping the most recent month."""
        cfg = self.config
        prices = self.data.adj_close
        return (
            prices.shift(cfg.skip_window) / prices.shift(cfg.long_window) - 1.0
        )

    def earnings_yield(self) -> pd.DataFrame:
        """Point-in-time net income over split-consistent market cap.

        Market cap uses Yahoo's split-adjusted (not dividend-adjusted) close
        and the latest share count. This keeps the ratio correct through
        splits and bonuses, unlike reported per-share EPS.
        """
        market_cap = self.data.close.mul(self.data.shares, axis=1)
        return self._point_in_time(self.data.net_income) / market_cap

    def value(self) -> pd.DataFrame:
        """Cross-sectional percentile rank of earnings yield."""
        return self.earnings_yield().rank(axis=1, pct=True)

    def return_stability(self) -> pd.DataFrame:
        """Signed R-squared of log price against time over the long window."""
        log_price = np.log(self.data.adj_close)
        time_index = pd.Series(
            np.arange(len(log_price), dtype=float),
            index=log_price.index,
        )
        corr = log_price.rolling(self.config.long_window).corr(time_index)
        return corr * corr.abs()

    def return_on_equity(self) -> pd.DataFrame:
        """Point-in-time ROE; negative book equity is treated as missing."""
        equity = self._point_in_time(self.data.equity)
        return self._point_in_time(self.data.net_income) / equity.where(
            equity > 0
        )

    def low_leverage(self) -> pd.DataFrame:
        """Negative debt-to-equity for non-financial companies only."""
        equity = self._point_in_time(self.data.equity)
        leverage = -self._point_in_time(self.data.total_debt) / equity.where(
            equity > 0
        )
        non_financial = ~leverage.columns.isin(list(self.config.financials))
        return leverage.loc[:, non_financial].reindex(columns=leverage.columns)

    def low_volatility(self) -> pd.DataFrame:
        """Negative rolling idiosyncratic volatility versus the benchmark.

        Uses the OLS identity Var(residual) = Var(r) - Cov(r, m)^2 / Var(m).
        """
        window = self.config.long_window
        rolling = self.returns.rolling(window)
        market_variance = self.benchmark_returns.rolling(window).var()
        covariance = rolling.cov(self.benchmark_returns)
        explained = (covariance**2).div(market_variance, axis=0)
        return -np.sqrt((rolling.var() - explained).clip(lower=0.0))

    def eligibility(self) -> pd.DataFrame:
        """Mask of names with adequate median traded value."""
        cfg = self.config
        traded_value = self.data.close * self.data.volume
        median_value = traded_value.rolling(cfg.inverse_vol_window).median()
        return median_value >= cfg.min_median_traded_value_inr

    def winsorize(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Clip each row at the configured cross-sectional percentiles."""
        cfg = self.config
        return frame.clip(
            lower=frame.quantile(cfg.winsor_lower, axis=1),
            upper=frame.quantile(cfg.winsor_upper, axis=1),
            axis=0,
        )

    @staticmethod
    def zscore(frame: pd.DataFrame) -> pd.DataFrame:
        """Standardize each row: Z = (X - mean) / std."""
        std = frame.std(axis=1).replace(0.0, np.nan)
        return frame.sub(frame.mean(axis=1), axis=0).div(std, axis=0)

    def normalize(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Winsorize then z-score, cross-sectionally."""
        return self.zscore(self.winsorize(frame))

    def composite(self, scores: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        """Weighted blend, re-weighted across each name's available factors."""
        cfg = self.config
        numerator: pd.DataFrame | float = 0.0
        denominator: pd.DataFrame | float = 0.0
        for name, weight in cfg.factor_weights.items():
            numerator = numerator + scores[name].fillna(0.0) * weight
            denominator = denominator + scores[name].notna() * weight
        blended = numerator / denominator.where(
            denominator >= cfg.min_composite_weight,
        )
        counts = blended.notna().sum(axis=1)
        return self.zscore(blended.where(counts >= cfg.min_names))

    def build(self) -> FactorSet:
        """Compute every normalized factor and the composite score."""
        dates = self.rebalance_dates()
        if dates.empty:
            raise RuntimeError("No completed rebalance dates in the window.")
        eligible = self.eligibility().reindex(dates)

        def prepare(frame: pd.DataFrame) -> pd.DataFrame:
            return frame.reindex(dates).where(eligible)

        quality_parts = pd.concat(
            [
                self.normalize(prepare(self.return_stability())),
                self.normalize(prepare(self.return_on_equity())),
                self.normalize(prepare(self.low_leverage())),
            ],
            keys=["stability", "roe", "leverage"],
        )
        raw: dict[str, pd.DataFrame] = {
            "momentum": prepare(self.momentum()),
            "quality": quality_parts.groupby(level=1).mean().reindex(dates),
            "value": prepare(self.value()),
            "low_vol": prepare(self.low_volatility()),
        }
        scores = {name: self.normalize(frame) for name, frame in raw.items()}
        return FactorSet(scores, self.composite(scores), dates)

    def _point_in_time(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Expand availability-dated values onto the trading calendar."""
        calendar = self.data.adj_close.index
        columns = self.data.adj_close.columns
        if panel.empty:
            return pd.DataFrame(np.nan, index=calendar, columns=columns)
        combined = panel.reindex(panel.index.union(calendar))
        filled = combined.ffill(limit=self.config.fundamental_max_staleness)
        return filled.reindex(index=calendar, columns=columns)


class FactorBacktester:
    """Monthly quintile backtest with drifting weights and turnover costs."""

    def __init__(
        self,
        data: MarketData,
        factors: FactorSet,
        config: EngineConfig,
    ) -> None:
        self.data = data
        self.factors = factors
        self.config = config
        self.returns: pd.DataFrame = data.adj_close.pct_change(
            fill_method=None
        )

    def assign_quintiles(self) -> pd.DataFrame:
        """Rank each rebalance into quintiles; 1 holds the highest scores."""
        count = self.config.quintiles
        composite = self.factors.composite.dropna(how="all")
        if composite.empty:
            raise RuntimeError(
                "Composite score is empty; check factor inputs."
            )
        ranks = composite.rank(
            axis=1, method="first", ascending=False, pct=True
        )
        return np.ceil((ranks * count).round(9)).clip(1, count)

    def target_weights(
        self, quintiles: pd.DataFrame, quintile: int
    ) -> pd.DataFrame:
        """Weights inside one quintile; non-members are explicitly zero."""
        members = quintiles.where(quintiles == quintile).notna()
        if self.config.weighting == "inverse_vol":
            volatility = (
                self.returns.rolling(self.config.inverse_vol_window)
                .std()
                .reindex(quintiles.index)
            )
            median = volatility.median(axis=1)
            raw = 1.0 / volatility.T.fillna(median).T.replace(0.0, np.nan)
            raw = raw.where(members)
        elif self.config.weighting == "equal":
            raw = members.astype(float).where(members)
        else:
            raise ValueError(f"Unknown weighting: {self.config.weighting}")
        return raw.div(raw.sum(axis=1), axis=0).fillna(0.0)

    def simulate(self, target: pd.DataFrame) -> LegResult:
        """Simulate a portfolio with buy-and-hold drift between rebalances.

        Between rebalances each position grows with its own price, so
        ``holdings = target * price / price_at_last_rebalance``. Turnover
        compares the new target against the drifted pre-trade weights.
        """
        prices = self.data.adj_close
        calendar = prices.index
        positions = (
            calendar.get_indexer(target.index) + self.config.execution_lag
        )
        tradable = positions < len(calendar)
        schedule = target.loc[tradable].copy()
        trade_dates = calendar[positions[tradable]]
        schedule.index = trade_dates

        weights = schedule.reindex(calendar).ffill().fillna(0.0)
        anchors = prices.loc[trade_dates].reindex(calendar).ffill()
        post_trade = self._normalized(weights * prices / anchors)
        pre_trade = self._normalized(
            weights.shift(1) * prices / anchors.shift(1)
        )

        holdings = post_trade.shift(1).fillna(0.0)
        gross = (holdings * self.returns).sum(axis=1)
        is_trade = pd.Series(calendar.isin(trade_dates), index=calendar)
        turnover = 0.5 * (weights - pre_trade).abs().sum(axis=1).where(
            is_trade, 0.0
        )
        cost = turnover * self.config.cost_bps / 10_000.0
        start = trade_dates[0]
        return LegResult(
            gross.loc[start:],
            cost.loc[start:],
            turnover.loc[start:],
            holdings.loc[start:],
        )

    def run(self) -> BacktestResult:
        """Backtest every quintile and the Q1-minus-Q5 spread."""
        quintiles = self.assign_quintiles()
        legs = {
            q: self.simulate(self.target_weights(quintiles, q))
            for q in range(1, self.config.quintiles + 1)
        }
        top, bottom = legs[1], legs[self.config.quintiles]
        benchmark = (
            self.data.benchmark.pct_change(fill_method=None)
            .reindex(
                top.gross.index,
            )
            .fillna(0.0)
        )
        benchmark.iloc[0] = 0.0
        return BacktestResult(
            legs=legs,
            long_short_net=top.net - bottom.gross - bottom.cost,
            long_short_gross=top.gross - bottom.gross,
            benchmark=benchmark,
            quintiles=quintiles,
        )

    @staticmethod
    def _normalized(holdings: pd.DataFrame) -> pd.DataFrame:
        filled = holdings.fillna(0.0)
        return filled.div(filled.sum(axis=1), axis=0).fillna(0.0)


@dataclass(frozen=True)
class RiskDecomposition:
    """Ex-ante annualized risk of one portfolio, split by source."""

    table: pd.DataFrame
    total_vol: float
    factor_vol: float
    specific_vol: float


class FactorRiskModel:
    """Barra-style fundamental factor risk model.

    Each day's stock returns are regressed cross-sectionally on a market
    factor and the style z-scores, using square-root market-cap weights:
    ``r = X f + u``. Exposures are those of the last rebalance strictly
    before the return date, so a factor return never uses later information.

    Industry factors are omitted: 30 names cannot support them alongside
    the style factors, so sector effects fall into specific returns.
    """

    def __init__(
        self,
        data: MarketData,
        factors: FactorSet,
        config: EngineConfig,
    ) -> None:
        self.config = config
        self.sources: tuple[str, ...] = (MARKET_FACTOR, *FACTOR_NAMES)
        calendar = data.adj_close.index
        self.columns: pd.Index = data.adj_close.columns
        self.scores: dict[str, pd.DataFrame] = {
            name: factors.scores[name]
            .reindex(columns=self.columns)
            .fillna(0.0)
            for name in FACTOR_NAMES
        }
        self.daily_exposures: dict[str, pd.DataFrame] = {
            name: frame.reindex(calendar, method="ffill").shift(1).fillna(0.0)
            for name, frame in self.scores.items()
        }
        self.factor_returns: pd.DataFrame
        self.specific_returns: pd.DataFrame
        self.adjusted_r2: pd.Series
        self._estimate(data, factors)

    def exposure_matrix(self, rebalance_date: pd.Timestamp) -> pd.DataFrame:
        """Stock-by-factor exposures, including a unit market exposure."""
        return pd.DataFrame(
            {
                MARKET_FACTOR: 1.0,
                **{
                    name: frame.loc[rebalance_date]
                    for name, frame in self.scores.items()
                },
            },
            index=self.columns,
        )

    def covariance(
        self,
        as_of: pd.Timestamp | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Annualized EWMA factor covariance and specific variances."""
        factor_history = self.factor_returns.loc[:as_of]
        specific_history = self.specific_returns.loc[:as_of]
        decay = 0.5 ** (1.0 / self.config.risk_half_life_days)
        weights = decay ** np.arange(len(factor_history) - 1, -1, -1.0)
        weights /= weights.sum()
        values = factor_history.to_numpy()
        covariance = (values.T * weights) @ values * TRADING_DAYS
        specific = ((specific_history**2).mul(weights, axis=0)).sum()
        return (
            pd.DataFrame(
                covariance,
                index=self.sources,
                columns=self.sources,
            ),
            specific * TRADING_DAYS,
        )

    def risk_decomposition(
        self,
        weights: pd.Series,
        rebalance_date: pd.Timestamp,
        as_of: pd.Timestamp | None = None,
    ) -> RiskDecomposition:
        """Split ex-ante variance w'(XFX' + D)w into additive sources."""
        aligned = weights.reindex(self.columns).fillna(0.0)
        covariance, specific = self.covariance(as_of)
        exposure = self.exposure_matrix(rebalance_date).T @ aligned
        marginal = covariance @ exposure
        contributions = exposure * marginal
        specific_variance = float((aligned**2 * specific).sum())
        factor_variance = float(contributions.sum())
        total_variance = factor_variance + specific_variance
        total_vol = float(np.sqrt(total_variance))

        table = pd.DataFrame(
            {
                "Exposure": exposure,
                "Risk Contribution": contributions / total_vol,
                "% of Risk": contributions / total_variance,
            }
        )
        table.loc[SPECIFIC_SOURCE] = [
            np.nan,
            specific_variance / total_vol,
            specific_variance / total_variance,
        ]
        return RiskDecomposition(
            table,
            total_vol,
            float(np.sqrt(factor_variance)),
            float(np.sqrt(specific_variance)),
        )

    def portfolio_exposures(self, holdings: pd.DataFrame) -> pd.DataFrame:
        """Daily portfolio exposure to every factor."""
        frames = {
            name: (holdings * exposure.reindex(holdings.index)).sum(axis=1)
            for name, exposure in self.daily_exposures.items()
        }
        return pd.DataFrame(
            {MARKET_FACTOR: holdings.sum(axis=1), **frames},
            index=holdings.index,
        )

    def return_attribution(
        self,
        holdings: pd.DataFrame,
        costs: pd.Series,
    ) -> pd.DataFrame:
        """Daily additive contributions; they sum to the net return."""
        index = holdings.index
        factor_returns = self.factor_returns.reindex(index).fillna(0.0)
        specific_returns = self.specific_returns.reindex(index).fillna(0.0)
        daily = self.portfolio_exposures(holdings) * factor_returns
        daily[SPECIFIC_SOURCE] = (holdings * specific_returns).sum(axis=1)
        daily[COST_SOURCE] = -costs.reindex(index).fillna(0.0)
        return daily

    def attribution_summary(
        self,
        holdings: pd.DataFrame,
        costs: pd.Series,
    ) -> pd.DataFrame:
        """Average exposures and annualized arithmetic contributions."""
        contribution = (
            self.return_attribution(holdings, costs).mean() * TRADING_DAYS
        )
        summary = pd.DataFrame(
            {
                "Avg Exposure": self.portfolio_exposures(holdings).mean(),
                "Ann. Contribution": contribution,
            },
            index=[*self.sources, SPECIFIC_SOURCE, COST_SOURCE],
        )
        summary.loc["total"] = [np.nan, float(contribution.sum())]
        return summary

    def factor_summary(
        self,
        start: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Return, volatility and t-statistic for each factor.

        A factor with no exposures has an exact zero return, so those
        unidentified days are excluded.
        """
        history = self.factor_returns.loc[start:].replace(0.0, np.nan)
        count = history.count()
        volatility = history.std()
        return pd.DataFrame(
            {
                "Ann. Return": history.mean() * TRADING_DAYS,
                "Ann. Vol": volatility * np.sqrt(TRADING_DAYS),
                "t-stat": history.mean() / volatility * np.sqrt(count),
                "Days": count.astype(float),
            }
        )

    def _regression_weights(
        self,
        data: MarketData,
        factors: FactorSet,
        rebalance_date: pd.Timestamp,
    ) -> np.ndarray:
        """Square-root market-cap weights; ineligible names get zero."""
        market_cap = data.close.loc[rebalance_date].mul(data.shares)
        root_cap = np.sqrt(market_cap.reindex(self.columns))
        median = root_cap.median()
        root_cap = root_cap.fillna(1.0 if pd.isna(median) else median)
        scored = pd.concat(
            [
                factors.scores[name].loc[rebalance_date].reindex(self.columns)
                for name in FACTOR_NAMES
            ],
            axis=1,
        )
        return root_cap.where(scored.notna().any(axis=1), 0.0).to_numpy()

    def _estimate(self, data: MarketData, factors: FactorSet) -> None:
        """Run the weighted cross-sectional regressions, period by period."""
        returns = data.adj_close.pct_change(fill_method=None).fillna(0.0)
        calendar = returns.index
        dates = factors.rebalance_dates
        factor_rows: list[np.ndarray] = []
        specific_rows: list[np.ndarray] = []
        fit_rows: list[np.ndarray] = []
        index_parts: list[pd.DatetimeIndex] = []

        for start, end in zip(dates, [*dates[1:], calendar[-1]]):
            days = calendar[(calendar > start) & (calendar <= end)]
            if days.empty:
                continue
            exposures = self.exposure_matrix(start).to_numpy()
            weights = self._regression_weights(data, factors, start)
            root = np.sqrt(weights)[:, np.newaxis]
            projector = np.linalg.pinv(exposures * root) * root.T
            stock_returns = returns.loc[days].to_numpy()
            factor_return = stock_returns @ projector.T
            residual = stock_returns - factor_return @ exposures.T

            mean = (stock_returns * weights).sum(axis=1, keepdims=True)
            mean /= weights.sum()
            total = (weights * (stock_returns - mean) ** 2).sum(axis=1)
            unexplained = (weights * residual**2).sum(axis=1)
            names = int((weights > 0).sum())
            rank = int(np.linalg.matrix_rank(exposures[weights > 0]))
            with np.errstate(divide="ignore", invalid="ignore"):
                adjusted = 1.0 - (unexplained / total) * (names - 1) / (
                    names - rank
                )

            factor_rows.append(factor_return)
            specific_rows.append(residual)
            fit_rows.append(adjusted)
            index_parts.append(days)

        index = index_parts[0].append(index_parts[1:])
        self.factor_returns = pd.DataFrame(
            np.vstack(factor_rows),
            index=index,
            columns=list(self.sources),
        )
        self.specific_returns = pd.DataFrame(
            np.vstack(specific_rows),
            index=index,
            columns=self.columns,
        )
        self.adjusted_r2 = pd.Series(
            np.concatenate(fit_rows),
            index=index,
            name="adjusted_r2",
        ).replace([np.inf, -np.inf], np.nan)


class RiskAnalytics:
    """Performance statistics, IC analysis and the HTML report."""

    def __init__(
        self,
        data: MarketData,
        factors: FactorSet,
        result: BacktestResult,
        config: EngineConfig,
    ) -> None:
        self.data = data
        self.factors = factors
        self.result = result
        self.config = config
        self.daily_rf: float = (1.0 + config.risk_free_rate) ** (
            1 / TRADING_DAYS
        ) - 1.0
        self.forward_returns: pd.DataFrame = self._forward_returns()
        self.composite_ic: pd.Series = self.information_coefficient(
            factors.composite,
        )

    @staticmethod
    def equity_curve(returns: pd.Series, base: float = 100.0) -> pd.Series:
        """Compound daily returns into an equity curve."""
        return base * (1.0 + returns).cumprod()

    @classmethod
    def drawdown(cls, returns: pd.Series) -> pd.Series:
        """Underwater curve as a fraction below the running peak."""
        equity = cls.equity_curve(returns)
        return equity / equity.cummax().clip(lower=100.0) - 1.0

    def rolling_sharpe(self, returns: pd.Series) -> pd.Series:
        """Annualized Sharpe ratio over the configured rolling window."""
        rolling = returns.sub(self.daily_rf).rolling(
            self.config.rolling_sharpe_window
        )
        return rolling.mean() / rolling.std() * np.sqrt(TRADING_DAYS)

    def metrics(
        self,
        returns: pd.Series,
        self_financing: bool = False,
    ) -> dict[str, float]:
        """Performance statistics for one daily return stream.

        Self-financing long-short returns are already excess returns, so no
        risk-free hurdle is deducted from them.
        """
        daily_rf = 0.0 if self_financing else self.daily_rf
        excess = returns - daily_rf
        market_excess = self.result.benchmark - self.daily_rf
        elapsed_days = (returns.index[-1] - returns.index[0]).days
        growth = float((1.0 + returns).prod())
        cagr = growth ** (CALENDAR_DAYS / elapsed_days) - 1.0
        volatility = returns.std() * np.sqrt(TRADING_DAYS)
        downside = np.sqrt((excess.clip(upper=0.0) ** 2).mean()) * np.sqrt(
            TRADING_DAYS
        )
        max_drawdown = float(self.drawdown(returns).min())
        beta = excess.cov(market_excess) / market_excess.var()
        annual_excess = excess.mean() * TRADING_DAYS
        return {
            "CAGR": cagr,
            "Volatility": volatility,
            "Sharpe": annual_excess / volatility if volatility else np.nan,
            "Sortino": annual_excess / downside if downside else np.nan,
            "Calmar": cagr / abs(max_drawdown) if max_drawdown else np.nan,
            "Max DD": max_drawdown,
            "Alpha": annual_excess
            - beta * market_excess.mean() * TRADING_DAYS,
            "Beta": beta,
        }

    def scorecard(self) -> pd.DataFrame:
        """Metrics for each quintile, the spread and the benchmark."""
        result = self.result
        rows = {
            f"Q{q}": self.metrics(leg.net) for q, leg in result.legs.items()
        }
        rows["Long-Short (Q1-Q5)"] = self.metrics(
            result.long_short_net,
            self_financing=True,
        )
        rows["Nifty 50 (price)"] = self.metrics(result.benchmark)
        table = pd.DataFrame(rows).T
        years = (
            result.benchmark.index[-1] - result.benchmark.index[0]
        ).days / CALENDAR_DAYS
        turnover = {
            f"Q{q}": leg.turnover.sum() / years
            for q, leg in result.legs.items()
        }
        turnover["Long-Short (Q1-Q5)"] = (
            turnover["Q1"] + turnover[f"Q{self.config.quintiles}"]
        )
        table["Turnover/yr"] = pd.Series(turnover)
        return table

    def information_coefficient(self, scores: pd.DataFrame) -> pd.Series:
        """Monthly Spearman correlation of scores with next-month returns."""
        forward = self.forward_returns.reindex(
            index=scores.index,
            columns=scores.columns,
        )
        paired = scores.notna() & forward.notna()
        score_ranks = scores.where(paired).rank(axis=1)
        forward_ranks = forward.where(paired).rank(axis=1)
        enough = paired.sum(axis=1) >= self.config.min_names
        return (
            score_ranks.loc[enough]
            .corrwith(
                forward_ranks.loc[enough],
                axis=1,
            )
            .dropna()
        )

    def ic_table(self) -> pd.DataFrame:
        """Mean IC, IR, t-statistic, hit rate and sample size by signal."""
        signals: dict[str, pd.Series] = {"composite": self.composite_ic}
        signals.update(
            {
                name: self.information_coefficient(self.factors.scores[name])
                for name in FACTOR_NAMES
            }
        )
        rows: dict[str, dict[str, float]] = {}
        for name, ic in signals.items():
            ratio = ic.mean() / ic.std() if len(ic) > 1 else np.nan
            rows[name] = {
                "Mean IC": ic.mean(),
                "Std IC": ic.std(),
                "IR": ratio,
                "t-stat": ratio * np.sqrt(len(ic)),
                "Hit rate": (ic > 0).mean(),
                "Months": float(len(ic)),
            }
        return pd.DataFrame(rows).T

    def _forward_returns(self) -> pd.DataFrame:
        monthly = self.data.adj_close.reindex(self.factors.rebalance_dates)
        return monthly.pct_change(fill_method=None).shift(-1)

    @staticmethod
    def _format_scorecard(table: pd.DataFrame) -> pd.DataFrame:
        percent = ("CAGR", "Volatility", "Max DD", "Alpha", "Turnover/yr")
        return pd.DataFrame(
            {
                column: table[column].map(
                    (lambda v: "-" if pd.isna(v) else f"{v:.1%}")
                    if column in percent
                    else (lambda v: "-" if pd.isna(v) else f"{v:.2f}")
                )
                for column in table.columns
            }
        )

    @staticmethod
    def _format_ic(table: pd.DataFrame) -> pd.DataFrame:
        formats = {
            "Mean IC": "{:+.3f}",
            "Std IC": "{:.3f}",
            "IR": "{:+.2f}",
            "t-stat": "{:+.2f}",
            "Hit rate": "{:.0%}",
            "Months": "{:.0f}",
        }
        return pd.DataFrame(
            {
                column: table[column].map(
                    lambda v, f=fmt: "-" if pd.isna(v) else f.format(v)
                )
                for column, fmt in formats.items()
            }
        )

    def print_scorecard(self) -> None:
        """Print the performance and IC tables to stdout."""
        result = self.result
        start, end = result.benchmark.index[[0, -1]]
        rule = "=" * 104
        print(f"\n{rule}\nALPHA ENGINE PERFORMANCE SCORECARD")
        print(
            f"{start.date()} to {end.date()} | "
            f"{self.data.adj_close.shape[1]} names | "
            f"{self.config.weighting} weighted | "
            f"{self.config.cost_bps:g} bps round-trip | "
            f"Rf {self.config.risk_free_rate:.1%}"
        )
        print(
            f"{rule}\n{self._format_scorecard(self.scorecard()).to_string()}"
        )
        ic_text = self._format_ic(self.ic_table()).to_string()
        print(f"\nINFORMATION COEFFICIENT\n{ic_text}")
        print(f"{rule}\n")

    @staticmethod
    def _layout(
        figure: go.Figure, title: str, y_title: str, height: int = 420
    ) -> None:
        figure.update_layout(
            title={
                "text": title,
                "font": {"size": 16, "color": INK_PRIMARY},
                "x": 0,
                "y": 0.97,
                "yanchor": "top",
            },
            height=height,
            margin={"l": 60, "r": 24, "t": 84, "b": 44},
            paper_bgcolor=SURFACE,
            plot_bgcolor=SURFACE,
            font={
                "family": 'system-ui, -apple-system, "Segoe UI", sans-serif',
                "color": INK_SECONDARY,
                "size": 12,
            },
            hovermode="x unified",
            legend={
                "orientation": "h",
                "y": 1.01,
                "x": 0,
                "yanchor": "bottom",
            },
        )
        figure.update_xaxes(showgrid=False, linecolor=AXIS, ticks="")
        figure.update_yaxes(
            title_text=y_title,
            gridcolor=GRID,
            zeroline=False,
            linecolor=AXIS,
        )

    @staticmethod
    def _line(
        series: pd.Series,
        name: str,
        color: str,
        hover_format: str,
    ) -> go.Scatter:
        return go.Scatter(
            x=series.index,
            y=series.to_numpy(),
            name=name,
            mode="lines",
            line={"color": color, "width": 2},
            hovertemplate=f"%{{y:{hover_format}}}",
        )

    def _core_series(self) -> list[tuple[pd.Series, str, str]]:
        result = self.result
        last = self.config.quintiles
        return [
            (result.legs[1].net, "Q1 top", COLOR_Q1),
            (result.legs[last].net, f"Q{last} bottom", COLOR_Q5),
            (result.benchmark, "Nifty 50", COLOR_BENCHMARK),
        ]

    def cumulative_figure(self) -> go.Figure:
        """Growth of 100 for Q1, Q5 and the benchmark."""
        figure = go.Figure()
        for returns, name, color in self._core_series():
            equity = self.equity_curve(returns)
            figure.add_trace(self._line(equity, name, color, ",.1f"))
        self._layout(figure, "Growth of 100, net of costs", "Portfolio value")
        return figure

    def long_short_figure(self) -> go.Figure:
        """Q1-minus-Q5 spread equity curve, gross and net of costs."""
        figure = go.Figure()
        figure.add_trace(
            self._line(
                self.equity_curve(self.result.long_short_gross),
                "Before costs",
                COLOR_BENCHMARK,
                ",.1f",
            )
        )
        figure.add_trace(
            self._line(
                self.equity_curve(self.result.long_short_net),
                "After costs",
                COLOR_LONG_SHORT,
                ",.1f",
            )
        )
        self._layout(
            figure, "Long-short spread (Q1 minus Q5)", "Growth of 100"
        )
        return figure

    def rolling_sharpe_figure(self) -> go.Figure:
        """Rolling six-month Sharpe ratio."""
        figure = go.Figure()
        for returns, name, color in self._core_series():
            figure.add_trace(
                self._line(
                    self.rolling_sharpe(returns).dropna(), name, color, ".2f"
                ),
            )
        self._layout(figure, "Rolling 6-month Sharpe ratio", "Sharpe ratio")
        return figure

    def drawdown_figure(self) -> go.Figure:
        """Underwater curves."""
        figure = go.Figure()
        for returns, name, color in self._core_series():
            figure.add_trace(
                self._line(self.drawdown(returns), name, color, ".1%")
            )
        self._layout(figure, "Drawdown from running peak", "Drawdown")
        figure.update_yaxes(tickformat=".0%")
        return figure

    def ic_figure(self) -> go.Figure:
        """Monthly composite IC bars beside their distribution."""
        ic = self.composite_ic
        figure = make_subplots(
            rows=1,
            cols=2,
            column_widths=[0.72, 0.28],
            horizontal_spacing=0.07,
            subplot_titles=("By month", "Distribution"),
        )
        for mask, name, color in (
            (ic >= 0, "IC above zero", COLOR_Q1),
            (ic < 0, "IC below zero", COLOR_NEGATIVE),
        ):
            figure.add_trace(
                go.Bar(
                    x=ic.index[mask],
                    y=ic[mask].to_numpy(),
                    name=name,
                    width=20 * 86_400_000,  # 20 days, in milliseconds
                    marker={"color": color},
                    hovertemplate="%{y:+.3f}",
                ),
                row=1,
                col=1,
            )
        figure.add_trace(
            go.Histogram(
                y=ic.to_numpy(),
                name="Months",
                showlegend=False,
                nbinsy=14,
                marker={
                    "color": COLOR_Q1,
                    "line": {"color": SURFACE, "width": 2},
                },
                hovertemplate="%{y}: %{x} months<extra></extra>",
            ),
            row=1,
            col=2,
        )
        self._layout(
            figure,
            "Monthly Information Coefficient (composite)",
            "Spearman IC",
        )
        figure.update_layout(barmode="overlay", hovermode="closest")
        figure.update_xaxes(title_text="Months", row=1, col=2)
        figure.update_yaxes(title_text="", row=1, col=2, matches="y")
        figure.update_annotations(font={"color": INK_SECONDARY, "size": 12})
        figure.add_hline(
            y=float(ic.mean()),
            row=1,
            col=1,
            line={"color": INK_PRIMARY, "width": 1},
            annotation_text=f"Mean {ic.mean():+.3f}",
            annotation_position="top right",
            annotation_font={"color": INK_PRIMARY},
        )
        return figure

    def data_notes(self) -> list[str]:
        """Limitations of this run, stated in the report itself."""
        cfg = self.config
        value = self.factors.scores["value"]
        populated = value.index[value.notna().sum(axis=1) >= cfg.min_names]
        value_start = (
            populated[0].date().isoformat() if len(populated) else "never"
        )
        repairs = [
            f"<strong>Data repair.</strong> {html.escape(repair)}."
            for repair in self.data.repairs
        ]
        return [
            *repairs,
            "<strong>Survivorship and selection bias.</strong> The universe "
            "is today's large caps held fixed through history, so past "
            "winners that joined the index recently are included from the "
            "start, and names removed from it are absent. Returns are biased "
            "upward.",
            f"<strong>Fundamental history is short.</strong> Yahoo supplies "
            f"about four fiscal years, so Value and the ROE and leverage "
            f"parts of Quality begin on {value_start}. Earlier rebalances "
            f"re-weight across Momentum, Return Stability and Low Volatility. "
            f"Fundamentals are lagged {cfg.fundamental_lag_days} days after "
            f"fiscal year-end.",
            "<strong>Market cap uses the latest share count.</strong> This "
            "is split-safe, but buybacks and issuance are not reflected "
            "historically.",
            "<strong>Benchmark is a price index.</strong> ^NSEI excludes "
            "dividends while the portfolios use dividend-adjusted prices, "
            "which flatters portfolio-versus-benchmark comparisons by "
            "roughly the index dividend yield.",
            f"<strong>Cost assumption.</strong> {cfg.cost_bps:g} bps per "
            "unit of one-way turnover is the configured input. Delivery STT "
            "alone is 0.1% per side (20 bps round-trip), so stress-test "
            "higher values.",
            "<strong>Q5 short leg is theoretical.</strong> Indian cash "
            "equities cannot be held short overnight; implementing it needs "
            "single-stock futures. Borrow, futures basis and margin carry "
            "are not modeled. The long-short Sharpe uses no risk-free hurdle "
            "because the spread is self-financing.",
            f"<strong>Execution.</strong> Signals from month-end closes trade "
            f"{cfg.execution_lag} session(s) later at the close. IC uses "
            "month-end to month-end forward returns.",
        ]

    def export_html(self) -> Path:
        """Write the self-contained interactive report and return its path."""
        cfg = self.config
        figures = [
            self.cumulative_figure(),
            self.long_short_figure(),
            self.rolling_sharpe_figure(),
            self.drawdown_figure(),
            self.ic_figure(),
        ]
        plot_config = {"displaylogo": False, "responsive": True}
        rendered = (
            figure.to_html(
                full_html=False,
                include_plotlyjs=index == 0,
                config=plot_config,
            )
            for index, figure in enumerate(figures)
        )
        chart_html = "\n".join(
            f'<section class="card">{chart}</section>' for chart in rendered
        )
        start, end = self.result.benchmark.index[[0, -1]]
        subtitle = html.escape(
            f"{start.date()} to {end.date()} | "
            f"{self.data.adj_close.shape[1]} names | "
            f"{cfg.weighting.replace('_', ' ')} weighted | "
            f"{cfg.cost_bps:g} bps round-trip | Rf {cfg.risk_free_rate:.1%}"
        )
        scorecard = self._format_scorecard(self.scorecard()).to_html(border=0)
        ic_table = self._format_ic(self.ic_table()).to_html(border=0)
        notes = "".join(f"<li>{note}</li>" for note in self.data_notes())
        document = _REPORT_TEMPLATE.format(
            subtitle=subtitle,
            scorecard=scorecard,
            ic_table=ic_table,
            charts=chart_html,
            notes=notes,
        )
        cfg.output.write_text(document, encoding="utf-8")
        return cfg.output


_REPORT_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Factor Backtest Report</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ margin: 0; background: #f9f9f7; color: #0b0b0b;
         font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }}
  main {{ max-width: 1120px; margin: 0 auto; padding: 32px 16px 56px; }}
  h1 {{ font-size: 26px; margin: 0 0 6px; }}
  h2 {{ font-size: 16px; margin: 0 0 12px; }}
  .sub {{ color: #52514e; margin: 0 0 24px; }}
  .card {{ background: #fcfcfb; border: 1px solid rgba(11,11,11,.10);
          border-radius: 10px; padding: 16px; margin-bottom: 16px;
          overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
  th, td {{ padding: 7px 10px; text-align: right; white-space: nowrap;
           border-bottom: 1px solid #e1e0d9;
           font-variant-numeric: tabular-nums; }}
  thead th {{ color: #52514e; font-weight: 600; }}
  tbody th {{ text-align: left; font-weight: 600; }}
  li {{ color: #52514e; line-height: 1.55; margin-bottom: 8px; }}
  li strong {{ color: #0b0b0b; }}
</style>
</head>
<body>
<main>
  <h1>Factor Backtest Report</h1>
  <p class="sub">{subtitle}</p>
  <section class="card">
    <h2>Performance scorecard (net of costs)</h2>{scorecard}
  </section>
  <section class="card">
    <h2>Information Coefficient by signal</h2>{ic_table}
  </section>
  {charts}
  <section class="card">
    <h2>Assumptions and limitations</h2><ul>{notes}</ul>
  </section>
</main>
</body>
</html>
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-factor quintile backtest for Indian equities.",
    )
    parser.add_argument("--years", type=int, default=5, help="backtest length")
    parser.add_argument(
        "--weighting",
        choices=["equal", "inverse_vol"],
        default="equal",
    )
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=15.0,
        help="round-trip cost in bps per unit of one-way turnover",
    )
    parser.add_argument("--risk-free", type=float, default=0.067)
    parser.add_argument(
        "--execution-lag",
        type=int,
        default=1,
        help="sessions between signal and trade (0 = at the signal close)",
    )
    parser.add_argument(
        "--factor-weights",
        type=float,
        nargs=4,
        default=[0.4, 0.3, 0.2, 0.1],
        metavar=("MOM", "QUAL", "VALUE", "LOWVOL"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("factor_backtest_report.html"),
    )
    return parser.parse_args(argv)


def run_pipeline(config: EngineConfig) -> RiskAnalytics:
    """Run ingestion, factor construction and the backtest."""
    data = DataIngestion(config).run()
    LOGGER.info("Constructing factors")
    factors = FactorEngine(data, config).build()
    LOGGER.info(
        "Running backtest over %d rebalances", len(factors.rebalance_dates)
    )
    result = FactorBacktester(data, factors, config).run()
    return RiskAnalytics(data, factors, result, config)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args(argv)
    total_weight = sum(args.factor_weights)
    if total_weight <= 0 or min(args.factor_weights) < 0:
        LOGGER.error(
            "Factor weights must be non-negative with a positive sum."
        )
        return 2
    config = EngineConfig(
        years=args.years,
        weighting=args.weighting,
        cost_bps=args.cost_bps,
        risk_free_rate=args.risk_free,
        execution_lag=args.execution_lag,
        factor_weights=dict(
            zip(
                FACTOR_NAMES,
                (w / total_weight for w in args.factor_weights),
            )
        ),
        output=args.output,
    )
    try:
        analytics = run_pipeline(config)
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        return 1
    analytics.print_scorecard()
    LOGGER.info("Report written to %s", analytics.export_html().resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
