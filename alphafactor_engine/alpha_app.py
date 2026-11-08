#!/usr/bin/env python3
"""Streamlit dashboard for the AlphaFactor-Engine backtester.

Run from this directory so the dark theme in ``.streamlit`` is applied:
    streamlit run alpha_app.py

Caching works in three layers, each keyed only by what it depends on:
    1. ``load_market_data``: Yahoo downloads, always at the maximum horizon.
    2. ``compute_factor_scores``: normalized factors, independent of controls.
    3. ``run_backtest``: one result per distinct set of control values.
Moving a control therefore never triggers a new download.
"""

from __future__ import annotations

from datetime import date
from typing import Sequence, TypedDict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from alpha_engine import (
    COST_SOURCE,
    DEFAULT_UNIVERSE,
    FACTOR_NAMES,
    MARKET_FACTOR,
    SPECIFIC_SOURCE,
    DataIngestion,
    EngineConfig,
    FactorBacktester,
    FactorEngine,
    FactorRiskModel,
    FactorSet,
    MarketData,
    RiskAnalytics,
    RiskDecomposition,
)

TITLE = (
    "AlphaFactor-Engine | Systematic Multi-Factor Cross-Sectional Backtester"
)
SUBTITLE = (
    "Nifty Universe | Factor Construction, Frictions & "
    "Barra-Style Risk Attribution"
)
AUTHOR = "Engineered by Gagan Gowda M"

MAX_YEARS = 5
CACHE_TTL_SECONDS = 12 * 60 * 60
BACKTEST_CACHE_ENTRIES = 256
LATE_START_TOLERANCE_DAYS = 45

FACTOR_CONTROLS: tuple[tuple[str, str, int], ...] = (
    ("momentum", "Momentum (12M - 1M)", 40),
    ("quality", "Quality (ROE + Stability + Leverage)", 30),
    ("value", "Value (Earnings Yield Rank)", 20),
    ("low_vol", "Low Volatility (Idiosyncratic Risk)", 10),
)
WEIGHTING_SCHEMES: dict[str, str] = {
    "Equal Weighted": "equal",
    "Inverse Volatility": "inverse_vol",
}

SURFACE = "#131b2a"
INK = "#e2e8f0"
INK_MUTED = "#94a3b8"
GRID = "#243044"
AXIS = "#3b4a61"
COLOR_Q1 = "#3987e5"
COLOR_Q2 = "#86b6ef"
COLOR_Q3 = "#7c8aa0"
COLOR_Q4 = "#eda380"
COLOR_Q5 = "#d95926"
COLOR_SPREAD = "#199e70"
COLOR_BENCHMARK = "#e2e8f0"
COLOR_NEGATIVE = "#e66767"

PORTFOLIO_Q1 = "Q1 Long"
PORTFOLIO_Q5 = "Q5 Short Basket"
PORTFOLIO_SPREAD = "Long-Short (Q1 - Q5)"

SOURCE_LABELS: dict[str, str] = {
    MARKET_FACTOR: "Market",
    "momentum": "Momentum",
    "quality": "Quality",
    "value": "Value",
    "low_vol": "Low Volatility",
    SPECIFIC_SOURCE: "Stock-Specific",
    COST_SOURCE: "Transaction Costs",
}
SOURCE_COLORS: dict[str, str] = {
    MARKET_FACTOR: "#e2e8f0",
    "momentum": "#3987e5",
    "quality": "#d95926",
    "value": "#199e70",
    "low_vol": "#c98500",
    SPECIFIC_SOURCE: "#d55181",
    COST_SOURCE: "#94a3b8",
}

Line = tuple[pd.Series, str, str]

PAGE_CSS = """
<style>
  .stApp { background: #0b1220; }
  [data-testid="stSidebar"] {
    background: #0f1726;
    border-right: 1px solid #243044;
  }
  [data-testid="stMetric"] {
    background: #131b2a;
    border: 1px solid #243044;
    border-radius: 10px;
    padding: 14px 16px;
  }
  [data-testid="stMetricValue"] { font-size: 1.65rem; }
  [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
    white-space: normal; overflow: visible; text-overflow: clip;
  }
  .ae-title { font-size: 1.7rem; font-weight: 700; color: #f1f5f9;
              line-height: 1.25; margin: 0; }
  .ae-subtitle { color: #94a3b8; margin: 6px 0 2px; font-size: 1rem; }
  .ae-author { color: #86b6ef; font-size: .85rem; margin: 0 0 18px; }
  .ae-weight { display: grid; align-items: center; gap: 8px;
               grid-template-columns: 72px 1fr 46px;
               color: #94a3b8; font-size: .8rem; margin: 3px 0; }
  .ae-track { background: #243044; border-radius: 2px; height: 6px; }
  .ae-fill { background: #3987e5; border-radius: 2px; height: 6px; }
  .ae-value { color: #e2e8f0; text-align: right;
              font-variant-numeric: tabular-nums; }
</style>
"""


class PortfolioAttribution(TypedDict):
    """Return and risk attribution for one portfolio."""

    summary: pd.DataFrame
    daily: pd.DataFrame
    risk: RiskDecomposition


class DashboardResult(TypedDict):
    """Everything the dashboard renders for one set of controls."""

    analytics: RiskAnalytics
    q1_holdings: pd.DataFrame
    q5_holdings: pd.DataFrame
    signal_date: pd.Timestamp
    trade_date: pd.Timestamp | None
    attribution: dict[str, PortfolioAttribution]
    factor_stats: pd.DataFrame
    adjusted_r2: float


def build_config(
    as_of: str,
    years: int,
    factor_weights: Sequence[float],
    weighting: str,
    cost_bps: float,
    execution_lag: int,
) -> EngineConfig:
    """Translate dashboard controls into an engine configuration."""
    names = [name for name, _, _ in FACTOR_CONTROLS]
    return EngineConfig(
        end_date=date.fromisoformat(as_of),
        years=years,
        factor_weights=dict(zip(names, factor_weights)),
        weighting=weighting,  # type: ignore[arg-type]
        cost_bps=cost_bps,
        execution_lag=execution_lag,
    )


def base_config(as_of: str) -> EngineConfig:
    """Configuration for the control-independent cached layers."""
    return EngineConfig(end_date=date.fromisoformat(as_of), years=MAX_YEARS)


@st.cache_data(ttl=CACHE_TTL_SECONDS, max_entries=3, show_spinner=False)
def load_market_data(as_of: str) -> MarketData:
    """Download and clean market data once, at the maximum horizon."""
    return DataIngestion(base_config(as_of)).run()


@st.cache_data(ttl=CACHE_TTL_SECONDS, max_entries=3, show_spinner=False)
def compute_factor_scores(as_of: str) -> FactorSet:
    """Compute normalized factors for every rebalance at the full horizon."""
    return FactorEngine(load_market_data(as_of), base_config(as_of)).build()


@st.cache_data(ttl=CACHE_TTL_SECONDS, max_entries=3, show_spinner=False)
def estimate_risk_model(as_of: str) -> FactorRiskModel:
    """Estimate the factor risk model, independent of every control."""
    return FactorRiskModel(
        load_market_data(as_of),
        compute_factor_scores(as_of),
        base_config(as_of),
    )


@st.cache_data(
    ttl=CACHE_TTL_SECONDS,
    max_entries=BACKTEST_CACHE_ENTRIES,
    show_spinner=False,
)
def run_backtest(
    as_of: str,
    years: int,
    factor_weights: tuple[float, ...],
    weighting: str,
    cost_bps: float,
    execution_lag: int,
) -> DashboardResult:
    """Run the backtest for one set of controls using cached inputs."""
    config = build_config(
        as_of,
        years,
        factor_weights,
        weighting,
        cost_bps,
        execution_lag,
    )
    data = load_market_data(as_of)
    full_scores = compute_factor_scores(as_of).scores

    engine = FactorEngine(data, config)
    dates = engine.rebalance_dates()
    scores = {
        name: frame.reindex(dates) for name, frame in full_scores.items()
    }
    factors = FactorSet(scores, engine.composite(scores), dates)

    backtester = FactorBacktester(data, factors, config)
    result = backtester.run()
    signal_date = result.quintiles.index[-1]
    bottom_quintile = config.quintiles
    targets = {
        quintile: backtester.target_weights(result.quintiles, quintile)
        for quintile in (1, bottom_quintile)
    }

    def holdings(quintile: int, ascending: bool) -> pd.DataFrame:
        latest = targets[quintile].loc[signal_date]
        members = latest.index[latest > 0]
        table = pd.DataFrame(
            {
                "Ticker": members,
                "Weight": latest[members].to_numpy() * 100.0,
                "Composite": factors.composite.loc[
                    signal_date, members
                ].to_numpy(),
                **{
                    label: scores[name].loc[signal_date, members].to_numpy()
                    for name, label in (
                        ("momentum", "Momentum"),
                        ("quality", "Quality"),
                        ("value", "Value"),
                        ("low_vol", "Low Vol"),
                    )
                },
            }
        )
        return table.sort_values("Composite", ascending=ascending)

    calendar = data.adj_close.index
    trade_position = calendar.get_indexer([signal_date])[0] + execution_lag
    trade_date = (
        calendar[trade_position] if trade_position < len(calendar) else None
    )

    model = estimate_risk_model(as_of)
    top, bottom = result.legs[1], result.legs[bottom_quintile]
    portfolios = {
        PORTFOLIO_Q1: (top.holdings, top.cost, targets[1]),
        PORTFOLIO_Q5: (bottom.holdings, bottom.cost, targets[bottom_quintile]),
        PORTFOLIO_SPREAD: (
            top.holdings - bottom.holdings,
            top.cost + bottom.cost,
            targets[1] - targets[bottom_quintile],
        ),
    }
    attribution = {
        label: PortfolioAttribution(
            summary=model.attribution_summary(held, cost),
            daily=model.return_attribution(held, cost),
            risk=model.risk_decomposition(
                target.loc[signal_date],
                signal_date,
            ),
        )
        for label, (held, cost, target) in portfolios.items()
    }
    window_start = result.benchmark.index[0]
    return DashboardResult(
        analytics=RiskAnalytics(data, factors, result, config),
        q1_holdings=holdings(1, ascending=False),
        q5_holdings=holdings(bottom_quintile, ascending=True),
        signal_date=signal_date,
        trade_date=trade_date,
        attribution=attribution,
        factor_stats=model.factor_summary(window_start),
        adjusted_r2=float(model.adjusted_r2.loc[window_start:].mean()),
    )


def render_header() -> None:
    """Draw the page title block."""
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    st.markdown(
        f'<p class="ae-title">{TITLE}</p>'
        f'<p class="ae-subtitle">{SUBTITLE.replace("&", "&amp;")}</p>'
        f'<p class="ae-author">{AUTHOR}</p>',
        unsafe_allow_html=True,
    )


def factor_weight_controls() -> tuple[float, ...] | None:
    """Draw factor sliders and return weights normalized to sum to one."""
    st.sidebar.subheader("Factor Weights")
    raw = [
        st.sidebar.slider(label, 0, 100, default, 5, key=f"weight_{name}")
        for name, label, default in FACTOR_CONTROLS
    ]
    total = sum(raw)
    if total == 0:
        st.sidebar.error("Set at least one factor weight above zero.")
        return None

    normalized = tuple(round(value / total, 6) for value in raw)
    short_names = ("Momentum", "Quality", "Value", "Low Vol")
    rows = "".join(
        f'<div class="ae-weight"><span>{name}</span>'
        f'<div class="ae-track"><div class="ae-fill" '
        f'style="width:{weight * 100:.1f}%"></div></div>'
        f'<span class="ae-value">{weight:.1%}</span></div>'
        for name, weight in zip(short_names, normalized)
    )
    st.sidebar.markdown(rows, unsafe_allow_html=True)
    st.sidebar.caption(f"Slider total {total}, normalized to 100%.")
    return normalized


def render_universe_panel() -> None:
    """Show the fixed universe in the sidebar."""
    st.sidebar.subheader("Universe")
    st.sidebar.markdown(
        f"Curated Nifty {len(DEFAULT_UNIVERSE)} large-caps + benchmark "
        "`^NSEI`"
    )
    with st.sidebar.expander("View tickers"):
        st.caption(", ".join(DEFAULT_UNIVERSE))


def line_figure(
    lines: Sequence[Line],
    y_title: str,
    hover_format: str,
    tick_format: str | None = None,
    height: int = 420,
) -> go.Figure:
    """Build a dark multi-series line chart on a single axis."""
    figure = go.Figure()
    for series, name, color in lines:
        figure.add_trace(
            go.Scatter(
                x=series.index,
                y=series.to_numpy(),
                name=name,
                mode="lines",
                line={"color": color, "width": 2},
                hovertemplate=f"%{{y:{hover_format}}}",
            )
        )
    apply_dark_layout(figure, y_title, height)
    figure.update_layout(hovermode="x unified")
    if tick_format:
        figure.update_yaxes(tickformat=tick_format)
    return figure


def apply_dark_layout(
    figure: go.Figure,
    y_title: str,
    height: int = 420,
) -> None:
    """Apply the shared slate chart styling."""
    figure.update_layout(
        height=height,
        margin={"l": 60, "r": 20, "t": 44, "b": 40},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={
            "family": 'system-ui, -apple-system, "Segoe UI", sans-serif',
            "color": INK_MUTED,
            "size": 12,
        },
        hoverlabel={"bgcolor": "#0f1726", "font": {"color": INK}},
        legend={"orientation": "h", "y": 1.02, "x": 0, "yanchor": "bottom"},
    )
    figure.update_xaxes(showgrid=False, linecolor=AXIS, ticks="")
    figure.update_yaxes(
        title_text=y_title,
        gridcolor=GRID,
        zeroline=False,
        linecolor=AXIS,
    )


def show_chart(figure: go.Figure) -> None:
    """Render a Plotly figure at full container width."""
    st.plotly_chart(
        figure,
        width="stretch",
        theme=None,
        config={"displayModeBar": False},
    )


def render_kpis(analytics: RiskAnalytics) -> None:
    """Draw the KPI summary ribbon."""
    q1 = analytics.scorecard().loc["Q1"]
    information_ratio = analytics.ic_table().loc["composite", "IR"]
    columns = st.columns(5)
    columns[0].metric("Q1 Net CAGR", f"{q1['CAGR']:.1%}")
    columns[1].metric("Q1 Sharpe Ratio", f"{q1['Sharpe']:.2f}")
    columns[2].metric("Annual Alpha vs Nifty 50", f"{q1['Alpha']:+.1%}")
    columns[3].metric("Max Drawdown", f"{q1['Max DD']:.1%}")
    columns[4].metric(
        "Composite IR",
        "-" if pd.isna(information_ratio) else f"{information_ratio:+.2f}",
    )
    st.caption(
        "Q1 figures are net of costs. Alpha is CAPM alpha against ^NSEI, "
        "which excludes dividends while Q1 includes them, so it is "
        "flattered by about the index dividend yield. IR is mean monthly "
        "IC divided by the standard deviation of IC."
    )


def render_performance_tab(analytics: RiskAnalytics) -> None:
    """Tab 1: cumulative performance and the long-short spread."""
    result = analytics.result
    st.subheader("Growth of 100")
    show_chart(
        line_figure(
            [
                (
                    analytics.equity_curve(result.legs[1].net),
                    "Q1 Long",
                    COLOR_Q1,
                ),
                (
                    analytics.equity_curve(result.legs[5].net),
                    "Q5 Short Basket",
                    COLOR_Q5,
                ),
                (
                    analytics.equity_curve(result.benchmark),
                    "Nifty 50",
                    COLOR_BENCHMARK,
                ),
            ],
            "Portfolio value",
            ",.1f",
        )
    )
    st.caption(
        "Net of costs. The Q5 line is the basket's own return, so a rising "
        "line is a loss for a short position."
    )
    st.subheader("Long-Short Spread (Q1 minus Q5)")
    show_chart(
        line_figure(
            [
                (
                    analytics.equity_curve(result.long_short_gross),
                    "Gross",
                    INK_MUTED,
                ),
                (
                    analytics.equity_curve(result.long_short_net),
                    "Net of turnover costs",
                    COLOR_SPREAD,
                ),
            ],
            "Growth of 100",
            ",.1f",
        )
    )


def render_risk_tab(analytics: RiskAnalytics) -> None:
    """Tab 2: drawdowns and rolling Sharpe."""
    result = analytics.result
    colors = (COLOR_Q1, COLOR_Q2, COLOR_Q3, COLOR_Q4, COLOR_Q5)
    st.subheader("Underwater Drawdown")
    drawdowns: list[Line] = [
        (analytics.drawdown(result.legs[q].net), f"Q{q}", colors[q - 1])
        for q in range(1, 6)
    ]
    drawdowns.append(
        (analytics.drawdown(result.benchmark), "Nifty 50", COLOR_BENCHMARK),
    )
    show_chart(line_figure(drawdowns, "Drawdown", ".1%", tick_format=".0%"))

    st.subheader("Rolling 6-Month Sharpe Ratio")
    streams: tuple[tuple[pd.Series, str, str], ...] = (
        (result.legs[1].net, "Q1 Long", COLOR_Q1),
        (result.legs[5].net, "Q5 Short Basket", COLOR_Q5),
        (result.benchmark, "Nifty 50", COLOR_BENCHMARK),
    )
    show_chart(
        line_figure(
            [
                (analytics.rolling_sharpe(stream).dropna(), name, color)
                for stream, name, color in streams
            ],
            "Sharpe ratio",
            ".2f",
        )
    )


def ic_bar_figure(ic: pd.Series) -> go.Figure:
    """Monthly IC bars, colored by sign."""
    figure = go.Figure()
    for mask, name, color in (
        (ic >= 0, "Positive alpha month", COLOR_Q1),
        (ic < 0, "Negative alpha month", COLOR_NEGATIVE),
    ):
        figure.add_trace(
            go.Bar(
                x=ic.index[mask],
                y=ic[mask].to_numpy(),
                name=name,
                width=20 * 86_400_000,  # 20 days, in milliseconds
                marker={"color": color},
                hovertemplate="%{x|%b %Y}: %{y:+.3f}<extra></extra>",
            )
        )
    apply_dark_layout(figure, "Spearman IC")
    figure.update_layout(barmode="overlay")
    return figure


def ic_histogram_figure(ic: pd.Series) -> go.Figure:
    """IC distribution with a vertical mean line."""
    figure = go.Figure(
        go.Histogram(
            x=ic.to_numpy(),
            nbinsx=16,
            name="Months",
            marker={"color": COLOR_Q1, "line": {"color": SURFACE, "width": 2}},
            hovertemplate="IC %{x}: %{y} months<extra></extra>",
        )
    )
    apply_dark_layout(figure, "Months")
    figure.update_layout(showlegend=False)
    figure.update_xaxes(title_text="Spearman IC")
    figure.add_vline(
        x=float(ic.mean()),
        line={"color": INK, "width": 2},
        annotation_text=f"Mean IC {ic.mean():+.3f}",
        annotation_position="top",
        annotation_font={"color": INK},
    )
    return figure


def render_ic_tab(analytics: RiskAnalytics) -> None:
    """Tab 3: IC charts and the factor scorecard."""
    ic = analytics.composite_ic
    if len(ic) < 2:
        st.info("Too few months of composite scores to analyze IC.")
    else:
        bars, histogram = st.columns([3, 2])
        with bars:
            st.subheader("Monthly IC (Composite)")
            show_chart(ic_bar_figure(ic))
        with histogram:
            st.subheader("IC Distribution")
            show_chart(ic_histogram_figure(ic))

    st.subheader("IC Scorecard by Factor")
    table = analytics.ic_table().rename(
        index={
            "composite": "Composite",
            "momentum": "Momentum",
            "quality": "Quality",
            "value": "Value",
            "low_vol": "Low Volatility",
        }
    )
    table.index.name = "Signal"
    table["Hit rate"] = table["Hit rate"] * 100.0
    st.dataframe(
        table.reset_index(),
        hide_index=True,
        width="stretch",
        column_config={
            "Mean IC": st.column_config.NumberColumn(format="%+.3f"),
            "Std IC": st.column_config.NumberColumn(format="%.3f"),
            "IR": st.column_config.NumberColumn(format="%+.2f"),
            "t-stat": st.column_config.NumberColumn(format="%+.2f"),
            "Hit rate": st.column_config.NumberColumn(format="%.0f%%"),
            "Months": st.column_config.NumberColumn(format="%d"),
        },
    )
    st.caption(
        "Value starts when point-in-time fundamentals begin, so it covers "
        "fewer months than the price-based factors."
    )


def render_holdings(title: str, holdings: pd.DataFrame) -> None:
    """Draw one basket's holdings table."""
    st.subheader(title)
    score_columns = ("Composite", "Momentum", "Quality", "Value", "Low Vol")
    st.dataframe(
        holdings,
        hide_index=True,
        width="stretch",
        column_config={
            "Weight": st.column_config.NumberColumn(format="%.1f%%"),
            **{
                name: st.column_config.NumberColumn(format="%+.2f")
                for name in score_columns
            },
        },
    )


def render_constituents_tab(result: DashboardResult) -> None:
    """Tab 4: baskets from the most recent completed rebalance."""
    trade_date = result["trade_date"]
    trade_text = (
        trade_date.date().isoformat()
        if trade_date is not None
        else "next session"
    )
    st.markdown(
        f"**Latest rebalance signal:** "
        f"{result['signal_date'].date().isoformat()} &nbsp;|&nbsp; "
        f"**Trade date:** {trade_text}"
    )
    render_holdings("Q1 | Long Alpha Basket", result["q1_holdings"])
    render_holdings("Q5 | Short / Underweight Basket", result["q5_holdings"])
    st.caption(
        "Scores are cross-sectional z-scores; higher is more attractive. "
        "Baskets update at each completed month-end."
    )


def contribution_bar_figure(values: pd.Series, x_title: str) -> go.Figure:
    """Horizontal bars by source, colored by sign, on one percent axis."""
    labels = [SOURCE_LABELS[source] for source in values.index]
    figure = go.Figure(
        go.Bar(
            y=labels,
            x=values.to_numpy(),
            orientation="h",
            marker={
                "color": [
                    COLOR_Q1 if value >= 0 else COLOR_NEGATIVE
                    for value in values
                ],
            },
            hovertemplate="%{y}: %{x:+.2%}<extra></extra>",
        )
    )
    apply_dark_layout(figure, "", height=340)
    figure.update_layout(showlegend=False, margin={"l": 130, "t": 16})
    figure.update_yaxes(autorange="reversed", gridcolor=SURFACE)
    figure.update_xaxes(
        title_text=x_title,
        tickformat=".0%",
        showgrid=True,
        gridcolor=GRID,
        zeroline=True,
        zerolinecolor=AXIS,
    )
    return figure


def attribution_table(attribution: PortfolioAttribution) -> pd.DataFrame:
    """Combine return attribution and current risk into one table."""
    summary = attribution["summary"]
    risk = attribution["risk"].table.reindex(summary.index)
    labels = {**SOURCE_LABELS, "total": "Total (Net)"}
    table = pd.DataFrame(
        {
            "Source": [labels[source] for source in summary.index],
            "Avg Exposure": summary["Avg Exposure"].to_numpy(),
            "Ann. Contribution": summary["Ann. Contribution"].to_numpy()
            * 100.0,
            "Current Exposure": risk["Exposure"].to_numpy(),
            "Risk Contribution": risk["Risk Contribution"].to_numpy() * 100.0,
            "% of Risk": risk["% of Risk"].to_numpy() * 100.0,
        }
    )
    total_row = table.index[table["Source"] == labels["total"]]
    table.loc[total_row, "Risk Contribution"] = (
        attribution["risk"].total_vol * 100.0
    )
    table.loc[total_row, "% of Risk"] = 100.0
    return table


def render_attribution_tab(result: DashboardResult) -> None:
    """Tab 5: factor attribution of returns and ex-ante risk."""
    selected = st.radio(
        "Portfolio",
        list(result["attribution"]),
        horizontal=True,
        key="attribution_portfolio",
        label_visibility="collapsed",
    )
    attribution = result["attribution"][selected]
    risk = attribution["risk"]

    columns = st.columns(4)
    columns[0].metric("Ex-Ante Volatility", f"{risk.total_vol:.1%}")
    columns[1].metric("Factor Volatility", f"{risk.factor_vol:.1%}")
    columns[2].metric("Specific Volatility", f"{risk.specific_vol:.1%}")
    columns[3].metric("Avg Adjusted R²", f"{result['adjusted_r2']:.1%}")

    returns_column, risk_column = st.columns(2)
    with returns_column:
        st.subheader("Where Return Came From")
        contributions = attribution["summary"]["Ann. Contribution"].drop(
            "total"
        )
        show_chart(
            contribution_bar_figure(
                contributions,
                "Annualized contribution",
            )
        )
    with risk_column:
        st.subheader("Where Risk Comes From")
        show_chart(
            contribution_bar_figure(
                risk.table["% of Risk"],
                "Share of current ex-ante variance",
            )
        )

    st.subheader("Cumulative Contribution by Source")
    cumulative = attribution["daily"].cumsum()
    show_chart(
        line_figure(
            [
                (cumulative[source], label, SOURCE_COLORS[source])
                for source, label in SOURCE_LABELS.items()
            ],
            "Sum of daily contributions",
            "+.1%",
            tick_format=".0%",
        )
    )

    st.subheader("Attribution Detail")
    st.dataframe(
        attribution_table(attribution),
        hide_index=True,
        placeholder="-",
        width="stretch",
        column_config={
            "Avg Exposure": st.column_config.NumberColumn(format="%+.2f"),
            "Ann. Contribution": st.column_config.NumberColumn(
                format="%+.2f%%"
            ),
            "Current Exposure": st.column_config.NumberColumn(format="%+.2f"),
            "Risk Contribution": st.column_config.NumberColumn(
                format="%+.2f%%"
            ),
            "% of Risk": st.column_config.NumberColumn(format="%+.1f%%"),
        },
    )

    st.subheader("Factor Returns")
    stats = result["factor_stats"].rename(index=SOURCE_LABELS)
    stats.index.name = "Factor"
    stats[["Ann. Return", "Ann. Vol"]] *= 100.0
    st.dataframe(
        stats.reset_index(),
        hide_index=True,
        width="stretch",
        column_config={
            "Ann. Return": st.column_config.NumberColumn(format="%+.2f%%"),
            "Ann. Vol": st.column_config.NumberColumn(format="%.2f%%"),
            "t-stat": st.column_config.NumberColumn(format="%+.2f"),
            "Days": st.column_config.NumberColumn(format="%d"),
        },
    )

    factor_list = ", ".join(SOURCE_LABELS[name] for name in FACTOR_NAMES)
    half_life = result["analytics"].config.risk_half_life_days
    st.caption(
        f"Model: daily weighted cross-sectional regressions on Market, "
        f"{factor_list}, with square-root market-cap weights and a "
        f"{half_life}-day half-life for risk. "
        "Contributions are arithmetic (mean daily x "
        "252), so they add exactly to the total but differ from CAGR. "
        "Current risk uses the latest rebalance weights. The Q5 basket is "
        "shown as its own return; a short position has the opposite sign. "
        "There are no industry factors, as 30 names cannot support them, "
        "so sector effects sit in Stock-Specific."
    )


def render_late_start_notice(analytics: RiskAnalytics) -> None:
    """Explain when the factor mix shortens the requested horizon."""
    first_signal = analytics.result.quintiles.index[0]
    requested = analytics.config.backtest_start
    if (first_signal - requested).days > LATE_START_TOLERANCE_DAYS:
        st.warning(
            f"The backtest starts at {first_signal.date()}, later than the "
            f"selected horizon. Value and the fundamental data begin in "
            f"mid-2023, and this weight mix leaves under half of the "
            f"composite available before the start date."
        )


def render_dashboard(result: DashboardResult) -> None:
    """Draw the KPI ribbon, tabs and limitations."""
    analytics = result["analytics"]
    render_late_start_notice(analytics)
    render_kpis(analytics)
    performance, risk, information, constituents, attribution = st.tabs(
        [
            "Cumulative Performance & Spreads",
            "Risk & Drawdown Dynamics",
            "Information Coefficient (IC) Analytics",
            "Live Portfolio Constituents",
            "Barra-Style Risk Attribution",
        ]
    )
    with performance:
        render_performance_tab(analytics)
    with risk:
        render_risk_tab(analytics)
    with information:
        render_ic_tab(analytics)
    with constituents:
        render_constituents_tab(result)
    with attribution:
        render_attribution_tab(result)
    with st.expander("Assumptions and limitations"):
        notes = "\n".join(f"- {note}" for note in analytics.data_notes())
        st.markdown(notes, unsafe_allow_html=True)


def main() -> None:
    """Application entry point."""
    st.set_page_config(
        page_title="AlphaFactor-Engine",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    render_header()

    render_universe_panel()
    st.sidebar.subheader("Lookback Horizon")
    years = st.sidebar.slider("Years", 3, MAX_YEARS, MAX_YEARS)
    factor_weights = factor_weight_controls()
    st.sidebar.subheader("Portfolio Weighting Scheme")
    scheme = st.sidebar.radio(
        "Scheme",
        list(WEIGHTING_SCHEMES),
        label_visibility="collapsed",
    )
    st.sidebar.subheader("Frictions & Execution")
    cost_bps = st.sidebar.slider("Round-trip Cost (bps)", 5, 50, 15)
    execution_lag = st.sidebar.radio(
        "Execution Lag (days)",
        [0, 1],
        index=1,
        horizontal=True,
    )

    if st.sidebar.button(
        "Run Multi-Factor Backtest",
        type="primary",
        width="stretch",
    ):
        st.session_state["has_run"] = True
    if st.sidebar.button("Refresh Market Data", width="stretch"):
        st.cache_data.clear()

    if not st.session_state.get("has_run"):
        st.info(
            "Set your parameters and press **Run Multi-Factor Backtest**. "
            "After the first run, changes to any control update instantly."
        )
        return
    if factor_weights is None:
        st.warning("Results are hidden until a factor weight is above zero.")
        return

    try:
        with st.spinner("Loading market data and running the backtest..."):
            result = run_backtest(
                date.today().isoformat(),
                years,
                factor_weights,
                WEIGHTING_SCHEMES[scheme],
                float(cost_bps),
                int(execution_lag),
            )
    except Exception as exc:  # shown to the user instead of a stack trace
        st.error(f"The backtest could not be completed: {exc}")
        return
    render_dashboard(result)


if __name__ == "__main__":
    main()
