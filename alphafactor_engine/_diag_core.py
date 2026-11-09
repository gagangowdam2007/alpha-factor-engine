"""Temporary fingerprint dump used to compare environments."""

import json
import sys

import numpy as np
import pandas as pd
import yfinance as yf


def dump(data, factors):
    cut = pd.Timestamp("2026-11-08")

    def items(frame, ticker, scale):
        series = frame[ticker].dropna()
        return [
            f"{d:%y%m%d}:{v / scale:.0f}" for d, v in series.items()
        ]

    fund = {}
    for ticker in data.adj_close.columns:
        fund[ticker.replace(".NS", "")] = {
            "ni": items(data.net_income, ticker, 1e7),
            "eq": items(data.equity, ticker, 1e7),
            "de": items(data.total_debt, ticker, 1e7),
            "sh": round(float(data.shares[ticker]) / 1e6, 1),
        }
    prices = data.adj_close.loc[:cut]
    closes = data.close.loc[:cut]
    return json.dumps(
        {
            "v": [sys.version.split()[0], pd.__version__, np.__version__,
                  yf.__version__],
            "n": [len(prices), str(prices.index[-1].date())],
            "rep": list(data.repairs),
            "f": fund,
            "adj": {
                t.replace(".NS", ""): round(float(prices[t].sum()), 1)
                for t in prices.columns
            },
            "cl": {
                t.replace(".NS", ""): round(float(closes[t].sum()), 1)
                for t in closes.columns
            },
            "dates": [f"{d:%y%m}" for d in factors.rebalance_dates],
            "sc": {
                name: [
                    round(float(v), 3)
                    for v in frame.abs().sum(axis=1).to_numpy()
                ]
                for name, frame in factors.scores.items()
            },
        },
        separators=(",", ":"),
    )
