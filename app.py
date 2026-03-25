import itertools
import io
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

try:
    from statsmodels.tsa.stattools import coint

    HAS_COINT = True
except Exception:
    HAS_COINT = False


NEON = {
    "green": "#39FF14",
    "red": "#FF3B3B",
    "cyan": "#00F5FF",
    "magenta": "#FF2BD6",
    "yellow": "#FFE600",
    "orange": "#FF8A00",
    "purple": "#B388FF",
    "blue": "#3B82F6",
    "white": "#EAEAEA",
}


def zscore(s: pd.Series, eps: float = 1e-12) -> pd.Series:
    return (s - s.mean()) / (s.std(ddof=0) + eps)


def hurst_rs(series: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 40:
        return float("nan")
    x = x - x.mean()
    max_k = min(128, len(x) // 2)
    if max_k < 16:
        return float("nan")
    ks = np.unique(np.logspace(math.log10(16), math.log10(max_k), num=8).astype(int))
    pts = []
    for k in ks:
        segs = len(x) // k
        if segs < 2:
            continue
        rs = []
        for i in range(segs):
            seg = x[i * k : (i + 1) * k]
            cum = np.cumsum(seg)
            r = cum.max() - cum.min()
            s = seg.std(ddof=0)
            rs.append(r / (s + eps))
        if rs:
            pts.append((k, float(np.mean(rs))))
    if len(pts) < 2:
        return float("nan")
    k_arr = np.array([p[0] for p in pts], dtype=float)
    rs_arr = np.array([p[1] for p in pts], dtype=float)
    return float(np.polyfit(np.log(k_arr), np.log(np.maximum(rs_arr, eps)), 1)[0])


def set_plotly_layout(fig: go.Figure, title: str):
    fig.update_layout(
        template="plotly_dark",
        title={"text": title, "x": 0.01},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")


def parse_round_day(fname: str) -> Optional[Tuple[str, int, int]]:
    m = re.match(r"^(prices|trades|observations)_round_(\d+)_day_(-?\d+)\.csv$", fname)
    if not m:
        return None
    k, r, d = m.groups()
    return k, int(r), int(d)


def discover_rounds_from_filenames(filenames: List[str]) -> List[int]:
    rounds = set()
    for fname in filenames:
        parsed = parse_round_day(fname)
        if parsed:
            _, r, _ = parsed
            rounds.add(r)
    return sorted(rounds)


@st.cache_data(show_spinner=False)
def load_infinite_data(uploaded_payload: Tuple[Tuple[str, bytes], ...], selected_rounds: Optional[Tuple[int, ...]] = None):
    rounds_set = None if selected_rounds is None else set(selected_rounds)
    price_files, trade_files, obs_files = [], [], []
    for fname, content in uploaded_payload:
        parsed = parse_round_day(fname)
        if not parsed:
            continue
        kind, r, d = parsed
        if rounds_set is not None and r not in rounds_set:
            continue
        item = (r, d, fname, content)
        if kind == "prices":
            price_files.append(item)
        elif kind == "trades":
            trade_files.append(item)
        else:
            obs_files.append(item)

    if not price_files:
        raise FileNotFoundError("No prices files for the chosen rounds.")

    price_files.sort(key=lambda x: (x[0], x[1]))
    offsets = {}
    prev_max = 0
    for r, d, _, content in price_files:
        ts = pd.read_csv(io.BytesIO(content), sep=";", usecols=["timestamp"])["timestamp"]
        offset = prev_max + 100
        offsets[(r, d)] = offset
        prev_max = int(ts.max()) + offset

    def read_one(content: bytes, kind: str) -> pd.DataFrame:
        sep = ";" if kind in ("prices", "trades") else ","
        return pd.read_csv(io.BytesIO(content), sep=sep)

    def load_group(files, kind, normalize=None):
        out = []
        for r, d, _, content in sorted(files, key=lambda x: (x[0], x[1])):
            df = read_one(content, kind)
            if "timestamp" not in df.columns:
                continue
            df["t"] = pd.to_numeric(df["timestamp"], errors="coerce") + offsets[(r, d)]
            df["round"] = r
            df["day"] = d
            if normalize:
                df = normalize(df)
            out.append(df)
        if not out:
            return pd.DataFrame()
        return pd.concat(out, ignore_index=True)

    def normalize_prices(df: pd.DataFrame):
        for side in ("bid", "ask"):
            for lvl in (1, 2, 3):
                pcol, vcol = f"{side}_price_{lvl}", f"{side}_volume_{lvl}"
                if pcol not in df.columns:
                    df[pcol] = np.nan
                if vcol not in df.columns:
                    df[vcol] = np.nan
        if "mid_price" not in df.columns:
            df["mid_price"] = 0.5 * (df["bid_price_1"] + df["ask_price_1"])
        return df

    def normalize_trades(df: pd.DataFrame):
        if "symbol" in df.columns and "product" not in df.columns:
            df = df.rename(columns={"symbol": "product"})
        return df

    prices = load_group(price_files, "prices", normalize=normalize_prices)
    trades = load_group(trade_files, "trades", normalize=normalize_trades)
    obs = load_group(obs_files, "observations")

    return prices, trades, obs


def kalman_beta(y: np.ndarray, x: np.ndarray):
    # Force strictly numeric vectors to avoid dtype/object issues.
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    state = np.array([0.0, 0.0], dtype=float)  # intercept, beta
    P = np.eye(2) * 1e3
    Q = np.diag([1e-6, 1e-5])
    r_obs = max(np.nanvar(y) * 1e-2, 1e-8)
    n = len(y)
    intercept, beta, resid = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        P = P + Q
        # Equivalent observation model: y_t = a_t + b_t * x_t
        # Using scalar arithmetic avoids ndarray->float conversion edge-cases.
        yhat = float(state[0] + state[1] * x[i])
        e = float(y[i] - yhat)
        H = np.array([[1.0, x[i]]], dtype=float)
        S = float(H @ P @ H.T + r_obs)
        K = (P @ H.T) / max(S, 1e-8)
        state = state + K.flatten() * e
        P = P - K @ H @ P
        intercept[i], beta[i], resid[i] = state[0], state[1], e
    return intercept, beta, resid


st.set_page_config(page_title="IMC Prosperity Research Dashboard", layout="wide")
st.title("IMC Prosperity - Streamlit Research App")
st.caption("Notebook-equivalent dashboard with round filter and interactive Plotly modules.")

uploaded_files = st.sidebar.file_uploader(
    "Upload IMC CSV files (prices/trades/observations)",
    type=["csv"],
    accept_multiple_files=True,
)
if not uploaded_files:
    st.info("Upload CSV files in the sidebar to start.")
    st.stop()

uploaded_payload = tuple((f.name, f.getvalue()) for f in uploaded_files)
all_rounds = discover_rounds_from_filenames([name for name, _ in uploaded_payload])
rounds_sel = st.sidebar.multiselect("Rounds to load", options=all_rounds, default=all_rounds or None)
horizon = st.sidebar.slider("Future horizon (steps)", 1, 50, 10)
max_points = st.sidebar.slider("Max points in heavy charts", 5000, 60000, 25000, step=1000)

if not all_rounds:
    st.warning("No valid round files detected in uploaded CSVs.")
    st.stop()

selected_tuple = tuple(rounds_sel) if rounds_sel else None
prices, trades, obs = load_infinite_data(uploaded_payload, selected_tuple)

if prices.empty:
    st.error("No price data loaded.")
    st.stop()

for col in ("mid_price", "t"):
    if col in prices.columns:
        prices[col] = pd.to_numeric(prices[col], errors="coerce")
prices = prices.sort_values(["product", "t"])
products = sorted(prices["product"].dropna().unique())

st.success(
    f"Loaded rounds: {rounds_sel if rounds_sel else 'ALL'} | "
    f"prices={len(prices):,} trades={len(trades):,} observations={len(obs):,} products={len(products)}"
)

tabs = st.tabs(
    [
        "Module 1: 3D Topography",
        "Module 2: Counterparty Edge",
        "Module 3: Stat-Arb",
        "Module 4: Microstructure",
        "Module 5: Env Fair Value",
        "Module 6: Spectral",
        "Module 7: Entropy",
        "Module 8: Liquidity Void",
        "Final JSON",
    ]
)

strategy = {"meta": {}, "stat_arb": {}, "microstructure": {}, "fair_value_environment": {}, "alpha_original": {}}
strategy["meta"] = {
    "selected_rounds": rounds_sel,
    "products": products,
    "latest_t": float(prices["t"].max()),
}

# Module 1
with tabs[0]:
    df = prices[["product", "t", "mid_price"]].dropna().copy()
    df["norm_price"] = df.groupby("product")["mid_price"].transform(zscore)
    p2i = {p: i for i, p in enumerate(products)}
    df["pid"] = df["product"].map(p2i)
    if len(df) > max_points:
        keep = []
        for p in products:
            sub = df[df["product"] == p]
            step = max(1, len(sub) // max(1, max_points // max(1, len(products))))
            keep.append(sub.iloc[::step])
        df = pd.concat(keep, ignore_index=True)
    view_mode = st.radio("Topography view", options=["3D", "2D"], horizontal=True, key="topography_view")

    fig = go.Figure()
    palette = [NEON["green"], NEON["red"], NEON["cyan"], NEON["magenta"], NEON["yellow"], NEON["purple"]]
    if view_mode == "3D":
        for i, p in enumerate(products):
            sub = df[df["product"] == p]
            fig.add_trace(
                go.Scatter3d(
                    x=sub["t"],
                    y=sub["pid"],
                    z=sub["norm_price"],
                    mode="lines",
                    line=dict(color=palette[i % len(palette)], width=2),
                    name=p,
                )
            )
        fig.update_layout(
            scene=dict(
                yaxis=dict(tickmode="array", tickvals=list(p2i.values()), ticktext=list(p2i.keys())),
                xaxis_title="t",
                yaxis_title="product",
                zaxis_title="normalized price",
            )
        )
        set_plotly_layout(fig, "Global Market Topography (3D)")
    else:
        for i, p in enumerate(products):
            sub = df[df["product"] == p]
            fig.add_trace(
                go.Scatter(
                    x=sub["t"],
                    y=sub["norm_price"],
                    mode="lines",
                    line=dict(color=palette[i % len(palette)], width=1.8),
                    name=p,
                )
            )
        fig.update_xaxes(title="t")
        fig.update_yaxes(title="normalized price (z-score)")
        set_plotly_layout(fig, "Global Market Topography (2D)")
    st.plotly_chart(fig, use_container_width=True)

# Module 2
with tabs[1]:
    if trades.empty or "product" not in trades.columns:
        st.info("Trades unavailable for counterparty edge.")
    else:
        m = prices[["product", "t", "mid_price"]].drop_duplicates(["product", "t"]).sort_values(["product", "t"]).copy()
        m["future_mid"] = m.groupby("product")["mid_price"].shift(-horizon)
        m["edge_buy"] = (m["future_mid"] - m["mid_price"]) / (m["mid_price"] + 1e-12)
        tr = trades.merge(m[["product", "t", "edge_buy"]], on=["product", "t"], how="left").dropna(subset=["edge_buy"])
        tr["edge_sell"] = -tr["edge_buy"]
        b = tr.groupby("buyer")["edge_buy"].agg(["count", "mean"]).reset_index().rename(columns={"buyer": "trader"})
        s = tr.groupby("seller")["edge_sell"].agg(["count", "mean"]).reset_index().rename(columns={"seller": "trader"})
        out = pd.merge(b, s, on="trader", how="outer", suffixes=("_buy", "_sell")).fillna(0)
        out["n_total"] = out["count_buy"] + out["count_sell"]
        out["mean_edge"] = (
            out["mean_buy"] * out["count_buy"] + out["mean_sell"] * out["count_sell"]
        ) / (out["n_total"] + 1e-12)
        top = out.sort_values("mean_edge", ascending=False).head(20)
        st.dataframe(top[["trader", "n_total", "mean_edge"]], use_container_width=True)
        fig = go.Figure([go.Bar(x=top["trader"], y=top["mean_edge"], marker_color=NEON["cyan"])])
        set_plotly_layout(fig, "Counterparty Edge Leaderboard")
        st.plotly_chart(fig, use_container_width=True)
        strategy["counterparty_edge"] = top[["trader", "n_total", "mean_edge"]].head(10).to_dict("records")

# Module 3
with tabs[2]:
    if not HAS_COINT:
        st.warning("statsmodels not available. Cointegration scan skipped.")
        strategy["stat_arb"] = {"pairs": []}
    else:
        pivot = (
            prices[["product", "t", "mid_price"]]
            .dropna()
            .pivot_table(index="t", columns="product", values="mid_price", aggfunc="last")
            .sort_index()
        )
        candidates = []
        attempted_pairs = 0
        skipped_short = 0
        for a, b in itertools.combinations([c for c in products if c in pivot.columns], 2):
            attempted_pairs += 1
            pair = pivot[[a, b]].dropna()
            if len(pair) < 200:
                skipped_short += 1
                continue
            y = np.log(pair[a].values + 1e-12)
            x = np.log(pair[b].values + 1e-12)
            try:
                _, p, _ = coint(y, x)
                candidates.append((float(p), a, b))
            except Exception:
                pass
        candidates.sort(key=lambda z: z[0])
        best = candidates[:3]
        st.caption(
            f"Attempted pairs: {attempted_pairs} | "
            f"with enough overlap (>=200): {attempted_pairs - skipped_short} | "
            f"cointegration candidates: {len(candidates)}"
        )

        if not best:
            st.info(
                "No cointegrated pairs found for the current uploaded data/rounds under the current overlap threshold."
            )
            # Fallback: show top correlated pairs so user still gets guidance.
            corr = pivot.corr().replace([np.inf, -np.inf], np.nan)
            rows_fallback = []
            cols = [c for c in products if c in corr.columns]
            for a, b in itertools.combinations(cols, 2):
                v = corr.loc[a, b]
                if pd.notna(v):
                    rows_fallback.append((a, b, float(v)))
            rows_fallback.sort(key=lambda t: abs(t[2]), reverse=True)
            top_corr = pd.DataFrame(rows_fallback[:10], columns=["a", "b", "corr"])
            if not top_corr.empty:
                st.write("Fallback: top correlated pairs")
                st.dataframe(top_corr, use_container_width=True)
            strategy["stat_arb"] = {"pairs": []}
        else:
            st.write("Top pairs:", best)
            rows = []
            for pval, a, b in best:
                pair = pivot[[a, b]].dropna()
                pair = pair.apply(pd.to_numeric, errors="coerce").dropna()
                t = pair.index.values
                y = np.log(pair[a].values + 1e-12)
                x = np.log(pair[b].values + 1e-12)
                intercept, beta, resid = kalman_beta(y, x)
                rs = pd.Series(resid)
                win = min(300, max(60, len(rs) // 3))
                z = (rs - rs.rolling(win, min_periods=max(20, win // 5)).mean()) / (
                    rs.rolling(win, min_periods=max(20, win // 5)).std(ddof=0) + 1e-12
                )
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12)
                fig.add_trace(go.Scatter(x=t, y=z, mode="lines", line=dict(color=NEON["cyan"]), name="z"), row=1, col=1)
                for s in (1, 2, 3):
                    fig.add_hline(y=s, line_dash="dash", line_color=NEON["red"], row=1, col=1)
                    fig.add_hline(y=-s, line_dash="dash", line_color=NEON["red"], row=1, col=1)
                fig.add_trace(go.Histogram(x=z.dropna(), nbinsx=60, marker_color=NEON["magenta"]), row=2, col=1)
                set_plotly_layout(fig, f"{a} vs {b} (p={pval:.3g})")
                st.plotly_chart(fig, use_container_width=True)
                rows.append(
                    {
                        "pair": [a, b],
                        "cointegration_pvalue": pval,
                        "beta_latest": float(beta[-1]),
                        "z_latest": float(z.dropna().iloc[-1]) if not z.dropna().empty else float("nan"),
                        "hurst_resid": hurst_rs(resid),
                    }
                )
            strategy["stat_arb"] = {"pairs": rows}

# Module 4
with tabs[3]:
    pick = st.selectbox("Product for microstructure", options=products, index=0)
    p = prices[prices["product"] == pick].sort_values("t").copy()
    bid_sum = p[["bid_volume_1", "bid_volume_2", "bid_volume_3"]].sum(axis=1, skipna=True)
    ask_sum = p[["ask_volume_1", "ask_volume_2", "ask_volume_3"]].sum(axis=1, skipna=True)
    p["obi"] = (bid_sum - ask_sum) / (bid_sum + ask_sum + 1e-12)
    d_bid_qty = p["bid_volume_1"].diff().fillna(0)
    d_ask_qty = p["ask_volume_1"].diff().fillna(0)
    d_bid_px = p["bid_price_1"].diff().fillna(0)
    d_ask_px = p["ask_price_1"].diff().fillna(0)
    p["ofi"] = np.where(d_bid_px >= 0, d_bid_qty, -d_bid_qty) - np.where(d_ask_px <= 0, d_ask_qty, -d_ask_qty)
    bpx = p[["bid_price_1", "bid_price_2", "bid_price_3"]].to_numpy(dtype=float)
    bv = p[["bid_volume_1", "bid_volume_2", "bid_volume_3"]].to_numpy(dtype=float)
    apx = p[["ask_price_1", "ask_price_2", "ask_price_3"]].to_numpy(dtype=float)
    av = p[["ask_volume_1", "ask_volume_2", "ask_volume_3"]].to_numpy(dtype=float)
    vb = np.nansum(bpx * bv, axis=1) / (np.nansum(bv, axis=1) + 1e-12)
    va = np.nansum(apx * av, axis=1) / (np.nansum(av, axis=1) + 1e-12)
    p["true_vwap"] = 0.5 * (vb + va)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=p["t"], y=p["obi"], mode="lines", line=dict(color=NEON["cyan"]), name="OBI"), row=1, col=1)
    fig.add_trace(go.Scatter(x=p["t"], y=p["ofi"], mode="lines", line=dict(color=NEON["magenta"]), name="OFI"), row=2, col=1)
    fig.add_trace(
        go.Scatter(x=p["t"], y=p["true_vwap"], mode="lines", line=dict(color=NEON["green"]), name="True VWAP"), row=3, col=1
    )
    fig.add_trace(go.Scatter(x=p["t"], y=p["mid_price"], mode="lines", line=dict(color=NEON["red"]), name="Mid"), row=3, col=1)
    set_plotly_layout(fig, f"Microstructure: {pick}")
    st.plotly_chart(fig, use_container_width=True)
    strategy["microstructure"] = {
        "product": pick,
        "obi_latest": float(p["obi"].iloc[-1]),
        "ofi_latest": float(p["ofi"].iloc[-1]),
        "true_vwap_latest": float(p["true_vwap"].iloc[-1]),
        "mid_latest": float(p["mid_price"].iloc[-1]),
    }

# Module 5
with tabs[4]:
    local = "ORCHIDS" if "ORCHIDS" in products else ("MAGNIFICENT_MACARONS" if "MAGNIFICENT_MACARONS" in products else None)
    if local is None or obs.empty:
        st.info("No environmental fair-value product/observations available.")
    else:
        loc = prices[prices["product"] == local][["t", "mid_price"]].drop_duplicates("t")
        o = obs.copy()
        if "humidity" not in o.columns:
            o["humidity"] = 0.0
        req = ["t", "bidPrice", "askPrice", "transportFees", "exportTariff", "importTariff", "sunlightIndex", "humidity"]
        miss = [c for c in req if c not in o.columns]
        if miss:
            st.warning(f"Missing observation columns: {miss}")
        else:
            o = o[req]
            df = loc.merge(o, on="t", how="inner").sort_values("t")
            df["implied_bid"] = df["bidPrice"] - df["exportTariff"] - df["transportFees"]
            df["implied_ask"] = df["askPrice"] + df["importTariff"] + df["transportFees"]
            df["center"] = 0.5 * (df["implied_bid"] + df["implied_ask"])
            y = (df["mid_price"] - df["center"]).to_numpy(float)
            sun = df["sunlightIndex"].to_numpy(float)
            hum = df["humidity"].to_numpy(float)
            sun_z = (sun - np.nanmean(sun)) / (np.nanstd(sun) + 1e-12)
            hum_z = (hum - np.nanmean(hum)) / (np.nanstd(hum) + 1e-12)
            X = np.column_stack([np.ones(len(df)), sun_z, hum_z])
            b, *_ = np.linalg.lstsq(X, y, rcond=None)
            df["fair"] = df["center"] + X @ b
            df["offset"] = df["mid_price"] - df["fair"]
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08)
            fig.add_trace(go.Scatter(x=df["t"], y=df["mid_price"], mode="lines", line=dict(color=NEON["green"]), name="Local"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df["t"], y=df["implied_bid"], mode="lines", line=dict(color=NEON["cyan"], dash="dot"), name="ImpBid"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df["t"], y=df["implied_ask"], mode="lines", line=dict(color=NEON["magenta"], dash="dot"), name="ImpAsk"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df["t"], y=df["fair"], mode="lines", line=dict(color=NEON["purple"]), name="Fair"), row=2, col=1)
            fig.add_trace(go.Scatter(x=df["t"], y=df["offset"], mode="lines", line=dict(color=NEON["yellow"]), name="Offset"), row=2, col=1)
            fig.add_trace(go.Scatter(x=df["t"], y=df["sunlightIndex"], mode="lines", line=dict(color=NEON["yellow"]), name="Sun"), row=3, col=1)
            fig.add_trace(go.Scatter(x=df["t"], y=df["humidity"], mode="lines", line=dict(color=NEON["magenta"]), name="Hum"), row=3, col=1)
            set_plotly_layout(fig, f"Environmental Fair Value ({local})")
            st.plotly_chart(fig, use_container_width=True)
            strategy["fair_value_environment"] = {
                "local_product": local,
                "fv_offset_latest": float(df["offset"].iloc[-1]),
                "fair_latest": float(df["fair"].iloc[-1]),
                "local_mid_latest": float(df["mid_price"].iloc[-1]),
                "sunlight_latest": float(df["sunlightIndex"].iloc[-1]),
                "humidity_latest": float(df["humidity"].iloc[-1]),
            }

# Module 6
with tabs[5]:
    pick6 = st.selectbox("Product for spectral analysis", options=products, index=0, key="spec_prod")
    s = prices[prices["product"] == pick6].sort_values("t").copy()
    s["ret"] = np.log(s["mid_price"] + 1e-12).diff()
    rv = s["ret"].dropna().to_numpy(float)
    W = min(512, max(128, len(rv) // 4))
    if len(rv) < W + 10:
        st.info("Not enough data for rolling FFT.")
    else:
        step = max(1, (len(rv) - W) // 80)
        freq = np.fft.rfftfreq(W, d=1.0)
        spectra, times, domf = [], [], []
        for i in range(0, len(rv) - W, step):
            w = rv[i : i + W] - np.mean(rv[i : i + W])
            mag = np.abs(np.fft.rfft(w))
            mag = mag / (np.max(mag) + 1e-12)
            spectra.append(np.log10(mag + 1e-12))
            j = int(np.argmax(mag[1:]) + 1) if len(mag) > 1 else 0
            domf.append(float(freq[j]))
            times.append(float(s["t"].iloc[i + W]))
        hm = go.Figure(go.Heatmap(x=freq[1:], y=times, z=np.vstack(spectra)[:, 1:], colorscale="Turbo"))
        set_plotly_layout(hm, f"Rolling FFT Spectrogram ({pick6})")
        st.plotly_chart(hm, use_container_width=True)
        dom_period = [1 / f if f > 1e-12 else np.nan for f in domf]
        line = go.Figure(go.Scatter(x=times, y=dom_period, mode="lines", line=dict(color=NEON["green"])))
        set_plotly_layout(line, "Dominant Cycle Period")
        st.plotly_chart(line, use_container_width=True)
        strategy["alpha_original"]["spectral"] = {
            "product": pick6,
            "dominant_freq_latest": float(domf[-1]) if domf else None,
            "dominant_period_latest": float(dom_period[-1]) if dom_period else None,
        }

# Module 7
with tabs[6]:
    pick7 = st.selectbox("Product for entropy", options=products, index=0, key="ent_prod")
    p = prices[prices["product"] == pick7].sort_values("t").copy()
    vols = p[["bid_volume_1", "bid_volume_2", "bid_volume_3", "ask_volume_1", "ask_volume_2", "ask_volume_3"]].to_numpy(float)
    vols = np.where(np.isfinite(vols), vols, 0.0)
    probs = vols / (np.sum(vols, axis=1, keepdims=True) + 1e-12)
    probs = np.clip(probs, 1e-12, 1.0)
    p["entropy"] = -(probs * np.log(probs)).sum(axis=1)
    p["future_mid"] = p["mid_price"].shift(-horizon)
    p["future_ret"] = (p["future_mid"] - p["mid_price"]) / (p["mid_price"] + 1e-12)
    sub = p.dropna(subset=["entropy", "future_ret"]).copy()
    if sub.empty:
        st.info("Not enough entropy data.")
    else:
        corr = float(sub["entropy"].corr(sub["future_ret"]))
        fig = make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.1)
        fig.add_trace(go.Scatter(x=sub["t"], y=sub["entropy"], mode="lines", line=dict(color=NEON["cyan"])), row=1, col=1)
        fig.add_trace(
            go.Scatter(
                x=sub["entropy"], y=sub["future_ret"], mode="markers", marker=dict(color=NEON["magenta"], size=4, opacity=0.5)
            ),
            row=2,
            col=1,
        )
        set_plotly_layout(fig, f"Entropy vs Future Return (corr={corr:.3g})")
        st.plotly_chart(fig, use_container_width=True)
        strategy["alpha_original"]["entropy"] = {"product": pick7, "corr_entropy_future_ret": corr}

# Module 8
with tabs[7]:
    pick8 = st.selectbox("Product for liquidity void", options=products, index=0, key="void_prod")
    p = prices[prices["product"] == pick8].sort_values("t").copy()
    p["future_mid"] = p["mid_price"].shift(-horizon)
    p["abs_future_ret"] = ((p["future_mid"] - p["mid_price"]) / (p["mid_price"] + 1e-12)).abs()
    sub = p.dropna(subset=["abs_future_ret"]).copy()
    if sub.empty:
        st.info("Not enough void data.")
    else:
        lvl = ["bid_volume_1", "bid_volume_2", "bid_volume_3", "ask_volume_1", "ask_volume_2", "ask_volume_3"]
        mat = []
        for c in lvl:
            s = sub[c].astype(float)
            q10, q90 = float(s.quantile(0.1)), float(s.quantile(0.9))
            den = (q90 - q10) if (q90 - q10) != 0 else 1.0
            mat.append(np.clip((q90 - s) / den, 0.0, 1.0).to_numpy(float))
        M = np.column_stack(mat)
        sub["void_score"] = M.mean(axis=1)
        expl_thr = float(sub["abs_future_ret"].quantile(0.8))
        sub["explosive"] = (sub["abs_future_ret"] >= expl_thr).astype(int)
        vthr = float(sub["void_score"].quantile(0.8))
        sub["void_high"] = (sub["void_score"] >= vthr).astype(int)
        p_hi = float(sub.loc[sub["void_high"] == 1, "explosive"].mean())
        p_lo = float(sub.loc[sub["void_high"] == 0, "explosive"].mean())
        corr = float(sub["void_score"].corr(sub["abs_future_ret"]))
        st.metric("P(explosive | void_high)", f"{p_hi:.3f}")
        st.metric("P(explosive | void_low)", f"{p_lo:.3f}")
        st.caption(f"corr(void_score, abs_future_ret)={corr:.3g}")
        hm = go.Figure(go.Heatmap(x=sub["t"], y=["bid1", "bid2", "bid3", "ask1", "ask2", "ask3"], z=M.T, colorscale="Turbo"))
        set_plotly_layout(hm, f"Liquidity Void Heatmap ({pick8})")
        st.plotly_chart(hm, use_container_width=True)
        strategy["alpha_original"]["liquidity_void"] = {
            "product": pick8,
            "prob_explosive_given_void_high": p_hi,
            "prob_explosive_given_void_low": p_lo,
            "corr_void_abs_future_ret": corr,
        }

# Final JSON
with tabs[8]:
    st.subheader("Strategy Dictionary JSON")
    st.code(json.dumps(strategy, indent=2), language="json")
    st.download_button(
        label="Download strategy.json",
        data=json.dumps(strategy, indent=2),
        file_name="strategy.json",
        mime="application/json",
    )

