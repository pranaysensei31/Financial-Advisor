import re
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

try:
    import yfinance as yf
    YF_AVAILABLE = True
except Exception:
    YF_AVAILABLE = False


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="FinSight | Commodities",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================
# STYLES (MATCH YOUR UI)
# =========================
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at 10% 0%, #0b1220 0%, #050814 45%, #050814 100%);
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
        color: #FFFFFF;
    }

    .block-container {
        max-width: 1300px;
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
    }

    .hero {
        border-radius: 26px;
        padding: 28px 28px;
        background: linear-gradient(180deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0.03) 100%);
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 20px 60px rgba(0,0,0,0.35);
        margin-bottom: 18px;
    }

    .hero h1 {
        font-size: 44px;
        font-weight: 950;
        line-height: 1.05;
        margin: 0 0 10px 0;
        color: #FFFFFF;
        letter-spacing: -0.8px;
    }

    .hero p {
        margin: 0;
        font-size: 15px;
        color: rgba(255,255,255,0.82);
        line-height: 1.6;
        max-width: 920px;
    }

    .tag {
        padding: 9px 14px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.05);
        font-size: 12px;
        font-weight: 800;
        color: rgba(255,255,255,0.9);
        display: inline-block;
        margin-top: 12px;
        margin-right: 8px;
    }

    .panel {
        border-radius: 26px;
        padding: 18px 18px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 20px 60px rgba(0,0,0,0.35);
    }

    .panel-title {
        font-size: 15px;
        font-weight: 950;
        color: #FFFFFF;
        margin: 0 0 12px 0;
    }

    .stMarkdown, .stMarkdown * {
        color: rgba(255,255,255,0.96) !important;
        opacity: 1 !important;
    }

    div[data-testid="stCaptionContainer"],
    div[data-testid="stCaptionContainer"] * {
        color: rgba(255,255,255,0.88) !important;
        opacity: 1 !important;
        font-weight: 850 !important;
    }

    label,
    div[data-testid="stWidgetLabel"] *,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stTextInput"] label,
    div[data-testid="stSlider"] label,
    div[data-testid="stTextArea"] label {
        color: #FFFFFF !important;
        opacity: 1 !important;
        font-weight: 900 !important;
        letter-spacing: 0.2px !important;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    textarea {
        background: rgba(10, 15, 35, 0.95) !important;
        border: 1px solid rgba(255,255,255,0.14) !important;
        border-radius: 16px !important;
        box-shadow: none !important;
    }

    div[data-baseweb="input"] input,
    textarea {
        color: #ffffff !important;
        font-weight: 950 !important;
        background: transparent !important;
    }

    div[data-baseweb="select"] span {
        color: #ffffff !important;
        font-weight: 950 !important;
    }

    ::placeholder {
        color: rgba(255,255,255,0.55) !important;
        font-weight: 800 !important;
    }

    div[data-testid="stMetricLabel"] * {
        color: rgba(255,255,255,0.90) !important;
        opacity: 1 !important;
        font-weight: 900 !important;
    }

    div[data-testid="stMetricValue"] * {
        color: #ffffff !important;
        opacity: 1 !important;
        font-weight: 950 !important;
    }

    div[data-testid="stMetricDelta"] * {
        opacity: 1 !important;
        font-weight: 950 !important;
    }

    button[data-baseweb="tab"] {
        color: rgba(255,255,255,0.85) !important;
        font-weight: 900 !important;
        opacity: 1 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #ffffff !important;
        font-weight: 950 !important;
    }
    div[data-baseweb="tab-highlight"] {
        background: linear-gradient(90deg, #3B82F6, #22C55E) !important;
        height: 3px !important;
        border-radius: 999px !important;
    }

    div.stButton > button,
    div[data-testid="stDownloadButton"] > button {
        border-radius: 18px !important;
        height: 56px !important;
        font-size: 15px !important;
        font-weight: 950 !important;
        color: #0B1220 !important;
        background: linear-gradient(135deg, #22C55E 0%, #3B82F6 100%) !important;
        border: none !important;
        width: 100%;
        box-shadow: 0 18px 60px rgba(0,0,0,0.45);
        transition: 0.2s;
    }

    div.stButton > button:hover,
    div[data-testid="stDownloadButton"] > button:hover {
        transform: translateY(-1px);
        filter: brightness(1.06);
    }
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="hero">
    <h1>Commodities — Live Prices (USD + INR), Risk & Comparison</h1>
    <p>
        Track global commodities like Gold, Silver, Crude Oil, Natural Gas and base metals.
        Generate price trend, risk metrics (volatility, drawdown, Sharpe), compare multiple commodities and export CSV.
    </p>
    <span class="tag">USD → INR Conversion</span>
    <span class="tag">Gold</span>
    <span class="tag">Silver</span>
    <span class="tag">Crude Oil</span>
    <span class="tag">Chart + CSV</span>
</div>
""", unsafe_allow_html=True)


# =========================
# COMMODITIES MASTER LIST
# =========================
COMMODITIES = {
    "Gold (Futures)": "GC=F",
    "Silver (Futures)": "SI=F",
    "Crude Oil WTI (Futures)": "CL=F",
    "Brent Oil (Futures)": "BZ=F",
    "Natural Gas (Futures)": "NG=F",
    "Copper (Futures)": "HG=F",
    "Platinum (Futures)": "PL=F",
    "Palladium (Futures)": "PA=F",
    "Corn (Futures)": "ZC=F",
    "Wheat (Futures)": "ZW=F",
    "Soybean (Futures)": "ZS=F",
}

EXTRA_HINTS = {
    "Gold ETF (US)": "GLD",
    "Silver ETF (US)": "SLV",
}


# =========================
# HELPERS
# =========================
def compute_returns(series: pd.Series) -> pd.Series:
    return series.pct_change().dropna()


def sharpe_ratio(returns: pd.Series, rf_annual: float = 0.0) -> float:
    if len(returns) < 2:
        return float("nan")
    rf_daily = rf_annual / 252
    excess = returns - rf_daily
    if excess.std() == 0:
        return float("nan")
    return float((excess.mean() / excess.std()) * math.sqrt(252))


def max_drawdown(series: pd.Series) -> float:
    prices = series.values
    peak = -np.inf
    mdd = 0.0
    for p in prices:
        peak = max(peak, p)
        dd = (peak - p) / peak if peak > 0 else 0.0
        mdd = max(mdd, dd)
    return float(mdd)


def risk_label(vol: float) -> str:
    if vol is None or np.isnan(vol):
        return "N/A"
    if vol < 0.18:
        return "LOW"
    if vol < 0.30:
        return "MEDIUM"
    return "HIGH"


def extract_tickers(text: str) -> list:
    raw = re.findall(r"\b[A-Za-z0-9=\^]{1,15}(?:\.[A-Za-z]{1,6})?\b", text or "")
    out = []
    for r in raw:
        r2 = r.strip()
        if not r2:
            continue
        out.append(r2)
    # remove duplicates, keep order
    return list(dict.fromkeys(out))


@st.cache_data(ttl=600)
def get_fx_rate(symbol: str) -> float:
    """
    Example symbols:
      USDINR=X
      EURINR=X
    """
    if not YF_AVAILABLE:
        return 0.0
    try:
        fx = yf.download(symbol, period="5d", interval="1d", progress=False)
        if fx is None or fx.empty:
            return 0.0
        close = fx["Close"].dropna()
        if close.empty:
            return 0.0
        return float(close.iloc[-1])
    except Exception:
        return 0.0


def convert_to_inr(price: float, currency: str):
    """
    Returns: (price_inr, fx_rate, fx_symbol)
    """
    try:
        currency = (currency or "").upper().strip()
        if currency == "USD":
            fx = get_fx_rate("USDINR=X")
            if fx > 0:
                return price * fx, fx, "USDINR=X"
            return None, None, None

        if currency == "EUR":
            fx = get_fx_rate("EURINR=X")
            if fx > 0:
                return price * fx, fx, "EURINR=X"
            return None, None, None

        if currency in {"INR", "₹"}:
            return price, 1.0, "INR"

        return None, None, None
    except Exception:
        return None, None, None


@st.cache_data(ttl=600)
def fetch_prices(ticker: str, days: int) -> pd.DataFrame:
    if not YF_AVAILABLE:
        return pd.DataFrame()

    try:
        df = yf.download(ticker, period=f"{days}d", interval="1d", progress=False)
        if df is None or df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        if "Date" not in df.columns or "Close" not in df.columns:
            return pd.DataFrame()

        df.rename(columns={"Date": "date", "Close": "close"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")

        df = df.dropna(subset=["date", "close"]).sort_values("date")
        return df[["date", "close"]]
    except Exception:
        return pd.DataFrame()


def live_quote(ticker: str) -> dict:
    if not YF_AVAILABLE:
        return {}

    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="2d", interval="1d")
        if hist is None or hist.empty:
            return {}

        last_close = float(hist["Close"].iloc[-1])
        prev_close = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else last_close

        change = last_close - prev_close
        pct = (change / prev_close) * 100 if prev_close != 0 else 0.0

        info = tk.info or {}
        currency = info.get("currency", "") or ""

        # INR Conversion
        price_inr, fx_rate, fx_symbol = convert_to_inr(last_close, currency)

        return {
            "price": last_close,
            "prev": prev_close,
            "change": change,
            "pct": pct,
            "currency": currency,
            "price_inr": price_inr,
            "fx_rate": fx_rate,
            "fx_symbol": fx_symbol,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception:
        return {}


def build_compare_table(tickers: list, days: int, rf_annual: float) -> pd.DataFrame:
    rows = []
    for t in tickers[:8]:
        df = fetch_prices(t, days)
        if df.empty:
            continue

        rets = compute_returns(df["close"])
        vol = float(rets.std() * math.sqrt(252)) if len(rets) > 2 else float("nan")
        sr = float(sharpe_ratio(rets, rf_annual)) if len(rets) > 2 else float("nan")
        mdd = float(max_drawdown(df["close"])) if not df.empty else float("nan")

        rows.append({
            "Ticker": t,
            "Last Close": float(df["close"].iloc[-1]),
            "Volatility": vol,
            "Sharpe": sr,
            "Max Drawdown": mdd,
            "Risk": risk_label(vol),
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(by="Sharpe", ascending=False)


def build_price_chart(tickers: list, days: int) -> go.Figure:
    fig = go.Figure()
    for t in tickers[:8]:
        df = fetch_prices(t, days)
        if df.empty:
            continue
        fig.add_trace(go.Scatter(x=df["date"], y=df["close"], mode="lines", name=t))

    fig.update_layout(
        title=f"Commodity Price Trend (Last {days} Days)",
        xaxis_title="Date",
        yaxis_title="Close Price",
        height=460
    )
    return fig


def build_csv_dataset(tickers: list, days: int) -> str:
    if not YF_AVAILABLE:
        return ""

    data_rows = []
    for t in tickers[:8]:
        df = fetch_prices(t, days)
        if df.empty:
            continue
        temp = df.copy()
        temp["ticker"] = t
        data_rows.append(temp)

    if not data_rows:
        return ""

    out = pd.concat(data_rows, ignore_index=True)
    out = out[["date", "ticker", "close"]]
    return out.to_csv(index=False)


# =========================
# STATE
# =========================
if "run_commodities" not in st.session_state:
    st.session_state["run_commodities"] = False


# =========================
# UI: LEFT / RIGHT
# =========================
left, right = st.columns([1.05, 1.65], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Input</div>', unsafe_allow_html=True)

    st.caption("Quick select (recommended)")
    quick = st.selectbox("Commodity", ["None"] + list(COMMODITIES.keys()), index=0)

    default_tickers = COMMODITIES[quick] if quick != "None" else "GC=F SI=F CL=F"

    query = st.text_input(
        "Commodities tickers",
        value=default_tickers,
        placeholder="Example: GC=F SI=F CL=F NG=F HG=F"
    )

    st.caption("Tip: You can also type ETFs like GLD, SLV")

    days = st.slider("History window (days)", min_value=7, max_value=3650, value=365, step=7)

    rf_ui = st.slider("Risk-free rate (annual, %)", min_value=0.0, max_value=12.0, value=4.0, step=0.25)
    rf_annual = rf_ui / 100.0

    tickers = extract_tickers(query)

    st.caption("Detected tickers: " + (", ".join(tickers[:10]) if tickers else "none"))

    st.write("")
    st.markdown("#### Live Prices (Top 4)")

    if tickers:
        cols = st.columns(min(4, len(tickers)))
        for i, t in enumerate(tickers[:4]):
            q = live_quote(t)

            if not q:
                cols[i].metric(label=t, value="N/A")
                continue

            delta = f"{q['change']:.2f} ({q['pct']:.2f}%)"
            currency = q.get("currency", "").upper()

            # If USD/EUR -> show USD + INR
            if currency in {"USD", "EUR"} and q.get("price_inr") is not None and q.get("fx_rate") is not None:
                symbol = "$" if currency == "USD" else "€"
                cols[i].metric(
                    label=t,
                    value=f"{symbol}{q['price']:.2f} | ₹{q['price_inr']:.2f}",
                    delta=delta
                )
                cols[i].caption(f"FX: {q.get('fx_symbol')} = {q.get('fx_rate'):.2f}")

            # If already INR or unknown currency
            else:
                cols[i].metric(
                    label=t,
                    value=f"{q['price']:.2f} {currency}",
                    delta=delta
                )
    else:
        st.info("Enter commodity tickers to see live prices.")

    st.write("")
    if st.button("Generate Commodities Report"):
        st.session_state["run_commodities"] = True

    st.markdown('</div>', unsafe_allow_html=True)


with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Results</div>', unsafe_allow_html=True)

    if not YF_AVAILABLE:
        st.error("yfinance is not available. Install it in your venv:  pip install yfinance")
    else:
        if not st.session_state["run_commodities"]:
            st.info("Enter tickers and click **Generate Commodities Report**.")
        else:
            if not tickers:
                st.error("Please enter at least one valid commodity ticker.")
                st.session_state["run_commodities"] = False
            else:
                primary = tickers[0]
                df_primary = fetch_prices(primary, days)

                if df_primary.empty or len(df_primary) < 10:
                    st.error("Not enough data for the selected commodity.")
                    st.session_state["run_commodities"] = False
                else:
                    rets = compute_returns(df_primary["close"])

                    vol = float(rets.std() * math.sqrt(252)) if len(rets) > 2 else float("nan")
                    sr = float(sharpe_ratio(rets, rf_annual)) if len(rets) > 2 else float("nan")
                    mdd = float(max_drawdown(df_primary["close"]))

                    risk = risk_label(vol)

                    comp_df = build_compare_table(tickers, days, rf_annual) if len(tickers) >= 2 else pd.DataFrame()
                    fig = build_price_chart(tickers, days)
                    csv_data = build_csv_dataset(tickers, days)

                    t0, t1, t2 = st.tabs(["Overview", "Compare", "Chart + CSV"])

                    with t0:
                        st.subheader("Commodity Overview")
                        a1, a2, a3, a4 = st.columns(4)

                        a1.metric("Primary", primary)
                        a2.metric("Volatility (ann.)", f"{vol:.2%}" if not np.isnan(vol) else "N/A")
                        a3.metric("Max Drawdown", f"{mdd:.2%}" if not np.isnan(mdd) else "N/A")
                        a4.metric("Sharpe", f"{sr:.2f}" if not np.isnan(sr) else "N/A")

                        st.caption(f"Risk Level: **{risk}**")

                    with t1:
                        st.subheader("Comparison")
                        if comp_df.empty:
                            st.info("Enter 2+ tickers to compare.")
                        else:
                            st.dataframe(comp_df, use_container_width=True)
                            best = comp_df.iloc[0]["Ticker"]
                            worst = comp_df.iloc[-1]["Ticker"]
                            st.caption(f"Top risk-adjusted: **{best}** | Weakest: **{worst}**")

                    with t2:
                        st.subheader("Price Trend")
                        st.plotly_chart(fig, use_container_width=True)

                        st.write("")
                        st.subheader("Export CSV")
                        if csv_data:
                            st.download_button(
                                "Download commodities_report.csv",
                                data=csv_data,
                                file_name="commodities_report.csv",
                                mime="text/csv"
                            )
                        else:
                            st.info("CSV not available for these tickers.")

                    st.session_state["run_commodities"] = False

    st.markdown("</div>", unsafe_allow_html=True)



st.caption("FinSight • Commodities • Yahoo Finance (yfinance)")




