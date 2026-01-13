import os
import re
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

# ------------------------------------------------------------
# Optional libraries
# ------------------------------------------------------------
try:
    import yfinance as yf
    YF_AVAILABLE = True
except Exception:
    YF_AVAILABLE = False

try:
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="FinSight | AI Financial Advisor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if "chat_memory" not in st.session_state:
    st.session_state["chat_memory"] = []

if "run_full_report" not in st.session_state:
    st.session_state["run_full_report"] = False


# ============================================================
# CONSTANTS
# ============================================================
SMART_TICKERS = {
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Microsoft": "MSFT",
    "NVIDIA": "NVDA",
    "Amazon": "AMZN",
    "Google": "GOOGL",
    "Meta": "META",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "Reliance": "RELIANCE.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Zomato": "ZOMATO.NS",
}


# ============================================================
# HELPERS
# ============================================================
def extract_tickers(query: str) -> list:
    raw = re.findall(r"\b[A-Za-z]{1,12}(?:\.[A-Za-z]{1,6})?\b", query)

    blacklist = {
        "and", "the", "with", "show", "what", "this", "that",
        "csv", "report", "for", "give", "export", "generate", "data",
        "analyze", "compare", "visualize", "plot", "trend", "me", "please",
        "full", "advisor", "risk", "prices"
    }

    tickers = []
    for r in raw:
        r_clean = r.upper()
        if r_clean.lower() in blacklist:
            continue
        if re.match(r"^[A-Z]{1,12}(\.[A-Z]{1,6})?$", r_clean):
            tickers.append(r_clean)

    return list(dict.fromkeys(tickers))


def compute_returns(close_series: pd.Series) -> pd.Series:
    return close_series.pct_change().dropna()


def sharpe_ratio(returns: pd.Series, rf_annual: float = 0.0) -> float:
    if len(returns) < 2:
        return float("nan")
    rf_daily = rf_annual / 252
    excess = returns - rf_daily
    if excess.std() == 0:
        return float("nan")
    return float((excess.mean() / excess.std()) * math.sqrt(252))


def max_drawdown(close_series: pd.Series) -> float:
    prices = close_series.values
    peak = -np.inf
    mdd = 0.0
    for p in prices:
        peak = max(peak, p)
        dd = (peak - p) / peak if peak > 0 else 0.0
        mdd = max(mdd, dd)
    return float(mdd)


def format_market_cap(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "N/A"
        x = float(x)
        if x >= 1e12:
            return f"{x/1e12:.2f}T"
        if x >= 1e9:
            return f"{x/1e9:.2f}B"
        if x >= 1e6:
            return f"{x/1e6:.2f}M"
        return f"{x:.0f}"
    except Exception:
        return "N/A"


@st.cache_data(ttl=600)
def get_fx_rate(symbol: str) -> float:
    """Yahoo FX symbols: USDINR=X, EURINR=X"""
    if not YF_AVAILABLE:
        return 0.0
    try:
        fx = yf.download(symbol, period="5d", interval="1d", progress=False, threads=True)
        if fx is None or fx.empty:
            return 0.0
        return float(fx["Close"].dropna().iloc[-1])
    except Exception:
        return 0.0


@st.cache_data(ttl=60)
def get_live_quote(ticker: str) -> dict:
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
        change_pct = (change / prev_close) * 100 if prev_close != 0 else 0.0

        info = tk.fast_info if hasattr(tk, "fast_info") else {}
        currency = ""
        try:
            currency = info.get("currency", "")
        except Exception:
            currency = ""

        # fallback currency lookup only if needed
        if not currency:
            try:
                currency = (tk.info or {}).get("currency", "")
            except Exception:
                currency = ""

        price_inr = None
        fx_rate = None

        if currency == "USD":
            fx_rate = get_fx_rate("USDINR=X")
            if fx_rate > 0:
                price_inr = last_close * fx_rate
        elif currency == "EUR":
            fx_rate = get_fx_rate("EURINR=X")
            if fx_rate > 0:
                price_inr = last_close * fx_rate

        return {
            "ticker": ticker,
            "price": last_close,
            "prev_close": prev_close,
            "change": change,
            "change_pct": change_pct,
            "currency": currency,
            "price_inr": price_inr,
            "fx_rate_to_inr": fx_rate,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception:
        return {}


# ============================================================
# FAST: DOWNLOAD ONCE (KEY OPTIMIZATION)
# ============================================================
@st.cache_data(ttl=600)
def download_all_prices(tickers: list, days: int) -> pd.DataFrame:
    """
    Downloads ALL tickers in one call.
    Returns MultiIndex dataframe like:
      columns: (Ticker, Open/High/Low/Close/Volume)
    """
    if not YF_AVAILABLE or not tickers:
        return pd.DataFrame()

    tickers = tickers[:5]
    try:
        df = yf.download(
            tickers=" ".join(tickers),
            period=f"{days}d",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True
        )
        if df is None or df.empty:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()


def extract_close_series(prices_df: pd.DataFrame, ticker: str) -> pd.Series:
    """Extracts close series from bulk dataframe safely."""
    if prices_df is None or prices_df.empty:
        return pd.Series(dtype=float)

    try:
        if isinstance(prices_df.columns, pd.MultiIndex):
            if ticker in prices_df.columns.get_level_values(0):
                s = prices_df[ticker]["Close"].dropna()
                s.name = ticker
                return s
        else:
            # Single ticker case (rare)
            if "Close" in prices_df.columns:
                return prices_df["Close"].dropna()
    except Exception:
        return pd.Series(dtype=float)

    return pd.Series(dtype=float)


def build_price_chart_from_bulk(prices_df: pd.DataFrame, tickers: list) -> go.Figure:
    fig = go.Figure()
    for t in tickers[:5]:
        close_s = extract_close_series(prices_df, t)
        if close_s.empty:
            continue
        fig.add_trace(go.Scatter(x=close_s.index, y=close_s.values, mode="lines", name=t))

    fig.update_layout(
        title=f"Price Trend (Last {len(prices_df)} Trading Days)",
        xaxis_title="Date",
        yaxis_title="Close Price"
    )
    return fig


def compare_table_from_bulk(prices_df: pd.DataFrame, tickers: list, rf_annual: float) -> pd.DataFrame:
    rows = []
    for t in tickers[:5]:
        close_s = extract_close_series(prices_df, t)
        if close_s.empty or len(close_s) < 5:
            continue
        rets = compute_returns(close_s)
        rows.append({
            "Ticker": t,
            "Last Close": float(close_s.iloc[-1]),
            "Volatility": float(rets.std() * math.sqrt(252)) if len(rets) >= 2 else float("nan"),
            "Sharpe": float(sharpe_ratio(rets, rf_annual)) if len(rets) >= 2 else float("nan"),
            "Max Drawdown": float(max_drawdown(close_s)),
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(by="Sharpe", ascending=False)


def build_csv_from_bulk(prices_df: pd.DataFrame, tickers: list) -> str:
    if prices_df is None or prices_df.empty:
        return ""

    records = []
    if not isinstance(prices_df.columns, pd.MultiIndex):
        temp = prices_df.reset_index()
        if "Date" not in temp.columns:
            # index is date
            temp.rename(columns={"index": "Date"}, inplace=True)
        t = tickers[0] if tickers else "TICKER"
        for _, row in temp.iterrows():
            records.append({"date": row["Date"], "ticker": t, "close": row.get("Close")})
    else:
        for t in tickers[:5]:
            if t not in prices_df.columns.get_level_values(0):
                continue
            temp = prices_df[t][["Close"]].dropna().reset_index()
            for _, row in temp.iterrows():
                records.append({"date": row["Date"], "ticker": t, "close": row["Close"]})

    out = pd.DataFrame(records)
    if out.empty:
        return ""
    return out.to_csv(index=False)


# ============================================================
# SLOW: Company overview (lazy-loaded)
# ============================================================
@st.cache_data(ttl=1800)
def get_stock_summary(ticker: str) -> dict:
    if not YF_AVAILABLE:
        return {}
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}   # ⚠️ slow → but cached and lazy-loaded
        return {
            "ticker": ticker,
            "name": info.get("longName") or info.get("shortName") or ticker,
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap"),
            "pe": info.get("trailingPE"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "website": info.get("website"),
            "exchange": info.get("exchange"),
            "currency": info.get("currency"),
        }
    except Exception:
        return {}


def llm_advisor_summary(user_query: str, analysis_text: str) -> str:
    if not GROQ_AVAILABLE:
        return ""

    key = os.getenv("GROQ_API_KEY", "")
    if not key.strip():
        return ""

    try:
        chat = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0.3,
            groq_api_key=key
        )

        prompt = f"""
You are a professional financial advisor.
User tickers: {user_query}

Create a short professional advisor-style summary:
- insights
- risk
- suggestion (no guarantee)
- caution

ANALYSIS:
{analysis_text}
"""
        return chat.invoke([HumanMessage(content=prompt)]).content.strip()
    except Exception:
        return ""


# ============================================================
# UI THEME (DO NOT CHANGE UI)
# ============================================================
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
        font-size: 46px;
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
        max-width: 900px;
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

    /* DO NOT LET STREAMLIT GREY ANYTHING */
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

    /* Inputs should NOT be white */
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

    /* METRICS bright */
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

    /* Tabs */
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

    /* Buttons */
    div.stButton > button {
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

    div.stButton > button:hover {
        transform: translateY(-1px);
        filter: brightness(1.06);
    }
    /* Force TextArea to dark background (Fix white textbox issue) */
div[data-testid="stTextArea"] textarea {
    background: rgba(10, 15, 35, 0.95) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 16px !important;
    font-weight: 950 !important;
}

/* Also fix the outer wrapper of textarea */
div[data-testid="stTextArea"] > div {
    background: transparent !important;
}


</style>
""", unsafe_allow_html=True)


# ============================================================
# HERO
# ============================================================
st.markdown("""
<div class="hero">
    <h1>Stock insights that feel effortless.</h1>
    <p>
        Enter tickers and generate a full report: live prices, company overview, risk metrics,
        comparison, charts, CSV export. Portfolio mode is also available.
    </p>
    <span class="tag">One-click report</span>
    <span class="tag">USD/EUR → INR</span>
    <span class="tag">Risk metrics</span>
    <span class="tag">Charts</span>
</div>
""", unsafe_allow_html=True)


# ============================================================
# INPUT PANEL
# ============================================================
left, right = st.columns([1.05, 1.65], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Input</div>', unsafe_allow_html=True)

    selected_company = st.selectbox(
        "Quick search (optional)",
        ["None"] + list(SMART_TICKERS.keys()),
        index=0
    )

    default_q = SMART_TICKERS[selected_company] if selected_company != "None" else "AAPL TSLA MSFT"

    query = st.text_input(
        "Tickers / Query",
        value=default_q,
        placeholder="Example: AAPL TSLA MSFT | TCS.NS INFY.NS"
    )

    days = st.slider("History window (days)", min_value=3, max_value=365, value=90)

    rf_annual_ui = st.slider("Risk-free rate (annual, %)", min_value=0.0, max_value=12.0, value=4.0, step=0.25)
    rf_annual = rf_annual_ui / 100.0

    tickers = extract_tickers(query)
    tickers = tickers[:5]  # ✅ hard limit for speed
    st.caption("Detected tickers: " + (", ".join(tickers[:10]) if tickers else "none"))

    st.write("")
    st.markdown("#### Live Prices (Top 4)")

    if tickers:
        cols = st.columns(min(4, len(tickers)))
        for i, t in enumerate(tickers[:4]):
            q = get_live_quote(t)
            if not q:
                cols[i].metric(label=t, value="N/A")
            else:
                delta_str = f"{q['change']:.2f} ({q['change_pct']:.2f}%)"
                currency = q.get("currency", "")
                symbol = {"USD": "$", "EUR": "€"}.get(currency, "")

                if currency in ["USD", "EUR"] and q.get("price_inr") is not None:
                    cols[i].metric(
                        label=t,
                        value=f"{symbol}{q['price']:.2f} | ₹{q['price_inr']:.2f}",
                        delta=delta_str
                    )
                else:
                    cols[i].metric(
                        label=t,
                        value=f"{q['price']:.2f} {currency}",
                        delta=delta_str
                    )
    else:
        st.info("Enter tickers to view live prices.")

    st.write("")
    st.markdown("#### Generate")

    if st.button("Generate Full Advisor Report"):
        st.session_state["run_full_report"] = True

    st.write("")
    st.markdown("#### Portfolio Mode")

    portfolio_text = st.text_area(
        "Portfolio input (ticker weight)",
        value="AAPL 40\nTSLA 30\nMSFT 30",
        height=120
    )

    run_portfolio = st.button("Run portfolio analysis")

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# PORTFOLIO MODE
# ============================================================
def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def parse_portfolio_input(text: str):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    items = []
    for ln in lines:
        parts = ln.split()
        if len(parts) != 2:
            continue
        t = parts[0].upper().strip()
        w = safe_float(parts[1], None)
        if w is None:
            continue
        items.append((t, w))

    if not items:
        return [], np.array([])

    tickers_ = [t for t, _ in items]
    weights = np.array([w for _, w in items], dtype=float)

    if weights.sum() <= 0:
        return [], np.array([])

    weights = weights / weights.sum()
    return tickers_, weights


def portfolio_metrics_bulk(prices_df: pd.DataFrame, tickers_: list, weights: np.ndarray, rf_annual_: float):
    # build close price matrix
    close_map = {}
    for t in tickers_:
        s = extract_close_series(prices_df, t)
        if not s.empty:
            close_map[t] = s

    if len(close_map) < 2:
        return {}

    price_df = pd.DataFrame(close_map).dropna()
    if price_df.empty or len(price_df) < 7:
        return {}

    returns = price_df.pct_change().dropna()
    w = np.array(weights, dtype=float)

    # align weights with available tickers
    available = list(price_df.columns)
    idx_map = {t: i for i, t in enumerate(tickers_)}
    w2 = np.array([w[idx_map[t]] for t in available], dtype=float)
    w2 = w2 / w2.sum()

    port_returns = returns.dot(w2)

    vol = float(port_returns.std() * math.sqrt(252))
    sr = float(sharpe_ratio(port_returns, rf_annual_))
    total_return = float((1 + port_returns).prod() - 1)

    contrib = returns.mean() * w2
    top_contrib = contrib.idxmax()
    worst_contrib = contrib.idxmin()

    return {
        "volatility": vol,
        "sharpe": sr,
        "total_return": total_return,
        "top_contributor": top_contrib,
        "worst_contributor": worst_contrib,
        "returns_series": port_returns
    }


# ============================================================
# RESULTS PANEL
# ============================================================
with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Results</div>', unsafe_allow_html=True)

    if run_portfolio:
        ptickers, pweights = parse_portfolio_input(portfolio_text)

        if len(ptickers) < 2:
            st.error("Enter at least 2 tickers in portfolio mode.")
        else:
            with st.spinner("Downloading & analyzing portfolio..."):
                prices_df = download_all_prices(ptickers[:5], days=max(days, 90))
                pm = portfolio_metrics_bulk(prices_df, ptickers[:5], pweights, rf_annual)

            if not pm:
                st.error("Not enough market data to compute portfolio metrics.")
            else:
                st.success("Portfolio analysis complete.")

                a1, a2, a3, a4 = st.columns(4)
                a1.metric("Volatility (annualized)", f"{pm['volatility']:.2%}")
                a2.metric("Sharpe Ratio", f"{pm['sharpe']:.2f}")
                a3.metric("Return (approx)", f"{pm['total_return']:.2%}")
                a4.metric("Top Contributor", pm["top_contributor"])

                st.caption(f"Worst contributor: {pm['worst_contributor']}")

                figp = go.Figure()
                figp.add_trace(go.Scatter(
                    x=pm["returns_series"].index,
                    y=(1 + pm["returns_series"]).cumprod(),
                    mode="lines",
                    name="Portfolio Growth"
                ))
                figp.update_layout(
                    title="Portfolio Growth Curve",
                    xaxis_title="Date",
                    yaxis_title="Growth"
                )
                st.plotly_chart(figp, use_container_width=True)

    elif st.session_state["run_full_report"]:
        if not tickers:
            st.error("Please enter at least one valid ticker (example: AAPL).")
            st.session_state["run_full_report"] = False
        else:
            primary = tickers[0]

            with st.spinner("Downloading market data (fast mode)..."):
                prices_df = download_all_prices(tickers, days)

            # primary close series
            close_primary = extract_close_series(prices_df, primary)

            if close_primary.empty:
                rets = pd.Series(dtype=float)
                vol = float("nan")
                mdd = float("nan")
                sr = float("nan")
                risk_label = "N/A"
            else:
                rets = compute_returns(close_primary)
                vol = float(rets.std() * math.sqrt(252)) if len(rets) > 2 else float("nan")
                mdd = float(max_drawdown(close_primary))
                sr = float(sharpe_ratio(rets, rf_annual)) if len(rets) > 2 else float("nan")

                if not np.isnan(vol):
                    if vol < 0.20:
                        risk_label = "LOW"
                    elif vol < 0.35:
                        risk_label = "MEDIUM"
                    else:
                        risk_label = "HIGH"
                else:
                    risk_label = "N/A"

            comp_df = compare_table_from_bulk(prices_df, tickers, rf_annual) if len(tickers) >= 2 else pd.DataFrame()
            fig = build_price_chart_from_bulk(prices_df, tickers)
            csv_data = build_csv_from_bulk(prices_df, tickers)

            analysis_text = f"""
Primary ticker: {primary}
Volatility (annualized): {vol}
Max Drawdown: {mdd}
Sharpe: {sr}
Risk Level: {risk_label}

Comparison:
{comp_df.to_string(index=False) if not comp_df.empty else 'N/A'}
"""

            advisor = llm_advisor_summary(query, analysis_text)

            st.session_state["chat_memory"].append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "query": query,
                "mode": "full_report",
                "response": advisor[:2000] if advisor else "Generated full report"
            })

            # Tabs (UI unchanged, News removed)
            t0, t1, t2, t3, t4 = st.tabs(
                ["Overview", "Advisor Summary", "Risk + Recommendation", "Compare", "Chart + CSV"]
            )

            with t0:
                st.subheader("Company Overview")

                # ✅ Lazy load (slow call only when you open this tab)
                with st.spinner("Loading company overview..."):
                    meta = get_stock_summary(primary)

                if meta:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Ticker", meta.get("ticker", "N/A"))
                    m2.metric("Company", meta.get("name", "N/A"))
                    m3.metric("Exchange", meta.get("exchange", "N/A"))
                    m4.metric("Currency", meta.get("currency", "N/A"))

                    m5, m6, m7, m8 = st.columns(4)
                    m5.metric("Market Cap", format_market_cap(meta.get("market_cap")))
                    pe_val = meta.get("pe")
                    m6.metric("P/E Ratio", f"{pe_val:.2f}" if isinstance(pe_val, (int, float)) else "N/A")

                    hi = meta.get("52w_high")
                    lo = meta.get("52w_low")
                    m7.metric("52W High", f"{hi:.2f}" if isinstance(hi, (int, float)) else "N/A")
                    m8.metric("52W Low", f"{lo:.2f}" if isinstance(lo, (int, float)) else "N/A")

                    st.caption(f"Sector: {meta.get('sector','N/A')} | Industry: {meta.get('industry','N/A')}")
                    if meta.get("website"):
                        st.caption(f"Website: {meta['website']}")
                else:
                    st.info("No overview available for this ticker.")

            with t1:
                st.subheader("Advisor Summary")
                if advisor:
                    st.markdown(advisor)
                else:
                    st.info("Groq API not configured. Set GROQ_API_KEY to enable AI advisor summary.")

            with t2:
                st.subheader("Risk Metrics (Primary ticker)")
                if close_primary.empty:
                    st.info("Not enough data for risk analysis.")
                else:
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Ticker", primary)
                    r2.metric("Volatility (ann.)", f"{vol:.2%}" if not np.isnan(vol) else "N/A")
                    r3.metric("Max Drawdown", f"{mdd:.2%}" if not np.isnan(mdd) else "N/A")
                    r4.metric("Sharpe", f"{sr:.2f}" if not np.isnan(sr) else "N/A")
                    st.caption(f"Risk Level: {risk_label}")

                st.subheader("Alerts")
                try:
                    if len(rets) >= 5:
                        last_return = float(rets.iloc[-1])
                        vol_alert = float(rets.std() * math.sqrt(252))

                        any_alert = False
                        if abs(last_return) > 0.05:
                            st.warning("High daily movement detected in the latest session.")
                            any_alert = True
                        if vol_alert > 0.45:
                            st.warning("Volatility is very high. This instrument may be risky in the short term.")
                            any_alert = True
                        if not any_alert:
                            st.success("No major alerts detected.")
                    else:
                        st.info("Not enough data to generate alerts.")
                except Exception:
                    st.info("Alerts unavailable for this ticker.")

            with t3:
                st.subheader("Comparison Report")
                if comp_df.empty:
                    st.info("Enter 2+ tickers to compare.")
                else:
                    st.dataframe(comp_df, use_container_width=True)
                    best = comp_df.iloc[0]["Ticker"]
                    worst = comp_df.iloc[-1]["Ticker"]
                    st.caption(f"Top risk-adjusted (Sharpe): {best} | Weakest: {worst}")

            with t4:
                st.subheader("Chart")
                st.plotly_chart(fig, use_container_width=True)

                st.write("")
                st.subheader("CSV Export")
                if csv_data:
                    st.download_button(
                        "Download report.csv",
                        data=csv_data,
                        file_name="report.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("CSV not available for these tickers.")

            st.session_state["run_full_report"] = False

    else:
        st.info("Enter tickers and click **Generate Full Advisor Report**, or use portfolio mode.")

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# RECENT ACTIVITY
# ============================================================
st.write("")
with st.expander("Recent activity", expanded=False):
    if len(st.session_state["chat_memory"]) == 0:
        st.info("No activity yet.")
    else:
        for chat in st.session_state["chat_memory"][::-1][:10]:
            st.markdown(f"**{chat['timestamp']}**")
            st.markdown(f"Action: `{chat['mode']}`")
            st.markdown(f"Tickers: `{chat['query']}`")
            st.markdown("---")

st.caption("FinSight • AI Financial Advisor • Streamlit + Yahoo Finance")

