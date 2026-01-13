import os
import re
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from datetime import datetime
from typing import TypedDict, List, Any, Dict

# ========= Optional dependencies =========
try:
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False

try:
    import yfinance as yf
    YF_AVAILABLE = True
except Exception:
    YF_AVAILABLE = False

# ========= GLOBALS =========
BAD_WORDS = {"kill", "suicide", "rape", "terrorist", "bomb", "porn"}


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

# ========= Streamlit page =========
st.set_page_config(
    page_title="FinSight | AI Financial Advisor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========= session state =========
if "chat_memory" not in st.session_state:
    st.session_state["chat_memory"] = []


# ============================================================
# ===================== UTILITY FUNCTIONS =====================
# ============================================================

def is_clean(text: str):
    lowered = (text or "").lower()
    for w in BAD_WORDS:
        if w in lowered:
            return False, w
    return True, ""


def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def extract_tickers(query: str):
    """
    Extract tickers like: AAPL, TSLA, MSFT, TCS.NS
    Filters common stop words.
    """
    if not query:
        return []

    raw = re.findall(r"\b[A-Za-z]{1,12}(?:\.[A-Za-z]{1,6})?\b", query)

    blacklist = {
        "and", "the", "with", "show", "what", "this", "that",
        "csv", "report", "for", "give", "export", "generate", "data",
        "analyze", "compare", "visualize", "plot", "trend", "me", "please",
        "risk", "analysis", "stock", "market", "price", "portfolio", "advisor",
        "news", "chart", "graph", "buy", "sell"
    }

    tickers = []
    for r in raw:
        t = r.strip().upper()

        if t.lower() in blacklist:
            continue

        if len(t) == 1:
            continue

        if not re.match(r"^[A-Z]{1,12}(\.[A-Z]{1,6})?$", t):
            continue

        tickers.append(t)

    # remove duplicates while keeping order
    return list(dict.fromkeys(tickers))


def fetch_prices(ticker: str, days: int) -> pd.DataFrame:
    if not YF_AVAILABLE:
        raise RuntimeError("yfinance not installed. Run: pip install yfinance")

    df = yf.download(ticker, period=f"{days}d", interval="1d", progress=False)

    if df is None or df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    if "Close" not in df.columns:
        raise ValueError(f"Close column missing for {ticker}")

    df.rename(columns={"Date": "date", "Close": "close"}, inplace=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])

    return df[["date", "close"]]


def compute_returns(df: pd.DataFrame) -> pd.Series:
    r = df["close"].pct_change()
    return r.dropna()


def sharpe_ratio(returns: pd.Series, risk_free_rate=0.0) -> float:
    """
    risk_free_rate: annual (example 0.04 = 4%)
    """
    if returns is None or len(returns) < 2:
        return float("nan")

    rf_daily = risk_free_rate / 252
    excess = returns - rf_daily

    sd = excess.std()
    if sd == 0 or np.isnan(sd):
        return float("nan")

    return float((excess.mean() / sd) * math.sqrt(252))


def max_drawdown(df: pd.DataFrame) -> float:
    prices = df["close"].values
    peak = None
    mdd = 0.0
    for p in prices:
        if peak is None or p > peak:
            peak = p
        if peak and peak != 0:
            dd = (peak - p) / peak
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


def get_stock_summary(ticker: str) -> dict:
    if not YF_AVAILABLE:
        return {}
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
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


def get_stock_news(ticker: str, limit: int = 10) -> list:
    if not YF_AVAILABLE:
        return []
    try:
        tk = yf.Ticker(ticker)
        if hasattr(tk, "get_news"):
            news = tk.get_news() or []
        else:
            news = tk.news or []
        return news[:limit]
    except Exception:
        return []


def simple_sentiment_score(text: str) -> float:
    if not text:
        return 0.0
    positive = {"up", "gain", "bull", "surge", "jump", "growth", "beat", "record", "strong", "profit"}
    negative = {"down", "loss", "bear", "drop", "fall", "decline", "miss", "weak", "crash", "lawsuit"}
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if not words:
        return 0.0
    pos = sum(1 for w in words if w in positive)
    neg = sum(1 for w in words if w in negative)
    return (pos - neg) / max(1, pos + neg)


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

        info = tk.info or {}
        volume = info.get("volume", None)
        currency = info.get("currency", "")

        return {
            "ticker": ticker,
            "price": last_close,
            "prev_close": prev_close,
            "change": change,
            "change_pct": change_pct,
            "volume": volume,
            "currency": currency,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception:
        return {}


def compare_stocks_table(tickers: list, days: int, risk_free_rate: float) -> pd.DataFrame:
    rows = []
    for t in tickers[:8]:
        df = fetch_prices(t, days)
        rets = compute_returns(df)
        rows.append({
            "Ticker": t,
            "Last Close": float(df["close"].iloc[-1]),
            "Volatility": float(rets.std() * math.sqrt(252)),
            "Sharpe": float(sharpe_ratio(rets, risk_free_rate=risk_free_rate)),
            "Max Drawdown": float(max_drawdown(df)),
        })
    comp = pd.DataFrame(rows)
    comp = comp.sort_values(by="Sharpe", ascending=False)
    return comp


def generate_csv_data(tickers: list, days: int) -> str:
    if not YF_AVAILABLE:
        return ""

    df = yf.download(
        tickers=" ".join(tickers[:8]),
        period=f"{days}d",
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False
    )

    if df is None or df.empty:
        return ""

    records = []
    if not isinstance(df.columns, pd.MultiIndex):
        df = df.reset_index()
        for _, row in df.iterrows():
            records.append({"date": row["Date"], "ticker": tickers[0], "close": row["Close"]})
    else:
        for t in tickers[:8]:
            if t not in df.columns.get_level_values(0):
                continue
            temp = df[t].reset_index()
            for _, row in temp.iterrows():
                records.append({"date": row["Date"], "ticker": t, "close": row["Close"]})

    out = pd.DataFrame(records)
    return out.to_csv(index=False)


def recommendation_from_metrics(vol: float, mdd: float, sharpe: float):
    if vol < 0.20 and mdd < 0.20:
        profile = "Conservative"
        action = "Relatively stable behavior observed in selected period."
        controls = [
            "Use SIP/DCA approach",
            "Avoid leverage",
            "Diversify across 3–6 assets"
        ]
    elif vol < 0.35 and mdd < 0.35:
        profile = "Moderate"
        action = "Balanced risk profile, suitable for medium volatility tolerance."
        controls = [
            "Scale positions gradually",
            "Maintain diversification",
            "Consider defined stop-loss levels if trading"
        ]
    else:
        profile = "Aggressive"
        action = "High variability; suitable only with strong risk tolerance."
        controls = [
            "Reduce position sizing",
            "Avoid emotional averaging down",
            "Limit concentration in a single asset"
        ]

    if sharpe < 0.2:
        note = "Sharpe is weak → risk taken is not being rewarded well."
    elif sharpe < 1:
        note = "Sharpe is decent → acceptable risk-adjusted performance."
    else:
        note = "Sharpe is strong → good risk-adjusted performance."

    return profile, action, controls, note


def call_groq_summary(query: str, computed_report: str) -> str:
    """
    LLM layer: makes output readable and professional.
    """
    if not GROQ_AVAILABLE:
        return ""

    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        return ""

    try:
        chat = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0.25,
            groq_api_key=key
        )

        prompt = f"""
You are a professional financial research analyst.
The user entered tickers / query.

User Query:
{query}

Computed Report:
{computed_report}

Rewrite as a clean fintech advisory note with this exact structure:

### Executive Summary
- ...

### Key Observations
- ...

### Risks & Red Flags
- ...

### Action Plan (Educational)
- ...

End with:
Disclaimer: Educational purposes only. Not financial advice.

DO NOT promise returns. Avoid guarantees.
"""

        result = chat.invoke([HumanMessage(content=prompt)]).content.strip()

        ok, bad = is_clean(result)
        if not ok:
            return "⚠️ AI output blocked due to unsafe word."

        return result
    except Exception:
        return ""


def parse_portfolio_input(text: str):
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
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

    tickers = [t for t, _ in items]
    weights = np.array([w for _, w in items], dtype=float)

    if weights.sum() <= 0:
        return [], np.array([])

    weights = weights / weights.sum()
    return tickers, weights


def portfolio_metrics(tickers: list, weights: np.ndarray, days: int, risk_free_rate: float):
    if len(tickers) == 0:
        return {}

    price_df = pd.DataFrame()
    for t in tickers[:10]:
        df = fetch_prices(t, days)
        df = df.rename(columns={"close": t}).set_index("date")
        price_df = df[[t]] if price_df.empty else price_df.join(df[[t]], how="outer")

    price_df = price_df.dropna()
    if price_df.empty or len(price_df) < 10:
        return {}

    returns = price_df.pct_change().dropna()
    w = np.array(weights, dtype=float)

    port_returns = returns.dot(w)

    vol = float(port_returns.std() * math.sqrt(252))
    sr = float(sharpe_ratio(port_returns, risk_free_rate=risk_free_rate))
    total_return = float((1 + port_returns).prod() - 1)

    contrib = returns.mean() * w
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
# =========================== UI CSS =========================
# ============================================================

st.markdown("""
<style>
/* ===== Background ===== */
.stApp {
    background: radial-gradient(circle at 15% 10%, #101b3a 0%, #050814 40%, #02040c 100%) !important;
    font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
    color: #ffffff !important;
}

/* remove grey padding */
.block-container {
    max-width: 1320px;
    padding-top: 1.2rem;
    padding-bottom: 2.5rem;
}

/* ===== Hero / Panel Styling ===== */
.hero {
    border-radius: 26px;
    padding: 28px 28px;
    background: linear-gradient(180deg, rgba(18, 29, 64, 0.75) 0%, rgba(5, 8, 20, 0.55) 100%) !important;
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 24px 80px rgba(0,0,0,0.60);
    margin-bottom: 18px;
}

.panel {
    border-radius: 26px;
    padding: 18px 18px;
    background: rgba(10, 15, 35, 0.78) !important;
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 24px 80px rgba(0,0,0,0.55);
}

/* ===== Improve all text contrast ===== */
.stMarkdown, .stMarkdown * {
    color: rgba(255,255,255,0.96) !important;
}

/* ===== Tabs (remove grey look) ===== */
button[data-baseweb="tab"] {
    color: rgba(255,255,255,0.70) !important;
    font-weight: 900 !important;
    font-size: 14px !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #ffffff !important;
}

div[data-baseweb="tab-highlight"] {
    background-color: #3B82F6 !important;
    height: 3px !important;
    border-radius: 999px !important;
}

/* ===== Inputs / dropdowns (remove grey background) ===== */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
textarea {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 16px !important;
}

input, textarea {
    color: #ffffff !important;
    font-weight: 900 !important;
}

/* Placeholders */
::placeholder {
    color: rgba(255,255,255,0.50) !important;
    font-weight: 700 !important;
}

/* ===== Metrics (this is the GREY issue) ===== */
div[data-testid="stMetricValue"] * {
    color: #ffffff !important;
    font-weight: 950 !important;
    letter-spacing: -0.3px !important;
}

div[data-testid="stMetricLabel"] * {
    color: rgba(255,255,255,0.72) !important;
    font-weight: 850 !important;
}

div[data-testid="stMetricDelta"] * {
    font-weight: 950 !important;
}

/* ===== Buttons ===== */
div.stButton > button {
    border-radius: 16px !important;
    height: 52px !important;
    font-size: 15px !important;
    font-weight: 950 !important;
    color: #ffffff !important;
    background: linear-gradient(135deg, #2563EB 0%, #22C55E 100%) !important;
    border: none !important;
    transition: 0.2s;
    width: 100%;
    box-shadow: 0 18px 60px rgba(0,0,0,0.45);
}

div.stButton > button:hover {
    transform: translateY(-1px);
    filter: brightness(1.06);
}

/* ===== Sliders accent ===== */
div[data-testid="stSlider"] [role="slider"] {
    background: #3B82F6 !important;
}

/* ===== Links ===== */
a {
    color: #60A5FA !important;
    font-weight: 900 !important;
    text-decoration: none !important;
}
a:hover { text-decoration: underline !important; }

</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="hero">
    <h1>FinSight — AI Financial Advisor</h1>
    <p>
        Enter tickers and generate a complete investor-ready report: live prices, company overview,
        risk metrics, comparisons, charts, news sentiment, alerts and CSV export.
    </p>
    <span class="tag">Full report</span>
    <span class="tag">Risk & return</span>
    <span class="tag">Comparison</span>
    <span class="tag">Charts</span>
    <span class="tag">News + sentiment</span>
</div>
""", unsafe_allow_html=True)


# ============================================================
# =========================== MAIN UI =========================
# ============================================================

left, right = st.columns([1.05, 1.65], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Input</div>', unsafe_allow_html=True)

    selected_company = st.selectbox("Quick search (optional)", ["None"] + list(SMART_TICKERS.keys()))
    default_q = SMART_TICKERS[selected_company] if selected_company != "None" else "AAPL TSLA MSFT"

    query = st.text_input(
        "Tickers / Query",
        value=default_q,
        placeholder="Example: AAPL TSLA MSFT | TCS.NS INFY.NS"
    )

    days = st.slider("History window (days)", min_value=7, max_value=365, value=90)

    risk_free_rate = st.slider(
        "Risk-free rate (annual, %)",
        min_value=0.0,
        max_value=12.0,
        value=4.0,
        step=0.25
    ) / 100.0

    tickers = extract_tickers(query)

    if tickers:
        st.caption("Detected tickers: " + ", ".join(tickers[:10]))
    else:
        st.caption("Detected tickers: none")

    # Live prices preview
    if tickers:
        st.write("")
        st.markdown("#### Live Prices (Top 4)")
        cols = st.columns(min(4, len(tickers)))
        for i, t in enumerate(tickers[:4]):
            q = get_live_quote(t)
            if not q:
                cols[i].metric(label=t, value="N/A")
            else:
                cols[i].metric(
                    label=t,
                    value=f"{q['price']:.2f} {q.get('currency','')}",
                    delta=f"{q['change']:.2f} ({q['change_pct']:.2f}%)"
                )

    st.write("")
    st.markdown("#### Generate")

    run_full = st.button("✅ Generate Full Advisor Report")

    st.write("")
    st.markdown("#### Portfolio Mode (optional)")

    portfolio_text = st.text_area(
        "Portfolio input (ticker weight)",
        value="AAPL 40\nTSLA 30\nMSFT 30",
        height=120
    )

    run_portfolio = st.button("Run portfolio analysis")

    st.markdown('</div>', unsafe_allow_html=True)


with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Results</div>', unsafe_allow_html=True)

    # ================= Portfolio Mode =================
    if run_portfolio:
        ptickers, pweights = parse_portfolio_input(portfolio_text)

        if len(ptickers) < 2:
            st.error("Enter at least 2 tickers in portfolio mode.")
        else:
            with st.spinner("Analyzing portfolio..."):
                pm = portfolio_metrics(ptickers, pweights, days=max(days, 90), risk_free_rate=risk_free_rate)

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
                figp.update_layout(title="Portfolio Growth Curve", xaxis_title="Date", yaxis_title="Growth")
                st.plotly_chart(figp, use_container_width=True)

    # ================= Full Report Mode =================
    elif run_full:
        if not tickers:
            st.error("Please enter at least one valid ticker (example: AAPL).")
        else:
            primary = tickers[0]
            with st.spinner("Generating full advisor report..."):
                # Overview
                meta = get_stock_summary(primary)
                news_items = get_stock_news(primary, limit=10)

                # Compute risk metrics for primary
                df_primary = fetch_prices(primary, days)
                rets_primary = compute_returns(df_primary)

                vol = float(rets_primary.std() * math.sqrt(252))
                mdd = float(max_drawdown(df_primary))
                sr = float(sharpe_ratio(rets_primary, risk_free_rate=risk_free_rate))

                # Recommendation
                profile, action, controls, note = recommendation_from_metrics(vol, mdd, sr)

                # Compare
                comp_df = None
                if len(tickers) >= 2:
                    comp_df = compare_stocks_table(tickers, days, risk_free_rate=risk_free_rate)

                # Chart
                fig = go.Figure()
                for t in tickers[:8]:
                    d = fetch_prices(t, days)
                    fig.add_trace(go.Scatter(x=d["date"], y=d["close"], mode="lines", name=t))
                fig.update_layout(title=f"Price Trend ({days} days)", xaxis_title="Date", yaxis_title="Close Price")

                # CSV
                csv_data = generate_csv_data(tickers, days)

                # Computed report (used for AI rewrite)
                computed_report = f"""
Instrument: {primary}
Volatility: {vol:.2%}
Max Drawdown: {mdd:.2%}
Sharpe Ratio: {sr:.2f}
Investor Profile Fit: {profile}
""" + (("\nComparison Table:\n" + comp_df.to_string(index=False)) if comp_df is not None else "")

                ai_summary = call_groq_summary(query, computed_report)

            # Save activity
            st.session_state["chat_memory"].append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "query": query,
                "mode": "full_report",
                "response": ai_summary if ai_summary else computed_report
            })

            # Tabs
            t0, t1, t2, t3, t4, t5 = st.tabs(
                ["Overview", "Advisor Summary", "Risk + Recommendation", "Compare", "Chart", "CSV + News"]
            )

            with t0:
                if meta:
                    st.subheader("Company Overview")
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
                    st.info("No company overview available.")

            with t1:
                if ai_summary:
                    st.markdown(ai_summary)
                else:
                    st.info("AI summary unavailable (Groq key missing). Showing computed report.")
                    st.code(computed_report)

            with t2:
                st.subheader("Risk Metrics")
                r1, r2, r3 = st.columns(3)
                r1.metric("Volatility (annualized)", f"{vol:.2%}")
                r2.metric("Max Drawdown", f"{mdd:.2%}")
                r3.metric("Sharpe Ratio", f"{sr:.2f}")

                st.subheader("Recommendation")
                st.success(f"Investor Profile Fit: **{profile}**")
                st.markdown(f"**Interpretation:** {action}")
                st.caption(note)

                st.markdown("**Suggested Risk Controls:**")
                for c in controls:
                    st.markdown(f"- {c}")

                st.caption("⚠️ Disclaimer: Educational purposes only. Not financial advice.")

            with t3:
                if comp_df is None:
                    st.info("Enter at least 2 tickers to see comparison.")
                else:
                    st.subheader("Comparison Table")
                    st.dataframe(comp_df, use_container_width=True)

            with t4:
                st.plotly_chart(fig, use_container_width=True)

            with t5:
                st.subheader("CSV Export")
                if csv_data:
                    st.download_button("Download report.csv", data=csv_data, file_name="report.csv", mime="text/csv")
                else:
                    st.info("CSV unavailable (no data returned).")

                st.divider()

                st.subheader("Latest News + Sentiment")
                if not news_items:
                    st.info("No news returned for this ticker.")
                else:
                    for item in news_items:
                        title = item.get("title", "Untitled")
                        link = item.get("link", "")
                        publisher = item.get("publisher", "Unknown")
                        tstamp = item.get("providerPublishTime")
                        score = simple_sentiment_score(title)

                        if score > 0.2:
                            label = "Positive"
                        elif score < -0.2:
                            label = "Negative"
                        else:
                            label = "Neutral"

                        date_text = ""
                        try:
                            date_text = datetime.fromtimestamp(tstamp).strftime("%Y-%m-%d %H:%M")
                        except Exception:
                            pass

                        st.markdown(f"**{title}**")
                        st.caption(f"{publisher} {('| ' + date_text) if date_text else ''} | Sentiment: {label}")
                        if link:
                            st.markdown(link)
                        st.divider()

    else:
        st.info("Enter tickers and click **Generate Full Advisor Report** (or use Portfolio Mode).")

    st.markdown('</div>', unsafe_allow_html=True)


# Recent activity
st.write("")
with st.expander("Recent activity", expanded=False):
    if len(st.session_state["chat_memory"]) == 0:
        st.info("No activity yet.")
    else:
        for chat in st.session_state["chat_memory"][::-1][:10]:
            st.markdown(f"**{chat['timestamp']}**")
            st.markdown(f"Mode: `{chat['mode']}`")
            st.markdown(f"Query: `{chat['query']}`")
            st.markdown("---")

st.caption("FinSight • AI Financial Advisor • Streamlit + yfinance + Groq (optional)")
