import os
import re
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from datetime import datetime
from typing import TypedDict, List, Any

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage

try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False

try:
    import yfinance as yf
    YF_AVAILABLE = True
except Exception:
    YF_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except Exception:
    NLTK_AVAILABLE = False


class AgentState(TypedDict):
    query: str
    mode: str
    days: int
    response: str
    csv: str
    fig: Any
    next: str
    log_steps: List[str]


BAD_WORDS = {"kill", "suicide", "rape", "terrorist", "bomb", "porn"}


def is_clean(text: str):
    if not NLTK_AVAILABLE:
        lowered = text.lower()
        for w in BAD_WORDS:
            if w in lowered:
                return False, w
        return True, ""

    try:
        tokens = word_tokenize(text.lower())
    except Exception:
        tokens = re.findall(r"[a-zA-Z']+", text.lower())

    for t in tokens:
        if t in BAD_WORDS:
            return False, t
    return True, ""


def extract_tickers(query: str):
    raw = re.findall(r"\b[A-Za-z]{1,12}(?:\.[A-Za-z]{1,6})?\b", query)

    blacklist = {
        "and", "the", "with", "show", "what", "this", "that",
        "csv", "report", "for", "give", "export", "generate", "data",
        "analyze", "compare", "visualize", "plot", "trend", "me", "please"
    }

    tickers = []
    for r in raw:
        r_clean = r.upper()
        if r_clean.lower() in blacklist:
            continue
        if re.match(r"^[A-Z]{1,12}(\.[A-Z]{1,6})?$", r_clean):
            tickers.append(r_clean)

    return list(dict.fromkeys(tickers))


def fetch_prices(ticker: str, days: int) -> pd.DataFrame:
    if not YF_AVAILABLE:
        raise RuntimeError("yfinance not installed. Install with: pip install yfinance")

    df = yf.download(ticker, period=f"{days}d", interval="1d", progress=False)

    if df is None or df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    if "Close" not in df.columns:
        raise ValueError(f"'Close' column missing for {ticker}. Columns: {df.columns}")

    df.rename(columns={"Date": "date", "Close": "close"}, inplace=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])

    return df[["date", "close"]]


def compute_returns(df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    df["returns"] = df["close"].pct_change()
    return df["returns"].dropna()


def sharpe_ratio(returns: pd.Series, risk_free_rate=0.0) -> float:
    if len(returns) < 2:
        return float("nan")
    excess = returns - risk_free_rate / 252
    if excess.std() == 0:
        return float("nan")
    return (excess.mean() / excess.std()) * math.sqrt(252)


def max_drawdown(df: pd.DataFrame) -> float:
    prices = df["close"].values
    peak = -np.inf
    mdd = 0
    for p in prices:
        peak = max(peak, p)
        dd = (peak - p) / peak
        mdd = max(mdd, dd)
    return mdd


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

    tickers = [t for t, _ in items]
    weights = np.array([w for _, w in items], dtype=float)

    if weights.sum() <= 0:
        return [], np.array([])

    weights = weights / weights.sum()
    return tickers, weights


def portfolio_metrics(tickers: list, weights: np.ndarray, days: int = 90):
    if len(tickers) == 0:
        return {}

    price_df = pd.DataFrame()

    for t in tickers:
        df = fetch_prices(t, days)
        df = df.rename(columns={"close": t})
        df = df.set_index("date")
        price_df = df[[t]] if price_df.empty else price_df.join(df[[t]], how="outer")

    price_df = price_df.dropna()
    if price_df.empty or len(price_df) < 7:
        return {}

    returns = price_df.pct_change().dropna()
    w = np.array(weights, dtype=float)

    port_returns = returns.dot(w)

    vol = float(port_returns.std() * math.sqrt(252))
    sr = float(sharpe_ratio(port_returns))
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


def router(state: AgentState) -> AgentState:
    state["log_steps"].append("Visited: router")
    state["next"] = {
        "risk_analysis": "analyze_risk",
        "comparison": "compare",
        "visualization": "visualize",
        "csv_report": "csv_report",
    }.get(state["mode"], END)
    return state


def analyze_stock_risk_trends(state: AgentState) -> AgentState:
    state["log_steps"].append("Visited: analyze_stock_risk_trends")

    tickers = extract_tickers(state["query"])
    if len(tickers) == 0:
        return {**state, "response": "No valid tickers detected."}

    ticker = tickers[0]
    days = state["days"]

    df = fetch_prices(ticker, days)
    rets = compute_returns(df)

    vol = float(rets.std() * math.sqrt(252))
    mdd = float(max_drawdown(df))
    sr = float(sharpe_ratio(rets))

    if vol < 0.2:
        risk_label = "LOW"
    elif vol < 0.35:
        risk_label = "MEDIUM"
    else:
        risk_label = "HIGH"

    current_price = float(df["close"].iloc[-1])
    latest_date = df["date"].iloc[-1].date()

    summary = (
        f"## Risk Analysis\n\n"
        f"**Ticker:** {ticker}\n\n"
        f"**Last Close:** {current_price:.2f} (as of {latest_date})\n\n"
        f"- Annualized Volatility: **{vol:.2%}**\n"
        f"- Max Drawdown: **{mdd:.2%}**\n"
        f"- Sharpe Ratio: **{sr:.2f}**\n"
        f"- Risk Level: **{risk_label}**\n\n"
        f"Interpretation: Higher volatility and deeper drawdown indicates higher risk."
    )

    return {**state, "response": summary}


def compare_stocks(state: AgentState) -> AgentState:
    state["log_steps"].append("Visited: compare_stocks")

    tickers = extract_tickers(state["query"])
    if len(tickers) < 2:
        return {**state, "response": "Enter at least 2 tickers for comparison. Example: AAPL TSLA"}

    days = state["days"]
    rows = []

    for t in tickers[:5]:
        df = fetch_prices(t, days)
        rets = compute_returns(df)
        rows.append({
            "Ticker": t,
            "Last Close": float(df["close"].iloc[-1]),
            "Volatility": float(rets.std() * math.sqrt(252)),
            "Sharpe": float(sharpe_ratio(rets)),
            "Max Drawdown": float(max_drawdown(df)),
        })

    comp = pd.DataFrame(rows).sort_values(by="Sharpe", ascending=False)

    best = comp.iloc[0]["Ticker"]
    worst = comp.iloc[-1]["Ticker"]

    summary = "## Comparison Report\n\n"
    summary += comp.to_markdown(index=False)
    summary += f"\n\n**Top Risk-Adjusted (Sharpe):** {best}"
    summary += f"\n\n**Weakest Risk-Adjusted:** {worst}"

    return {**state, "response": summary}


def generate_csv(state: AgentState) -> AgentState:
    state["log_steps"].append("Visited: generate_csv")

    tickers = extract_tickers(state["query"])
    if len(tickers) == 0:
        return {**state, "csv": "", "response": "No valid tickers detected."}

    days = state["days"]

    df = yf.download(
        tickers=" ".join(tickers[:5]),
        period=f"{days}d",
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False
    )

    if df is None or df.empty:
        return {**state, "csv": "", "response": "No data returned from Yahoo Finance."}

    records = []

    if not isinstance(df.columns, pd.MultiIndex):
        df = df.reset_index()
        for _, row in df.iterrows():
            records.append({
                "date": row["Date"],
                "ticker": tickers[0],
                "close": row["Close"]
            })
    else:
        for t in tickers[:5]:
            if t not in df.columns.get_level_values(0):
                continue
            temp = df[t].reset_index()
            for _, row in temp.iterrows():
                records.append({
                    "date": row["Date"],
                    "ticker": t,
                    "close": row["Close"]
                })

    out = pd.DataFrame(records)
    csv_data = out.to_csv(index=False)

    return {**state, "csv": csv_data, "response": f"CSV generated for: {', '.join(tickers[:5])}"}


def visualize_data(state: AgentState) -> AgentState:
    state["log_steps"].append("Visited: visualize_data")

    tickers = extract_tickers(state["query"])
    if len(tickers) == 0:
        return {**state, "response": "No valid tickers detected for visualization."}

    days = state["days"]
    fig = go.Figure()

    for t in tickers[:5]:
        df = fetch_prices(t, days)
        fig.add_trace(go.Scatter(x=df["date"], y=df["close"], mode="lines", name=t))

    fig.update_layout(
        title=f"Price Trend (Last {days} Days)",
        xaxis_title="Date",
        yaxis_title="Close Price"
    )

    return {**state, "fig": fig, "response": "Chart generated successfully."}


def respond_llm(state: AgentState) -> AgentState:
    state["log_steps"].append("Visited: respond_llm")

    if not GROQ_AVAILABLE:
        return {**state, "response": state["response"]}

    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    if GROQ_API_KEY.strip() == "":
        return {**state, "response": state["response"]}

    try:
        chat = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0.3,
            groq_api_key=GROQ_API_KEY
        )

        prompt = f"""
You are a professional financial advisor.
User Query: {state['query']}
Mode: {state['mode']}
Data Summary: {state['response']}

Give a clear summary and recommendation in short bullet points.
Avoid strong financial guarantees.
"""

        result = chat.invoke([HumanMessage(content=prompt)]).content.strip()

        safe, bad_word = is_clean(result)
        if not safe:
            return {**state, "response": f"Blocked due to unsafe word: {bad_word}"}

        return {**state, "response": result}

    except Exception:
        return {**state, "response": state["response"]}


builder = StateGraph(AgentState)
builder.set_entry_point("router")

builder.add_node("router", RunnableLambda(router))
builder.add_node("analyze_risk", RunnableLambda(analyze_stock_risk_trends))
builder.add_node("compare", RunnableLambda(compare_stocks))
builder.add_node("visualize", RunnableLambda(visualize_data))
builder.add_node("csv_report", RunnableLambda(generate_csv))
builder.add_node("respond_llm", RunnableLambda(respond_llm))

builder.add_conditional_edges("router", lambda s: s["next"], {
    "analyze_risk": "analyze_risk",
    "compare": "compare",
    "visualize": "visualize",
    "csv_report": "csv_report",
})

builder.add_edge("analyze_risk", "respond_llm")
builder.add_edge("compare", "respond_llm")
builder.add_edge("respond_llm", END)
builder.add_edge("visualize", END)
builder.add_edge("csv_report", END)

graph = builder.compile()


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


st.set_page_config(
    page_title="FinSight | AI Financial Advisor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at 10% 0%, #0b1220 0%, #050814 45%, #050814 100%);
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
        color: #E5E7EB;
    }
    .block-container {
        max-width: 1300px;
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
    }
    .nav {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 14px 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 15px 40px rgba(0,0,0,0.25);
        backdrop-filter: blur(10px);
        margin-bottom: 18px;
    }
    .brand {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .brand-badge {
        width: 34px;
        height: 34px;
        border-radius: 12px;
        background: linear-gradient(135deg, #22C55E, #3B82F6);
        box-shadow: 0 0 0 6px rgba(34,197,94,0.10);
    }
    .brand-title {
        font-size: 18px;
        font-weight: 900;
        color: #F9FAFB;
        margin: 0;
        line-height: 1;
    }
    .brand-subtitle {
        margin: 0;
        color: rgba(229,231,235,0.65);
        font-size: 12px;
        line-height: 1.2;
    }
    .pill {
        padding: 8px 12px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.05);
        font-size: 12px;
        font-weight: 800;
        color: rgba(229,231,235,0.9);
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
        color: #F9FAFB;
        letter-spacing: -0.8px;
    }
    .hero p {
        margin: 0;
        font-size: 15px;
        color: rgba(229,231,235,0.78);
        line-height: 1.6;
        max-width: 850px;
    }
    .tag {
        padding: 9px 14px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.05);
        font-size: 12px;
        font-weight: 800;
        color: rgba(229,231,235,0.9);
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
        color: #F9FAFB;
        margin: 0 0 12px 0;
    }
    div.stButton > button {
        border-radius: 16px;
        height: 52px;
        font-size: 15px;
        font-weight: 950;
        color: #0B1220;
        background: linear-gradient(135deg, #22C55E 0%, #3B82F6 100%);
        border: none;
        transition: 0.2s;
        width: 100%;
    }

    div[data-testid="stMetricLabel"] * {
        color: rgba(255, 255, 255, 0.80) !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetricValue"] * {
        color: #FFFFFF !important;
        font-weight: 900 !important;
    }
    div[data-testid="stMetricDelta"] * {
        font-weight: 800 !important;
    }
</style>
""", unsafe_allow_html=True)


if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = []


st.markdown("""
<div class="nav">
    <div class="brand">
        <div class="brand-badge"></div>
        <div>
            <p class="brand-title">FinSight</p>
            <p class="brand-subtitle">AI Financial Advisor</p>
        </div>
    </div>
    <div style="display:flex;gap:10px;flex-wrap:wrap;">
        <div class="pill">Yahoo Finance</div>
        <div class="pill">LangGraph</div>
        <div class="pill">Streamlit</div>
    </div>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<div class="hero">
    <h1>Stock insights that feel effortless.</h1>
    <p>
        Enter tickers, view live prices, run risk analysis, compare instruments, visualize history and export datasets.
        Portfolio mode helps evaluate combined risk and performance. Company overview and news improve decision-making.
    </p>
    <span class="tag">Live prices</span>
    <span class="tag">Portfolio mode</span>
    <span class="tag">Smart search</span>
    <span class="tag">Risk analysis</span>
    <span class="tag">News & alerts</span>
</div>
""", unsafe_allow_html=True)


left, right = st.columns([1.05, 1.65], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Input</div>', unsafe_allow_html=True)

    selected_company = st.selectbox(
        "Quick search (optional)",
        ["None"] + list(SMART_TICKERS.keys()),
        index=0
    )

    if selected_company != "None":
        default_q = SMART_TICKERS[selected_company]
    else:
        default_q = "AAPL TSLA"

    query = st.text_input(
        "Tickers",
        value=default_q,
        placeholder="Example: AAPL TSLA MSFT | TCS.NS INFY.NS"
    )

    days = st.slider("History window (days)", min_value=3, max_value=365, value=30)

    preview = extract_tickers(query)
    if preview:
        st.caption("Detected tickers: " + ", ".join(preview[:10]))
    else:
        st.caption("Detected tickers: none")

    if preview:
        st.write("")
        st.markdown("#### Live Prices")

        cols = st.columns(min(4, len(preview)))
        for i, t in enumerate(preview[:4]):
            q = get_live_quote(t)
            if not q:
                cols[i].metric(label=t, value="N/A")
            else:
                delta_str = f"{q['change']:.2f} ({q['change_pct']:.2f}%)"
                cols[i].metric(
                    label=t,
                    value=f"{q['price']:.2f} {q.get('currency','')}",
                    delta=delta_str
                )

    st.write("")
    st.markdown("#### Actions")

    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)

    run_risk = c1.button("Risk analysis")
    run_compare = c2.button("Compare")
    run_viz = c3.button("Visualize")
    run_csv = c4.button("Export CSV")

    mode = None
    if run_risk:
        mode = "risk_analysis"
    elif run_compare:
        mode = "comparison"
    elif run_viz:
        mode = "visualization"
    elif run_csv:
        mode = "csv_report"

    st.write("")
    st.markdown("#### Portfolio Mode")

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

    if run_portfolio:
        ptickers, pweights = parse_portfolio_input(portfolio_text)

        if len(ptickers) < 2:
            st.error("Enter at least 2 tickers in portfolio mode.")
        else:
            with st.spinner("Analyzing portfolio..."):
                pm = portfolio_metrics(ptickers, pweights, days=max(days, 90))

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
                st.plotly_chart(figp, width="stretch")

    elif mode:
        with st.spinner("Processing request..."):
            try:
                result = graph.invoke({
                    "query": query,
                    "mode": mode,
                    "days": days,
                    "response": "",
                    "csv": "",
                    "fig": None,
                    "next": "",
                    "log_steps": []
                })

                tickers_found = extract_tickers(query)
                primary = tickers_found[0] if tickers_found else None

                meta = get_stock_summary(primary) if primary else {}
                news_items = get_stock_news(primary, limit=10) if primary else []

                st.session_state.chat_memory.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "query": query,
                    "mode": mode,
                    "response": result["response"]
                })

                t0, t1, t2, t3, t4, t5 = st.tabs(["Overview", "Summary", "Chart", "CSV", "News", "Trace"])

                with t0:
                    if primary and meta:
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
                        st.info("No overview available. Enter a valid ticker like AAPL or TCS.NS.")

                with t1:
                    st.markdown(result["response"])

                with t2:
                    if result.get("fig") is not None:
                        st.plotly_chart(result["fig"], width="stretch")
                    else:
                        st.info("No chart generated for this action.")

                with t3:
                    if result.get("csv"):
                        st.download_button(
                            "Download report.csv",
                            data=result["csv"],
                            file_name="report.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No CSV generated for this action.")

                with t4:
                    if not primary:
                        st.info("Enter at least one ticker to view news.")
                    else:
                        st.subheader("Latest News")

                        if not news_items:
                            st.info("No news available for this ticker (Yahoo sometimes returns empty).")
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

                        st.subheader("Alerts")
                        try:
                            df_alert = fetch_prices(primary, max(days, 30))
                            rets_alert = compute_returns(df_alert)

                            if len(rets_alert) >= 5:
                                last_return = float(rets_alert.iloc[-1])
                                vol = float(rets_alert.std() * math.sqrt(252))

                                any_alert = False

                                if abs(last_return) > 0.05:
                                    st.warning("High daily movement detected in the latest session.")
                                    any_alert = True

                                if vol > 0.45:
                                    st.warning("Volatility is very high. This instrument may be risky in short-term.")
                                    any_alert = True

                                if not any_alert:
                                    st.success("No major alerts detected.")
                            else:
                                st.info("Not enough data to generate alerts.")
                        except Exception:
                            st.info("Alerts unavailable for this ticker.")

                with t5:
                    if result.get("log_steps"):
                        for step in result["log_steps"]:
                            st.markdown(f"- {step}")
                    else:
                        st.info("No trace available.")

            except Exception as e:
                st.error(f"Application error: {e}")
    else:
        st.info("Enter tickers and select an action, or use portfolio mode.")

    st.markdown('</div>', unsafe_allow_html=True)


st.write("")
with st.expander("Recent activity", expanded=False):
    if len(st.session_state.chat_memory) == 0:
        st.info("No activity yet.")
    else:
        for chat in st.session_state.chat_memory[::-1][:10]:
            st.markdown(f"**{chat['timestamp']}**")
            st.markdown(f"Action: `{chat['mode']}`")
            st.markdown(f"Tickers: `{chat['query']}`")
            st.markdown("---")

st.caption("FinSight • AI Financial Advisor • Streamlit + LangGraph")

