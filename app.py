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
    raw = re.findall(r"\b[A-Za-z]{1,12}(?:\.[A-Za-z]{1,5})?\b", query)

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
        if re.match(r"^[A-Z]{1,12}(\.[A-Z]{1,5})?$", r_clean):
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
        f"**Current Price:** {current_price:.2f} (as of {latest_date})\n\n"
        f"- Annualized Volatility: **{vol:.2%}**\n"
        f"- Max Drawdown: **{mdd:.2%}**\n"
        f"- Sharpe Ratio: **{sr:.2f}**\n"
        f"- Risk Level: **{risk_label}**\n\n"
        f"Interpretation: higher volatility and drawdown indicates higher risk."
    )

    return {**state, "response": summary}


def compare_stocks(state: AgentState) -> AgentState:
    state["log_steps"].append("Visited: compare_stocks")

    tickers = extract_tickers(state["query"])
    if len(tickers) < 2:
        return {**state, "response": "Please enter at least 2 tickers for comparison (example: AAPL TSLA)."}

    days = state["days"]
    rows = []

    for t in tickers[:5]:
        df = fetch_prices(t, days)
        rets = compute_returns(df)
        rows.append({
            "Ticker": t,
            "Current Price": float(df["close"].iloc[-1]),
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
        return {**state, "response": state["response"] + "\n\nNote: LLM advisory disabled (langchain_groq not installed)."}

    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    if GROQ_API_KEY.strip() == "":
        return {**state, "response": state["response"] + "\n\nNote: LLM advisory disabled (GROQ_API_KEY not set)."}

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

Give a clear, respectful summary and a direct recommendation in short bullet points.
Avoid strong financial guarantees.
"""

        result = chat.invoke([HumanMessage(content=prompt)]).content.strip()

        safe, bad_word = is_clean(result)
        if not safe:
            return {**state, "response": f"Blocked due to unsafe word: {bad_word}"}

        return {**state, "response": result}

    except Exception as e:
        return {**state, "response": f"LLM advisory failed: {e}"}


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
        Enter one or more tickers and run analysis instantly.
        No command keywords needed. Export reports, visualize trends and compare risk metrics in one interface.
    </p>
    <span class="tag">Risk analysis</span>
    <span class="tag">Comparison</span>
    <span class="tag">Visualization</span>
    <span class="tag">CSV export</span>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.05, 1.65], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Input</div>', unsafe_allow_html=True)

    query = st.text_input(
        "Tickers",
        value="AAPL TSLA",
        placeholder="Example: AAPL | AAPL TSLA MSFT | TCS.NS INFY.NS"
    )

    days = st.slider("History window (days)", min_value=3, max_value=365, value=30)

    tickers_preview = extract_tickers(query)
    if tickers_preview:
        st.caption("Detected tickers: " + ", ".join(tickers_preview[:10]))
    else:
        st.caption("Detected tickers: none")

    st.write("")

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

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Results</div>', unsafe_allow_html=True)

    if mode:
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

                st.session_state.chat_memory.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "query": query,
                    "mode": mode,
                    "response": result["response"]
                })

                t1, t2, t3, t4 = st.tabs(["Summary", "Chart", "CSV", "Trace"])

                with t1:
                    st.markdown(result["response"])

                with t2:
                    if result.get("fig") is not None:
                        st.plotly_chart(result["fig"], use_container_width=True)
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
                    if result.get("log_steps"):
                        for step in result["log_steps"]:
                            st.markdown(f"- {step}")
                    else:
                        st.info("No trace available.")

            except Exception as e:
                st.error(f"Application error: {e}")
    else:
        st.info("Enter tickers on the left and select an action.")

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
