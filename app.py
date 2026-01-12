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



BAD_WORDS = {
    "kill", "suicide", "rape", "terrorist", "bomb", "porn"
}

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
    raw = re.findall(r"\b[A-Za-z]{1,10}(?:\.[A-Za-z]{1,5})?\b", query)

    blacklist = {
        "and", "the", "with", "show", "what", "this", "that",
        "csv", "report", "for", "give", "export", "generate", "data",
        "analyze", "compare", "visualize", "plot", "trend"
    }

    tickers = []
    for r in raw:
        r_clean = r.upper()
        if r_clean.lower() in blacklist:
            continue
        
        if re.match(r"^[A-Z]{1,10}(\.[A-Z]{1,5})?$", r_clean):
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
        "csv_report": "csv_report",
        "comparison": "compare",
        "visualization": "visualize"
    }.get(state["mode"], END)
    return state


def analyze_stock_risk_trends(state: AgentState) -> AgentState:
    state["log_steps"].append("Visited: analyze_stock_risk_trends")

    tickers = extract_tickers(state["query"])
    if len(tickers) == 0:
        return {**state, "response": "❌ No valid stock tickers found in query."}

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

    summary = (
        f"📌 **Risk Analysis for {ticker}** (last {days} days)\n\n"
        f"- Annualized Volatility: **{vol:.2%}**\n"
        f"- Max Drawdown: **{mdd:.2%}**\n"
        f"- Sharpe Ratio: **{sr:.2f}**\n"
        f"- Risk Level: **{risk_label}**\n\n"
        f"Interpretation: Higher volatility and deeper drawdown = higher risk."
    )

    return {**state, "response": summary}


def compare_stocks(state: AgentState) -> AgentState:
    state["log_steps"].append("Visited: compare_stocks")

    tickers = extract_tickers(state["query"])
    if len(tickers) < 2:
        return {**state, "response": "❌ Please provide at least 2 tickers for comparison (example: 'Compare AAPL and TSLA')."}

    days = state["days"]
    rows = []

    for t in tickers[:5]:
        df = fetch_prices(t, days)
        rets = compute_returns(df)
        rows.append({
            "ticker": t,
            "volatility": float(rets.std() * math.sqrt(252)),
            "sharpe": float(sharpe_ratio(rets)),
            "max_drawdown": float(max_drawdown(df)),
            "last_close": float(df["close"].iloc[-1]),
        })

    comp = pd.DataFrame(rows).sort_values(by="sharpe", ascending=False)

    # create summary
    best = comp.iloc[0]["ticker"]
    worst = comp.iloc[-1]["ticker"]

    summary = "📊 **Stock Comparison**\n\n"
    summary += comp.to_markdown(index=False)
    summary += f"\n\n✅ Best risk-adjusted (Sharpe): **{best}**"
    summary += f"\n⚠️ Weakest risk-adjusted: **{worst}**"

    return {**state, "response": summary}


def generate_csv(state: AgentState) -> AgentState:
    state["log_steps"].append("Visited: generate_csv")

    tickers = extract_tickers(state["query"])
    if len(tickers) == 0:
        return {**state, "csv": "", "response": "❌ No tickers found."}

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
        return {**state, "csv": "", "response": "❌ No data returned from Yahoo Finance."}

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

    return {
        **state,
        "csv": csv_data,
        "response": f"✅ CSV report generated for: {', '.join(tickers[:5])}"
    }



def visualize_data(state: AgentState) -> AgentState:
    state["log_steps"].append("Visited: visualize_data")

    tickers = extract_tickers(state["query"])
    if len(tickers) == 0:
        return {**state, "response": "❌ No tickers found for visualization."}

    days = state["days"]

    fig = go.Figure()

    for t in tickers[:5]:
        df = fetch_prices(t, days)
        fig.add_trace(go.Scatter(x=df["date"], y=df["close"], mode="lines", name=t))

    fig.update_layout(
        title=f"Stock Price Trend (Last {days} Days)",
        xaxis_title="Date",
        yaxis_title="Close Price"
    )

    return {**state, "fig": fig, "response": "✅ Chart generated."}


def respond_llm(state: AgentState) -> AgentState:
    state["log_steps"].append("Visited: respond_llm")


    if not GROQ_AVAILABLE:
        fallback = (
            "**(LLM disabled: langchain_groq not installed)**\n\n"
            + state["response"]
            + "\n\n💡 Recommendation: Diversify, avoid high-risk concentration, and invest based on your time horizon."
        )
        return {**state, "response": fallback}

 
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    if GROQ_API_KEY.strip() == "":
        fallback = (
            "**(LLM disabled: GROQ_API_KEY not set)**\n\n"
            + state["response"]
            + "\n\n💡 Recommendation: Use risk metrics as guidance. Avoid over-leveraging."
        )
        return {**state, "response": fallback}

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

        is_safe, bad_word = is_clean(result)
        if not is_safe:
            return {**state, "response": f"⚠️ Blocked due to unsafe word: '{bad_word}'"}

        return {**state, "response": result}

    except Exception as e:
        return {**state, "response": f"⚠️ Advisory failed: {e}"}


builder = StateGraph(AgentState)
builder.set_entry_point("router")

builder.add_node("router", RunnableLambda(router))
builder.add_node("analyze_risk", RunnableLambda(analyze_stock_risk_trends))
builder.add_node("csv_report", RunnableLambda(generate_csv))
builder.add_node("compare", RunnableLambda(compare_stocks))
builder.add_node("visualize", RunnableLambda(visualize_data))
builder.add_node("respond_llm", RunnableLambda(respond_llm))

builder.add_conditional_edges("router", lambda s: s["next"], {
    "analyze_risk": "analyze_risk",
    "csv_report": "csv_report",
    "compare": "compare",
    "visualize": "visualize",
})

builder.add_edge("analyze_risk", "respond_llm")
builder.add_edge("compare", "respond_llm")
builder.add_edge("respond_llm", END)

builder.add_edge("csv_report", END)
builder.add_edge("visualize", END)

graph = builder.compile()


st.set_page_config(
    page_title="FinSight | AI Financial Advisor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at 10% 0%, #0b1220 0%, #050814 40%, #050814 100%);
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
        color: #E5E7EB;
    }

    /* Remove Streamlit default paddings */
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1300px;
    }

    /* Navbar */
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
        letter-spacing: 0.2px;
        color: #F9FAFB;
        margin: 0;
        line-height: 1;
    }
    .brand-subtitle {
        margin: 0;
        color: rgba(229,231,235,0.7);
        font-size: 12px;
        line-height: 1.2;
    }
    .nav-right {
        display: flex;
        gap: 10px;
        align-items: center;
        color: rgba(229,231,235,0.85);
        font-size: 13px;
        font-weight: 600;
    }
    .pill {
        padding: 8px 12px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.05);
    }

    /* Hero */
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
    .hero-tags {
        margin-top: 16px;
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    .tag {
        padding: 9px 14px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.05);
        font-size: 12px;
        font-weight: 700;
        color: rgba(229,231,235,0.9);
    }

    /* Cards grid */
    .grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 14px;
        margin-bottom: 18px;
    }
    .feature-card {
        border-radius: 22px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 16px 18px;
        box-shadow: 0 16px 45px rgba(0,0,0,0.25);
    }
    .feature-title {
        font-size: 14px;
        font-weight: 900;
        margin: 0 0 4px 0;
        color: #F9FAFB;
        letter-spacing: -0.2px;
    }
    .feature-desc {
        font-size: 13px;
        margin: 0;
        color: rgba(229,231,235,0.72);
        line-height: 1.5;
    }

    /* Input / Output panels */
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
        margin: 0 0 12px 0;
        color: #F9FAFB;
    }

    /* Inputs */
    div[data-baseweb="input"] > div {
        border-radius: 16px !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        background: rgba(0,0,0,0.25) !important;
        color: #E5E7EB !important;
    }
    div[data-baseweb="select"] > div {
        border-radius: 16px !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        background: rgba(0,0,0,0.25) !important;
        color: #E5E7EB !important;
    }

    /* Run button */
    div.stButton > button {
        width: 100%;
        border-radius: 16px;
        height: 52px;
        font-size: 15px;
        font-weight: 950;
        color: #0B1220;
        background: linear-gradient(135deg, #22C55E 0%, #3B82F6 100%);
        border: none;
        box-shadow: 0 18px 45px rgba(34,197,94,0.12);
        transition: 0.2s;
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
        opacity: 0.98;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 999px !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        background: rgba(255,255,255,0.05) !important;
        padding: 10px 14px !important;
        color: rgba(229,231,235,0.9) !important;
        font-weight: 800 !important;
    }

    /* Footer */
    .footer {
        margin-top: 20px;
        color: rgba(229,231,235,0.55);
        font-size: 12px;
        text-align: center;
    }

    @media (max-width: 1100px) {
        .grid { grid-template-columns: 1fr; }
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
            <p class="brand-subtitle">AI financial advisor</p>
        </div>
    </div>
    <div class="nav-right">
        <div class="pill">Data: Yahoo Finance</div>
        <div class="pill">Workflow: LangGraph</div>
        <div class="pill">UI: Streamlit</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>Smart, fast and clean stock insights.</h1>
    <p>
        FinSight is a production-ready AI agent interface for stock risk analysis, comparison, visualization,
        and CSV reporting. Powered by real historical market data with an optional LLM advisory layer.
    </p>
    <div class="hero-tags">
        <span class="tag">Risk analysis</span>
        <span class="tag">Stock comparison</span>
        <span class="tag">Trend visualization</span>
        <span class="tag">CSV reporting</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="grid">
  <div class="feature-card">
    <div class="feature-title">Risk Engine</div>
    <p class="feature-desc">Annualized volatility, Sharpe ratio and max drawdown to understand risk clearly.</p>
  </div>
  <div class="feature-card">
    <div class="feature-title">Comparison Suite</div>
    <p class="feature-desc">Compare up to 5 instruments with consistent metrics and clear ranking.</p>
  </div>
  <div class="feature-card">
    <div class="feature-title">Reporting</div>
    <p class="feature-desc">Export CSV datasets for your documentation, analysis or project submission.</p>
  </div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.05, 1.55], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Run analysis</div>', unsafe_allow_html=True)

    query = st.text_input(
        label="Query",
        value="Compare AAPL and TSLA",
        placeholder="Example: Analyze AAPL | Compare AAPL TSLA MSFT | Show TCS.NS INFY.NS"
    )

    mode = st.selectbox(
        "Mode",
        ["risk_analysis", "comparison", "visualization", "csv_report"],
        index=1
    )

    days = st.slider("History (days)", min_value=3, max_value=365, value=30)

    st.write("")
    run = st.button("Run")

    st.write("")
    st.markdown("**Examples**")
    st.code("Analyze AAPL", language="text")
    st.code("Compare AAPL TSLA MSFT", language="text")
    st.code("Show TCS.NS INFY.NS", language="text")
    st.code("CSV report for AAPL TSLA", language="text")

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Results</div>', unsafe_allow_html=True)

    if run:
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

                st.success("Completed successfully.")

                t1, t2, t3, t4 = st.tabs(["Summary", "Chart", "CSV", "Trace"])

                with t1:
                    st.markdown(result["response"])

                with t2:
                    if result.get("fig") is not None:
                        st.plotly_chart(result["fig"], use_container_width=True)
                    else:
                        st.info("No chart generated for this mode.")

                with t3:
                    if result.get("csv"):
                        st.download_button(
                            "Download report.csv",
                            data=result["csv"],
                            file_name="report.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No CSV generated for this mode.")

                with t4:
                    if result.get("log_steps"):
                        for step in result["log_steps"]:
                            st.markdown(f"- {step}")
                    else:
                        st.info("No trace available.")

            except Exception as e:
                st.error(f"Application error: {e}")

    else:
        st.info("Enter a query on the left and run analysis to view results.")

    st.markdown('</div>', unsafe_allow_html=True)

st.write("")
with st.expander("Recent activity", expanded=False):
    if len(st.session_state.chat_memory) == 0:
        st.info("No activity yet.")
    else:
        for chat in st.session_state.chat_memory[::-1][:10]:
            st.markdown(f"**{chat['timestamp']}**")
            st.markdown(f"Mode: `{chat['mode']}`")
            st.markdown(f"Query: `{chat['query']}`")
            st.markdown("---")

st.markdown('<div class="footer">FinSight • AI Financial Advisor • Streamlit + LangGraph</div>', unsafe_allow_html=True)
