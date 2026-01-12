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


st.set_page_config(page_title="📈 AI Financial Agent", layout="centered")
st.title("📈 AI Financial Advisor (Working LangGraph Version)")

if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = []

with st.sidebar.expander("🧠 Chat History", expanded=True):
    if len(st.session_state.chat_memory) == 0:
        st.info("No history yet.")
    for chat in st.session_state.chat_memory[::-1]:
        st.markdown(f"🕒 {chat['timestamp']}")
        st.markdown(f"**Query:** {chat['query']}")
        st.markdown(f"**Mode:** {chat['mode']}")
        st.markdown(f"**Response:** {chat['response'][:200]}...")
        st.markdown("---")

query = st.text_input("💬 Enter your query", "Analyze AAPL and TSLA")
mode = st.selectbox("🧭 Mode", ["risk_analysis", "comparison", "csv_report", "visualization"])
days = st.slider("📆 Days", min_value=3, max_value=365, value=30)

if st.button("🚀 Run Agent"):
    with st.spinner("Processing..."):
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

            st.markdown(result["response"])

            if result.get("fig") is not None:
                st.plotly_chart(result["fig"], use_container_width=True)

            if result.get("csv"):
                st.download_button("📥 Download CSV", result["csv"], "report.csv")

            if result.get("log_steps"):
                with st.expander("⚙️ Execution Trace"):
                    for step in result["log_steps"]:
                        st.markdown(f"- {step}")

        except Exception as e:
            st.error(f"App error: {e}")
