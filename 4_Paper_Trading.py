import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from utils.auth import signup_user, login_user, logout
from utils.trading_db import (
    ensure_wallet,
    get_balance,
    get_holdings,
    get_trades,
    buy_stock,
    sell_stock,
)


st.set_page_config(
    page_title="FinSight | Paper Trading",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown("""
<style>
/* =========================
   GLOBAL APP THEME
========================= */
.stApp {
    background: radial-gradient(circle at 10% 0%, #0b1220 0%, #050814 45%, #050814 100%);
    font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
    color: #ffffff !important;
}

.block-container {
    max-width: 1500px;
    padding-top: 0.8rem;
    padding-bottom: 2.5rem;
}

/* Headings */
h1, h2, h3, h4, h5 {
    color: #ffffff !important;
    font-weight: 950 !important;
    letter-spacing: -0.2px;
}

/* All markdown text (prevents fade / low contrast) */
.stMarkdown, .stMarkdown * {
    color: rgba(255,255,255,0.96) !important;
    opacity: 1 !important;
}

/* Captions */
div[data-testid="stCaptionContainer"], 
div[data-testid="stCaptionContainer"] * {
    color: rgba(255,255,255,0.78) !important;
    font-weight: 750 !important;
    opacity: 1 !important;
}

/* =========================
   PANEL / CARD LOOK
========================= */
.panel {
    border-radius: 22px;
    padding: 18px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 18px 60px rgba(0,0,0,0.40);
}

/* =========================
   METRICS (NUMBERS WHITE FIX)
========================= */
div[data-testid="stMetricLabel"] *,
div[data-testid="stMetricLabel"] {
    color: rgba(255,255,255,0.86) !important;
    font-weight: 900 !important;
    opacity: 1 !important;
}

div[data-testid="stMetricValue"] *,
div[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 950 !important;
    opacity: 1 !important;
}

div[data-testid="stMetricDelta"] *,
div[data-testid="stMetricDelta"] {
    font-weight: 950 !important;
    opacity: 1 !important;
}

/* =========================
   INPUT LABELS (Ticker, Quantity etc.)
========================= */
label,
div[data-testid="stWidgetLabel"] * {
    color: #ffffff !important;
    opacity: 1 !important;
    font-weight: 900 !important;
    letter-spacing: 0.2px !important;
}

/* =========================
   ✅ FIX WHITE INPUT BOXES (IMPORTANT UPDATE)
========================= */

/* Outer input containers (Ticker, Search, etc.) */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
div[data-baseweb="base-input"] > div,
div[data-baseweb="base-input"],
textarea {
    background: rgba(10, 15, 35, 0.92) !important;
    border: 1px solid rgba(255,255,255,0.16) !important;
    border-radius: 18px !important;
    box-shadow: none !important;
}

/* Input text itself */
div[data-baseweb="input"] input,
div[data-baseweb="base-input"] input,
textarea {
    color: #ffffff !important;
    font-weight: 950 !important;
    background: transparent !important;
}

/* Placeholder */
div[data-baseweb="input"] input::placeholder,
textarea::placeholder {
    color: rgba(255,255,255,0.55) !important;
    font-weight: 800 !important;
}

/* Sometimes Streamlit forces white input backgrounds */
input {
    background: transparent !important;
    color: #ffffff !important;
}

/* =========================
   ✅ FIX NUMBER INPUT (+ / -) STYLING
========================= */
div[data-testid="stNumberInput"] div[data-baseweb="input"] > div {
    background: rgba(10, 15, 35, 0.92) !important;
}

div[data-testid="stNumberInput"] button {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 14px !important;
    color: #ffffff !important;
    font-weight: 950 !important;
}

div[data-testid="stNumberInput"] button:hover {
    background: rgba(255,255,255,0.10) !important;
    border-color: rgba(34,197,94,0.35) !important;
    transform: translateY(-1px);
}

/* =========================
   BUTTONS (PRO THEME)
========================= */
div.stButton > button,
div[data-testid="stDownloadButton"] > button {
    border-radius: 16px !important;
    height: 52px !important;
    font-size: 14px !important;
    font-weight: 950 !important;
    color: #07101a !important;
    background: linear-gradient(135deg, #22C55E 0%, #3B82F6 100%) !important;
    border: none !important;
    width: 100%;
    box-shadow: 0 16px 60px rgba(0,0,0,0.35);
    transition: 0.2s;
}

div.stButton > button:hover,
div[data-testid="stDownloadButton"] > button:hover {
    transform: translateY(-2px);
    filter: brightness(1.05);
}

/* =========================
   TABS (Explore / Paper Trade etc.)
========================= */
button[data-baseweb="tab"] {
    font-weight: 950 !important;
    opacity: 0.92 !important;
    color: rgba(255,255,255,0.75) !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #ffffff !important;
    opacity: 1 !important;
}

div[data-baseweb="tab-highlight"] {
    background: linear-gradient(90deg, #22C55E, #3B82F6) !important;
    height: 3px !important;
    border-radius: 999px !important;
}

/* =========================
   DATAFRAME LOOK (optional)
========================= */
div[data-testid="stDataFrame"] {
    border-radius: 18px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
}

/* =========================
   MOBILE RESPONSIVE
========================= */
@media(max-width: 1200px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}
</style>
""", unsafe_allow_html=True)





if "user" not in st.session_state:
    st.session_state["user"] = None

if "paper_selected_ticker" not in st.session_state:
    st.session_state["paper_selected_ticker"] = "AAPL"



@st.cache_data(ttl=120)
def yf_quote(ticker: str):
    """Return last close, change, change% (safe)"""
    try:
        df = yf.download(ticker, period="5d", interval="1d", progress=False)
        if df is None or df.empty:
            return None, None, None
        close = df["Close"].dropna()
        if len(close) < 2:
            return None, None, None

        last_close = float(close.iloc[-1])
        prev_close = float(close.iloc[-2])
        change = last_close - prev_close
        pct = (change / prev_close) * 100 if prev_close != 0 else 0.0
        return last_close, change, pct
    except Exception:
        return None, None, None


@st.cache_data(ttl=300)
def fx_usdinr():
    """USD to INR"""
    try:
        df = yf.download("USDINR=X", period="5d", interval="1d", progress=False)
        if df is None or df.empty:
            return 0.0
        return float(df["Close"].dropna().iloc[-1])
    except Exception:
        return 0.0


@st.cache_data(ttl=60)
def live_price_inr(ticker: str):
    """Get live price in INR even for US stocks."""
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        currency = (info.get("currency") or "").upper()

        hist = tk.history(period="2d", interval="1d")
        if hist is None or hist.empty:
            return None, currency

        price = float(hist["Close"].iloc[-1])

        if currency == "USD":
            fx = fx_usdinr()
            if fx > 0:
                return price * fx, currency
            return None, currency

        return price, currency
    except Exception:
        return None, ""


def fmt_inr(x):
    try:
        return f"₹{float(x):,.2f}"
    except Exception:
        return "₹0.00"



if not st.session_state["user"]:
    st.title("🧾 FinSight Paper Trading")
    st.caption("Educational simulator. Trade with virtual INR using real market prices (Yahoo Finance).")

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.subheader("Login")
        login_email = st.text_input("Email", key="login_email")
        login_pass = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login", use_container_width=True):
            user = login_user(login_email, login_pass)
            if not user:
                st.error("Invalid email or password.")
            else:
                st.session_state["user"] = user
                st.success("Login successful!")
                st.rerun()

    with c2:
        st.subheader("Signup")
        signup_email = st.text_input("New Email", key="signup_email")
        signup_pass = st.text_input("New Password", type="password", key="signup_pass")
        if st.button("Create Account", use_container_width=True):
            ok, msg = signup_user(signup_email, signup_pass)
            st.success(msg) if ok else st.error(msg)

    st.stop()



user = st.session_state["user"]
user_id = user["id"]

ensure_wallet(user_id)

# Top Bar
top1, top2, top3 = st.columns([2.3, 2.2, 1.0], gap="large")

with top1:
    st.markdown("## 📈 FinSight Dashboard")

with top2:
    search = st.text_input(
        "Search",
        placeholder="Search ticker… (AAPL / TSLA / TCS.NS / RELIANCE.NS)",
        label_visibility="collapsed"
    )

with top3:
    if st.button("Logout", use_container_width=True):
        logout()
        st.session_state["user"] = None
        st.rerun()

st.write("")

# Market strip (Streamlit-only, no HTML)
st.markdown("### Market Overview")
idx_cols = st.columns(4, gap="large")

indices = [
    ("NIFTY", "^NSEI"),
    ("SENSEX", "^BSESN"),
    ("BANKNIFTY", "^NSEBANK"),
    ("NASDAQ", "^IXIC"),
]

for i, (name, tick) in enumerate(indices):
    price, chg, pct = yf_quote(tick)
    if price is None:
        idx_cols[i].metric(name, "N/A", "")
    else:
        idx_cols[i].metric(name, f"{price:,.2f}", f"{chg:+.2f} ({pct:+.2f}%)")

st.write("")

# Tabs
tab_explore, tab_trade, tab_portfolio, tab_orders = st.tabs(
    ["Explore", "Paper Trade", "Portfolio", "Orders"]
)


with tab_explore:
    st.subheader("Most traded stocks on FinSight")

    most_traded = {
    # India (NSE)
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "SBI": "SBIN.NS",
    "ITC": "ITC.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Axis Bank": "AXISBANK.NS",
    "Kotak Bank": "KOTAKBANK.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "L&T": "LT.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "Tata Steel": "TATASTEEL.NS",
    "Wipro": "WIPRO.NS",
    "Zomato": "ZOMATO.NS",

    # US (NASDAQ/NYSE)
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "Google": "GOOGL",
    "Meta": "META",
    "Tesla": "TSLA",
    "NVIDIA": "NVDA",
    "AMD": "AMD",
    "Netflix": "NFLX",
    "Intel": "INTC",
}


    if search.strip():
        most_traded = {"Search Result": search.strip().upper()}

    grid = st.columns(4, gap="large")
    items = list(most_traded.items())


    for i, (nm, tk) in enumerate(items):
        price, chg, pct = yf_quote(tk)
        with grid[i % 4]:
            if price is None:
                st.metric(f"{nm} ({tk})", "N/A", "")
            else:
                st.metric(f"{nm} ({tk})", f"{price:,.2f}", f"{chg:+.2f} ({pct:+.2f}%)")
            if st.button(f"Trade {tk}", key=f"trade_btn_{tk}", use_container_width=True):
                st.session_state["paper_selected_ticker"] = tk
                st.success(f"Selected for trading: {tk}")

    st.write("")
    st.subheader("Your investments")
    holdings = get_holdings(user_id)
    if holdings.empty:
        st.info("You haven't invested yet.")
    else:
        wallet = get_balance(user_id)
        st.metric("Wallet Cash", fmt_inr(wallet))
        st.dataframe(holdings, use_container_width=True)


with tab_trade:
    st.subheader("Paper Trade (Virtual INR)")

    left, right = st.columns([1.05, 1.65], gap="large")

    with left:
        wallet = get_balance(user_id)
        st.metric("Wallet Balance", fmt_inr(wallet))

        ticker = st.text_input("Ticker", value=st.session_state["paper_selected_ticker"])
        ticker = (ticker or "").strip().upper()
        if ticker:
            st.session_state["paper_selected_ticker"] = ticker

        qty = st.number_input("Quantity", min_value=1.0, value=1.0, step=1.0)

        if st.button("Fetch Live Price", use_container_width=True):
            p, c = live_price_inr(ticker)
            if p is None:
                st.error("Could not fetch live price.")
            else:
                st.success(f"Live Price (INR): ₹{p:,.2f}  | Currency: {c}")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ BUY", use_container_width=True):
                p, c = live_price_inr(ticker)
                if p is None:
                    st.error("Price not available.")
                else:
                    ok, msg = buy_stock(user_id=user_id, ticker=ticker, qty=float(qty), price_inr=float(p))
                    st.success(msg) if ok else st.error(msg)

        with c2:
            if st.button("❌ SELL", use_container_width=True):
                p, c = live_price_inr(ticker)
                if p is None:
                    st.error("Price not available.")
                else:
                    ok, msg = sell_stock(user_id=user_id, ticker=ticker, qty=float(qty), price_inr=float(p))
                    st.success(msg) if ok else st.error(msg)

    with right:
        st.subheader("Portfolio Snapshot")
        holdings = get_holdings(user_id)

        if holdings.empty:
            st.info("No holdings yet.")
        else:
            live_values = []
            pnl_values = []

            for _, row in holdings.iterrows():
                t = row["ticker"]
                q = float(row["qty"])
                avg = float(row["avg_buy_price_inr"])

                live, _ = live_price_inr(t)
                if live is None:
                    live = avg

                val = live * q
                pnl = (live - avg) * q
                live_values.append(val)
                pnl_values.append(pnl)

            holdings = holdings.copy()
            holdings["live_value_inr"] = live_values
            holdings["pnl_inr"] = pnl_values

            total_holdings = float(np.sum(live_values))
            wallet = float(get_balance(user_id))
            total_portfolio = wallet + total_holdings

            m1, m2, m3 = st.columns(3)
            m1.metric("Holdings Value", fmt_inr(total_holdings))
            m2.metric("Wallet Cash", fmt_inr(wallet))
            m3.metric("Total Portfolio", fmt_inr(total_portfolio))

            st.dataframe(holdings, use_container_width=True)

# =========================
# PORTFOLIO
# =========================
with tab_portfolio:
    st.subheader("Portfolio")
    holdings = get_holdings(user_id)
    if holdings.empty:
        st.info("No holdings yet.")
    else:
        st.dataframe(holdings, use_container_width=True)

with tab_orders:
    st.subheader("Orders / Trades")
    trades = get_trades(user_id)
    if trades.empty:
        st.info("No trades yet.")
    else:
        st.dataframe(trades, use_container_width=True)

st.caption("FinSight • Paper Trading • Educational only • Not financial advice")
# Floating chatbot (if exists)
try:
    from utils.floating_chatbot import render_floating_chatbot
    render_floating_chatbot()
except ImportError:
    pass