# app.py - HOME PAGE
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="FinSight | AI Financial Advisor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: #0a0e1a;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Navigation */
    .nav-container {
        background: linear-gradient(135deg, rgba(26, 31, 53, 0.9) 0%, rgba(15, 18, 25, 0.9) 100%);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(255,255,255,0.08);
        padding: 20px 40px;
        margin: -60px -60px 30px -60px;
        position: sticky;
        top: 0;
        z-index: 100;
    }
    
    .nav-logo {
        font-size: 28px;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4ff 0%, #00ff88 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
        margin-right: 40px;
    }
    
    /* Hero Section */
    .hero {
        background: linear-gradient(135deg, #1a1f35 0%, #0f1219 100%);
        padding: 60px 40px;
        border-radius: 24px;
        margin-bottom: 40px;
        border: 1px solid rgba(255,255,255,0.08);
        text-align: center;
    }
    
    .hero-title {
        font-size: 56px;
        font-weight: 900;
        background: linear-gradient(135deg, #00d4ff 0%, #00ff88 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
        line-height: 1.2;
    }
    
    .hero-subtitle {
        font-size: 20px;
        color: rgba(255,255,255,0.7);
        margin-bottom: 30px;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .feature-pills {
        margin-top: 25px;
    }
    
    .feature-pill {
        display: inline-block;
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid rgba(0, 212, 255, 0.3);
        color: #00d4ff;
        padding: 10px 20px;
        border-radius: 25px;
        font-size: 14px;
        font-weight: 600;
        margin: 5px;
    }
    
    /* Market Ticker */
    .market-ticker {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .market-ticker:hover {
        background: rgba(255,255,255,0.05);
        border-color: rgba(0, 212, 255, 0.2);
        transform: translateY(-3px);
    }
    
    .ticker-name {
        font-size: 13px;
        color: rgba(255,255,255,0.6);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    
    .ticker-price {
        font-size: 24px;
        font-weight: 700;
        color: #ffffff;
        margin: 8px 0;
    }
    
    .ticker-change {
        font-size: 15px;
        font-weight: 600;
    }
    
    .positive { color: #00ff88; }
    .negative { color: #ff4757; }
    
    /* Section Cards */
    .section-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 35px;
        transition: all 0.3s ease;
        height: 100%;
        cursor: pointer;

        /* ✅ ADDED (alignment fix) */
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 420px;
    }

    /* ✅ ADDED */
    .card-content {
        flex: 1;
    }
    
    .section-card:hover {
        transform: translateY(-8px);
        border-color: rgba(0, 212, 255, 0.4);
        box-shadow: 0 15px 50px rgba(0, 212, 255, 0.2);
    }
    
    .card-icon {
        font-size: 48px;
        margin-bottom: 20px;
    }
    
    .card-title {
        font-size: 24px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 12px;
    }
    
    .card-description {
        font-size: 15px;
        color: rgba(255,255,255,0.6);
        line-height: 1.7;
        margin-bottom: 20px;
    }
    
    .section-header {
        font-size: 32px;
        font-weight: 800;
        color: #ffffff;
        margin: 50px 0 30px 0;
    }
    
    /* Buttons */
    div.stButton > button {
        border-radius: 12px !important;
        height: 50px !important;
        font-size: 15px !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #00d4ff 0%, #00ff88 100%) !important;
        color: #0a0e1a !important;
        border: none !important;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 10px 40px rgba(0, 212, 255, 0.3);
    }
    
    /* Trending Section */
    .trending-stock {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 18px;
        margin: 10px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: all 0.2s ease;
    }
    
    .trending-stock:hover {
        background: rgba(255,255,255,0.06);
        border-color: rgba(0, 212, 255, 0.3);
    }
    
    .stock-name {
        font-size: 16px;
        font-weight: 600;
        color: #ffffff;
    }
    
    .stock-price {
        font-size: 18px;
        font-weight: 700;
        color: #00d4ff;
    }
</style>
""", unsafe_allow_html=True)

# Navigation Bar
st.markdown("""
<div class="nav-container">
    <span class="nav-logo">📊 FinSight</span>
</div>
""", unsafe_allow_html=True)

# ✅ REMOVED: Navigation buttons block (big green pills)
# nav_cols = st.columns([1, 1, 1, 1, 6])
# with nav_cols[0]:
#     if st.button("🏠 Home", use_container_width=True):
#         st.rerun()
# with nav_cols[1]:
#     if st.button("📈 Stocks", use_container_width=True):
#         st.switch_page("pages/1_Stocks.py")
# with nav_cols[2]:
#     if st.button("💼 Mutual Funds", use_container_width=True):
#         st.switch_page("pages/2_Mutual_funds.py")
# with nav_cols[3]:
#     if st.button("🥇 Commodities", use_container_width=True):
#         st.switch_page("pages/3_commodities.py")

st.markdown("<br>", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero">
    <div class="hero-title">Welcome to FinSight AI</div>
    <div class="hero-subtitle">
        Your intelligent financial companion for stocks, mutual funds, and commodities analysis.
        Powered by advanced AI and real-time market data.
    </div>
    <div class="feature-pills">
        <span class="feature-pill">🤖 AI-Powered Insights</span>
        <span class="feature-pill">📊 Real-time Data</span>
        <span class="feature-pill">💼 Portfolio Analytics</span>
        <span class="feature-pill">🎯 Risk Assessment</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Market Overview
st.markdown('<div class="section-header">📊 Market Overview</div>', unsafe_allow_html=True)

@st.cache_data(ttl=300)
def get_index_data(ticker):
    try:
        data = yf.Ticker(ticker)
        hist = data.history(period="2d")
        if len(hist) >= 2:
            current = hist['Close'].iloc[-1]
            prev = hist['Close'].iloc[-2]
            change = current - prev
            change_pct = (change / prev) * 100
            return current, change, change_pct
        return None, None, None
    except:
        return None, None, None

indices = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "BANK NIFTY": "^NSEBANK",
    "NASDAQ": "^IXIC"
}

market_cols = st.columns(4)
for idx, (name, ticker) in enumerate(indices.items()):
    price, change, change_pct = get_index_data(ticker)
    
    if price:
        change_class = "positive" if change >= 0 else "negative"
        sign = "+" if change >= 0 else ""
        
        market_cols[idx].markdown(f"""
        <div class="market-ticker">
            <div class="ticker-name">{name}</div>
            <div class="ticker-price">{price:,.2f}</div>
            <div class="ticker-change {change_class}">{sign}{change:,.2f} ({sign}{change_pct:.2f}%)</div>
        </div>
        """, unsafe_allow_html=True)

# Quick Access Sections
st.markdown('<div class="section-header"> Explore Features</div>', unsafe_allow_html=True)

section_cols = st.columns(3)

with section_cols[0]:
    st.markdown("""
    <div class="section-card">
        <div class="card-content">
            <div class="card-icon">📈</div>
            <div class="card-title">Stock Analysis</div>
            <div class="card-description">
                Deep dive into stocks with AI-powered insights, fundamentals, risk metrics, 
                and comparison tools. Generate comprehensive reports instantly.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Analyze Stocks →", key="goto_stocks"):
        st.switch_page("pages/0_Stocks.py")

with section_cols[1]:
    st.markdown("""
    <div class="section-card">
        <div class="card-content">
            <div class="card-icon">💼</div>
            <div class="card-title">Mutual Funds</div>
            <div class="card-description">
                Explore top-performing mutual funds, compare returns, and build your 
                investment portfolio with expert recommendations.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("View Funds →", key="goto_mf"):
        st.switch_page("pages/2_Mutual_Funds_Predictor.py")

with section_cols[2]:
    st.markdown("""
    <div class="section-card">
        <div class="card-content">
            <div class="card-icon">🥇</div>
            <div class="card-title">Commodities</div>
            <div class="card-description">
                Track real-time prices of gold, silver, crude oil, and other commodities. 
                Stay updated with market trends.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("View Commodities →", key="goto_comm"):
        st.switch_page("pages/3_Commodities.py")

# Trending Stocks
st.markdown('<div class="section-header"> Trending Stocks</div>', unsafe_allow_html=True)

trending_cols = st.columns(2)

trending_stocks_left = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS"
}

trending_stocks_right = {
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Wipro": "WIPRO.NS"
}

with trending_cols[0]:
    for name, ticker in trending_stocks_left.items():
        price, change, change_pct = get_index_data(ticker)
        if price:
            change_class = "positive" if change >= 0 else "negative"
            sign = "+" if change >= 0 else ""
            st.markdown(f"""
            <div class="trending-stock">
                <div>
                    <div class="stock-name">{name}</div>
                    <div class="ticker-change {change_class}">{sign}{change_pct:.2f}%</div>
                </div>
                <div class="stock-price">₹{price:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

with trending_cols[1]:
    for name, ticker in trending_stocks_right.items():
        price, change, change_pct = get_index_data(ticker)
        if price:
            change_class = "positive" if change >= 0 else "negative"
            sign = "+" if change >= 0 else ""
            st.markdown(f"""
            <div class="trending-stock">
                <div>
                    <div class="stock-name">{name}</div>
                    <div class="ticker-change {change_class}">{sign}{change_pct:.2f}%</div>
                </div>
                <div class="stock-price">₹{price:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

st.caption("FinSight • AI Financial Advisor • Streamlit + Yahoo Finance")

# Floating chatbot (if exists)
try:
    from utils.floating_chatbot import render_floating_chatbot
    render_floating_chatbot()
except ImportError:
    pass


