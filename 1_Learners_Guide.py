import streamlit as st

st.set_page_config(
    page_title="FinSight | Investor Education",
    layout="wide"
)

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
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #FFFFFF !important;
        font-weight: 950 !important;
        letter-spacing: -0.6px;
    }

    p, li {
        color: rgba(255,255,255,0.86) !important;
        font-size: 15px;
        line-height: 1.7;
    }

    .hero {
        border-radius: 26px;
        padding: 28px 28px;
        background: linear-gradient(180deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0.03) 100%);
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 20px 60px rgba(0,0,0,0.35);
        margin-bottom: 18px;
    }

    .hero-title {
        font-size: 46px;
        font-weight: 950;
        margin: 0 0 10px 0;
        line-height: 1.05;
        color: #FFFFFF;
    }

    .hero-sub {
        margin: 0;
        font-size: 15px;
        color: rgba(255,255,255,0.82);
        line-height: 1.6;
        max-width: 1000px;
    }

    .tag {
        padding: 9px 14px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.05);
        font-size: 12px;
        font-weight: 850;
        color: rgba(255,255,255,0.95);
        display: inline-block;
        margin-top: 14px;
        margin-right: 8px;
    }

    .card {
        border-radius: 26px;
        padding: 20px 20px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 20px 60px rgba(0,0,0,0.35);
        margin-bottom: 18px;
    }

    .callout {
        margin-top: 14px;
        border-radius: 18px;
        padding: 14px 14px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.06);
        color: rgba(255,255,255,0.90);
        font-weight: 750;
    }

    .muted {
        color: rgba(255,255,255,0.70) !important;
        font-size: 13px;
    }

    hr {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.10);
        margin: 26px 0px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-title">Investor Education</div>
    <p class="hero-sub">
        A professional guide to stock market fundamentals — aligned with real-world market structure.
        Learn how stocks are issued, how exchanges function, how prices are discovered, and how investors
        should think about risk.
    </p>
    <span class="tag">Market structure</span>
    <span class="tag">Primary vs Secondary market</span>
    <span class="tag">Price discovery</span>
    <span class="tag">Risk & volatility</span>
    <span class="tag">Long-term investing</span>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.5, 1], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("1) What is the stock market?")
    st.markdown("""
The **stock market** refers to the collection of exchanges and marketplaces where shares of publicly traded companies are **issued** and **traded**.

At a high level, the stock market supports two major objectives:

- **Capital formation:** companies raise money to expand operations (factories, hiring, R&D)
- **Investment access:** investors can participate in business growth through ownership

Owning a share represents ownership in the company and can produce returns through:
- **Capital gains** (price increases)
- **Dividends** (profit distribution to shareholders)
""")
    st.markdown('<div class="callout">Professional view: stock markets exist primarily to allocate capital efficiently and enable price discovery — not just “buy low, sell high”.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("2) Primary market vs Secondary market")
    st.markdown("""
This distinction is essential and is often ignored by beginners.

### Primary market
This is where companies **issue new shares** and raise capital from investors.

Examples:
- **IPO (Initial Public Offering)**
- Follow-on public offers / new share issuance

### Secondary market
This is where investors **trade shares among themselves** on exchanges.

Key point:
- When you buy a stock normally on NSE/BSE/NYSE, you are almost always buying in the **secondary market**
- The company usually does **not** directly receive your money in a regular trade
""")
    st.markdown('<div class="callout">Why this matters: the stock market is both a capital-raising engine (primary) and a trading engine (secondary).</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("3) How stock exchanges work")
    st.markdown("""
A stock exchange is an organized marketplace that ensures trading occurs under standardized rules.

In modern markets:
- orders are matched electronically (buyer and seller)
- trades happen via brokers / trading platforms
- settlement happens after the trade (example: **T+1 / T+2** depending on the market)

Exchanges exist to provide:
- **liquidity** (ability to buy/sell easily)
- **transparency**
- standardized trading rules
""")
    st.markdown('<div class="callout">Professional term: exchanges enable “order matching” and make price information publicly available.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("4) Price discovery (why prices move)")
    st.markdown("""
The stock market is fundamentally a system of **price discovery**.

Prices move because the market continuously updates expectations about:
- future earnings
- growth
- interest rates
- competition
- risk

Common short-term price drivers:
- earnings results vs expectations
- guidance / outlook
- macroeconomic news (rates, inflation, policy)
- sector sentiment
- geopolitical news
""")
    st.markdown('<div class="callout">Investing insight: the market reacts to changes in expectations — not only whether the news is “good” or “bad”.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("5) Risk in the stock market")
    st.markdown("""
Professionals treat risk as measurable.

Key risk concepts:
- **Volatility:** intensity of price movements (day-to-day uncertainty)
- **Drawdown:** fall from peak to trough (loss depth)
- **Concentration risk:** too much exposure to one stock/sector
- **Liquidity risk:** inability to exit at a fair price during stress

FinSight uses:
- annualized volatility
- max drawdown
- Sharpe ratio (risk-adjusted performance)
""")
    st.markdown('<div class="callout">Rule: returns are optional; risk is unavoidable. Always manage risk before chasing returns.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Glossary")
    st.markdown("""
- **Stock / Share:** ownership unit in a company  
- **IPO:** first time shares are sold publicly  
- **Market Cap:** company value estimate (price × shares)  
- **Exchange:** marketplace for trading securities  
- **Liquidity:** ease of buying/selling without huge price impact  
- **Index:** benchmark basket (NIFTY 50 / S&P 500)  
- **ETF:** instrument that tracks an index/basket  
- **Volatility:** risk measured through price movement  
- **Sharpe Ratio:** return per unit risk  
- **Drawdown:** peak-to-trough fall percentage  
""")
    st.markdown('<p class="muted">These terms map directly to the data shown in FinSight.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Common mistakes")
    st.markdown("""
- trading without understanding volatility and drawdowns  
- buying based on hype without reading fundamentals  
- ignoring diversification  
- confusing “good company” with “good entry price”  
- investing without a time horizon  
""")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

st.header("A professional learning path")
c1, c2, c3 = st.columns(3, gap="large")

with c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Step 1: Learn market structure")
    st.markdown("Understand IPOs, exchanges, liquidity and price discovery before stock picking.")
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Step 2: Learn risk metrics")
    st.markdown("Use volatility, drawdown, and Sharpe ratio to evaluate instruments professionally.")
    st.markdown('</div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Step 3: Build portfolio discipline")
    st.markdown("Diversify, rebalance, avoid concentrated positions and short-term emotional decisions.")
    st.markdown('</div>', unsafe_allow_html=True)

with st.expander("FAQ", expanded=False):
    st.markdown("""
**Is the stock market the economy?**  
Not exactly. The stock market is forward-looking and reflects expectations about future profits and risk.

**Does buying stock help the company?**  
In the primary market (IPO/new issuance): yes. In normal trading (secondary): usually no.

**Should I use AI to invest?**  
AI should support learning and analysis. Final decisions should reflect your own risk tolerance and time horizon.
""")

st.caption("FinSight Education • For learning purposes only • Not financial advice")
