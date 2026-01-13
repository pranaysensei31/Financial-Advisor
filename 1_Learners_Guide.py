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
        max-width: 950px;
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
        font-weight: 700;
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
        This section provides structured learning content about the stock market.
        It is designed for beginners, but written in a professional investor tone.
        Use this guide to understand concepts like volatility, drawdowns, valuation, and portfolio risk.
    </p>
    <span class="tag">Beginner → Intermediate</span>
    <span class="tag">Risk & volatility</span>
    <span class="tag">Long-term investing</span>
    <span class="tag">Portfolio discipline</span>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.5, 1], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("1) What is the stock market?")
    st.markdown("""
The stock market is a system where ownership units of public companies (**shares**) are traded.
When you buy a share, you become a partial owner of the company.

Returns come mainly from:
- **Price appreciation** (capital gain)
- **Dividends** (profit distribution)
""")
    st.markdown('<div class="callout">Key idea: The stock market is a pricing mechanism that reflects business performance, future expectations and risk.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("2) Why do stock prices move?")
    st.markdown("""
Prices change because markets continuously update beliefs about a company’s value.

Main drivers:
- **Earnings vs expectations**
- **Guidance / future outlook**
- **Macro factors** (rates, inflation, recession)
- **Sector sentiment**
- **Regulation, lawsuits, innovation**
""")
    st.markdown('<div class="callout">Reality: markets react to changes in expectation, not only to good or bad news.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("3) Risk vs Return (most important)")
    st.markdown("""
**Return** = how much you make  
**Risk** = uncertainty of outcomes

Risk in real markets appears as:
- **Volatility** (how aggressively price moves)
- **Drawdown** (how deeply the price falls from highs)
- **Business risk** (competition, debt, disruption)
""")
    st.markdown('<div class="callout">FinSight focuses heavily on volatility, drawdown and Sharpe ratio because they quantify risk in real terms.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Glossary")
    st.markdown("""
- **Market Cap:** Company size estimate  
- **P/E Ratio:** Valuation relative to earnings  
- **Index:** Benchmark basket (NIFTY 50 / S&P 500)  
- **ETF:** Basket instrument tracking an index  
- **Volatility:** Price movement intensity  
- **Sharpe Ratio:** Risk-adjusted performance  
- **Drawdown:** Peak-to-trough fall (%)  
""")
    st.markdown('<p class="muted">Tip: These terms appear directly inside FinSight analytics.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Common beginner mistakes")
    st.markdown("""
- Buying purely from social media hype  
- Overtrading (too much noise, no system)  
- No diversification  
- Not understanding drawdowns  
- Confusing a great company with a great price  
""")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

st.header("Suggested learning path")
c1, c2, c3 = st.columns(3, gap="large")

with c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Step 1: Understand indices")
    st.markdown("Track benchmark movements and learn market cycles before picking individual stocks.")
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Step 2: Learn risk metrics")
    st.markdown("Use volatility, Sharpe ratio and drawdown to compare instruments properly.")
    st.markdown('</div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Step 3: Portfolio discipline")
    st.markdown("Diversify, rebalance, and avoid concentration in highly volatile instruments.")
    st.markdown('</div>', unsafe_allow_html=True)

with st.expander("FAQ", expanded=False):
    st.markdown("""
**Can I invest without picking stocks?**  
Yes. Index ETFs provide diversified exposure.

**Is volatility always bad?**  
Not always, but it increases uncertainty and drawdowns. Your time horizon matters.

**Should I trust AI recommendations fully?**  
No. AI should assist decisions, not replace your judgment.
""")

st.caption("FinSight Education • For learning purposes only • Not financial advice")

