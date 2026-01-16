import re
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import requests
from datetime import datetime

try:
    import yfinance as yf
    YF_AVAILABLE = True
except Exception:
    YF_AVAILABLE = False


st.set_page_config(
    page_title="FinSight | Mutual Funds",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed"
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
        font-size: 44px;
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
        max-width: 920px;
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
    div[data-testid="stWidgetLabel"] * {
        color: #FFFFFF !important;
        opacity: 1 !important;
        font-weight: 900 !important;
        letter-spacing: 0.2px !important;
    }

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

    div[data-testid="stDownloadButton"] > button {
        border-radius: 18px !important;
        height: 56px !important;
        font-size: 15px !important;
        font-weight: 950 !important;
        color: #0B1220 !important;
        background: linear-gradient(135deg, #22C55E 0%, #3B82F6 100%) !important;
        border: none !important;
        width: 100% !important;
        box-shadow: 0 18px 60px rgba(0,0,0,0.45) !important;
        transition: 0.2s !important;
    }

    div[data-testid="stDownloadButton"] > button:hover {
        transform: translateY(-1px) !important;
        filter: brightness(1.06) !important;
    }

    div[data-baseweb="select"] * {
        color: #ffffff !important;
        font-weight: 950 !important;
        opacity: 1 !important;
    }

    div[data-testid="stSlider"] * {
        color: #ffffff !important;
        opacity: 1 !important;
        font-weight: 950 !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>Mutual Funds — NAV, Risk & Benchmark</h1>
    <p>
        Select mutual funds like Groww: choose the main fund name, then pick Direct/Regular and Growth/IDCW.
        Generate NAV trend, risk metrics, compare schemes and benchmark vs NIFTY 50 + SENSEX.
    </p>
    <span class="tag">AMFI NAV</span>
    <span class="tag">Risk Analysis</span>
    <span class="tag">Benchmark Comparison</span>
    <span class="tag">Chart + CSV</span>
</div>
""", unsafe_allow_html=True)


@st.cache_data(ttl=86400)
def fetch_amfi_navall_text():
    url = "https://www.amfiindia.com/spages/NAVAll.txt"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.text


def parse_navall(text: str) -> pd.DataFrame:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    rows = []
    for ln in lines:
        if ln.startswith("Scheme Code") or ln.startswith("---"):
            continue
        parts = ln.split(";")
        if len(parts) < 6:
            continue
        rows.append({
            "scheme_code": parts[0].strip(),
            "scheme_name": parts[3].strip(),
            "nav": parts[4].strip(),
            "date": parts[5].strip()
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna(subset=["nav"])
    return df


@st.cache_data(ttl=86400)
def load_fund_master() -> pd.DataFrame:
    return parse_navall(fetch_amfi_navall_text())


@st.cache_data(ttl=86400)
def fetch_scheme_nav_history(scheme_code: str) -> pd.DataFrame:
    url = f"https://api.mfapi.in/mf/{scheme_code}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()
    nav_data = data.get("data", []) or []
    out = []
    for item in nav_data:
        out.append({"date": item.get("date"), "nav": item.get("nav")})
    df = pd.DataFrame(out)
    if df.empty:
        return df
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="%d-%m-%Y")
    df = df.dropna(subset=["date", "nav"]).sort_values("date")
    return df


def normalize_fund_name(name: str) -> str:
    if not name:
        return ""
    s = str(name)
    patterns = [
        r"\s*-\s*Direct\s*Plan.*",
        r"\s*-\s*Regular\s*Plan.*",
        r"\s*-\s*Plan.*",
        r"\s*-\s*Growth.*",
        r"\s*-\s*IDCW.*",
        r"\s*-\s*Dividend.*",
        r"\s*-\s*Bonus.*",
        r"\s*-\s*Daily.*",
        r"\s*-\s*Weekly.*",
        r"\s*-\s*Monthly.*",
        r"\s*-\s*Quarterly.*",
        r"\s*-\s*Annual.*",
    ]
    for pat in patterns:
        s = re.sub(pat, "", s, flags=re.IGNORECASE).strip()
    return re.sub(r"\s+", " ", s).strip()


def variant_label(full_name: str) -> str:
    x = (full_name or "").lower()
    plan = "Direct" if "direct" in x else ("Regular" if "regular" in x else "Plan")
    option = "Growth" if "growth" in x else ("IDCW" if ("idcw" in x or "dividend" in x) else "Option")
    return f"{plan} • {option}"


def compute_returns(series: pd.Series) -> pd.Series:
    return series.pct_change().dropna()


def sharpe_ratio(returns: pd.Series, rf_annual: float = 0.0) -> float:
    if len(returns) < 2:
        return float("nan")
    rf_daily = rf_annual / 252
    excess = returns - rf_daily
    if excess.std() == 0:
        return float("nan")
    return float((excess.mean() / excess.std()) * math.sqrt(252))


def max_drawdown(series: pd.Series) -> float:
    prices = series.values
    peak = -np.inf
    mdd = 0.0
    for p in prices:
        peak = max(peak, p)
        dd = (peak - p) / peak if peak > 0 else 0.0
        mdd = max(mdd, dd)
    return float(mdd)


def cagr_from_nav(df_nav: pd.DataFrame) -> float:
    if df_nav.empty or len(df_nav) < 2:
        return float("nan")
    start = float(df_nav["nav"].iloc[0])
    end = float(df_nav["nav"].iloc[-1])
    d0 = df_nav["date"].iloc[0]
    d1 = df_nav["date"].iloc[-1]
    years = max(1e-9, (d1 - d0).days / 365.25)
    if start <= 0:
        return float("nan")
    return float((end / start) ** (1 / years) - 1)


def trailing_return(df_nav: pd.DataFrame, days: int) -> float:
    if df_nav.empty or len(df_nav) < 2:
        return float("nan")
    latest_date = df_nav["date"].iloc[-1]
    cutoff = latest_date - pd.Timedelta(days=days)
    df2 = df_nav[df_nav["date"] >= cutoff]
    if len(df2) < 2:
        return float("nan")
    start = float(df2["nav"].iloc[0])
    end = float(df2["nav"].iloc[-1])
    if start <= 0:
        return float("nan")
    return float(end / start - 1)


def risk_label(vol: float) -> str:
    if vol is None or np.isnan(vol):
        return "N/A"
    if vol < 0.12:
        return "LOW"
    if vol < 0.22:
        return "MEDIUM"
    return "HIGH"


@st.cache_data(ttl=3600)
def fetch_benchmark_prices(ticker: str, start_date, end_date) -> pd.DataFrame:
    if not YF_AVAILABLE:
        return pd.DataFrame()
    try:
        df = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        if "Date" not in df.columns or "Close" not in df.columns:
            return pd.DataFrame()
        df.rename(columns={"Date": "date", "Close": "close"}, inplace=True)
        df = df.dropna(subset=["date", "close"])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["date", "close"]).sort_values("date")
        return df[["date", "close"]]
    except Exception:
        return pd.DataFrame()


def normalize_growth(series: pd.Series) -> pd.Series:
    if series is None or len(series) == 0:
        return series
    first = float(series.iloc[0])
    if first == 0:
        return series
    return series / first


def align_series_on_dates(df_nav: pd.DataFrame, df_bench: pd.DataFrame) -> pd.DataFrame:
    fund = df_nav[["date", "nav"]].copy()
    fund.rename(columns={"nav": "fund"}, inplace=True)

    bench = df_bench[["date", "close"]].copy()
    bench.rename(columns={"close": "bench"}, inplace=True)

    merged = pd.merge(fund, bench, on="date", how="inner")
    merged = merged.dropna()
    return merged


def build_compare_row(display_name: str, scheme_code: str, df_nav: pd.DataFrame, rf_annual: float) -> dict:
    rets = compute_returns(df_nav["nav"])
    vol = float(rets.std() * math.sqrt(252)) if len(rets) > 2 else float("nan")
    sr = float(sharpe_ratio(rets, rf_annual)) if len(rets) > 2 else float("nan")
    mdd = float(max_drawdown(df_nav["nav"])) if not df_nav.empty else float("nan")
    return {
        "Fund": display_name,
        "Scheme Code": scheme_code,
        "Latest NAV": float(df_nav["nav"].iloc[-1]) if not df_nav.empty else float("nan"),
        "1M Return": trailing_return(df_nav, 30),
        "3M Return": trailing_return(df_nav, 90),
        "1Y Return": trailing_return(df_nav, 365),
        "CAGR (overall)": cagr_from_nav(df_nav),
        "Volatility": vol,
        "Sharpe": sr,
        "Max Drawdown": mdd,
        "Risk": risk_label(vol),
    }


if "mf_selected_compare" not in st.session_state:
    st.session_state["mf_selected_compare"] = []

if "mf_run_report" not in st.session_state:
    st.session_state["mf_run_report"] = False

if "mf_run_compare" not in st.session_state:
    st.session_state["mf_run_compare"] = False


left, right = st.columns([1.05, 1.65], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Input</div>', unsafe_allow_html=True)

    df_master = load_fund_master()

    query = st.text_input(
        "Mutual fund search",
        value="",
        placeholder="Example: HDFC Balanced | HDFC Flexi | ICICI Bluechip | Parag Parikh | Quant"
    )

    days_limit = st.slider("History window (days)", min_value=365, max_value=3650, value=1800, step=30)

    rf_annual_ui = st.slider("Risk-free rate (annual, %)", min_value=0.0, max_value=12.0, value=4.0, step=0.25)
    rf_annual = rf_annual_ui / 100.0

    chosen_code = None
    chosen_display = None

    if query and not df_master.empty:
        temp = df_master[df_master["scheme_name"].str.lower().str.contains(query.lower(), na=False)].copy()

        if temp.empty:
            st.info("No funds found. Try a different keyword.")
        else:
            temp["base_name"] = temp["scheme_name"].apply(normalize_fund_name)

            base_names = (
                temp[["base_name"]]
                .dropna()
                .drop_duplicates()
                .head(250)["base_name"]
                .tolist()
            )

            st.caption("Select Mutual Fund (Main Name)")
            base_selected = st.selectbox("Mutual Fund", base_names, index=0)

            variants = temp[temp["base_name"] == base_selected].copy()
            variants["variant"] = variants["scheme_name"].apply(variant_label)

            def rank_variant(v: str) -> int:
                vl = v.lower()
                score = 100
                if "direct" in vl:
                    score -= 20
                if "growth" in vl:
                    score -= 10
                if "regular" in vl:
                    score += 10
                if "idcw" in vl:
                    score += 5
                return score

            variants["rank"] = variants["variant"].apply(rank_variant)
            variants = variants.sort_values(["rank", "scheme_name"], ascending=[True, True])

            st.caption("Select Variant")
            variant_options = variants["variant"].drop_duplicates().tolist()
            variant_selected = st.selectbox("Variant", variant_options, index=0)

            chosen_row = variants[variants["variant"] == variant_selected].iloc[0]
            chosen_code = str(chosen_row["scheme_code"])
            chosen_display = base_selected + " — " + variant_selected

            st.caption(f"Selected: {chosen_display}")

            st.write("")
            st.markdown("#### Compare list")

            if st.button("Add this fund to Compare"):
                existing_codes = [x["scheme_code"] for x in st.session_state["mf_selected_compare"]]
                if chosen_code and chosen_display and chosen_code not in existing_codes:
                    st.session_state["mf_selected_compare"].append({
                        "scheme_code": chosen_code,
                        "display": chosen_display
                    })
                    st.success("Added to compare list.")
                else:
                    st.info("Already added or not selected.")
    else:
        st.info("Start typing a mutual fund name to search.")

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Generate Fund Report"):
            st.session_state["mf_run_report"] = True
            st.session_state["mf_run_compare"] = False
    with c2:
        if st.button("Run Comparison"):
            st.session_state["mf_run_report"] = False
            st.session_state["mf_run_compare"] = True

    st.write("")
    if st.session_state["mf_selected_compare"]:
        show_list = [f"{i+1}. {x['display']}" for i, x in enumerate(st.session_state["mf_selected_compare"])]
        st.caption("Compare selected: " + " | ".join(show_list[:5]))

        d1, d2 = st.columns(2)
        with d1:
            if st.button("Remove last"):
                st.session_state["mf_selected_compare"].pop()
        with d2:
            if st.button("Clear compare list"):
                st.session_state["mf_selected_compare"] = []

    st.markdown("</div>", unsafe_allow_html=True)


with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Results</div>', unsafe_allow_html=True)

    if st.session_state["mf_run_compare"]:
        selected = st.session_state["mf_selected_compare"][:5]
        if len(selected) < 2:
            st.info("Add at least 2 funds in compare list, then click **Run Comparison**.")
        else:
            with st.spinner("Generating comparison report..."):
                compare_rows = []
                nav_map = {}
                for item in selected:
                    df_nav = fetch_scheme_nav_history(item["scheme_code"])
                    if df_nav.empty:
                        continue
                    df_nav = df_nav.tail(days_limit).copy()
                    nav_map[item["display"]] = df_nav
                    compare_rows.append(build_compare_row(item["display"], item["scheme_code"], df_nav, rf_annual))
                comp_df = pd.DataFrame(compare_rows)

            if comp_df.empty:
                st.error("Could not fetch NAV data for selected schemes.")
            else:
                t0, t1, t2, t3 = st.tabs(["Overview", "Risk Analysis", "Compare Table", "Chart"])

                with t0:
                    st.subheader("Comparison Overview")
                    st.caption("Key highlights across all selected funds.")

                    best_sharpe = comp_df.sort_values("Sharpe", ascending=False).iloc[0]["Fund"]
                    lowest_dd = comp_df.sort_values("Max Drawdown", ascending=True).iloc[0]["Fund"]
                    highest_cagr = comp_df.sort_values("CAGR (overall)", ascending=False).iloc[0]["Fund"]

                    a1, a2, a3 = st.columns(3)
                    with a1:
                        st.markdown("### Best Sharpe")
                        st.markdown(f"**{best_sharpe}**")
                    with a2:
                        st.markdown("### Lowest Drawdown")
                        st.markdown(f"**{lowest_dd}**")
                    with a3:
                        st.markdown("### Highest CAGR")
                        st.markdown(f"**{highest_cagr}**")

                    st.write("")
                    st.subheader("Export Comparison CSV")
                    st.download_button(
                        "Download mutual_funds_comparison.csv",
                        data=comp_df.to_csv(index=False),
                        file_name="mutual_funds_comparison.csv",
                        mime="text/csv"
                    )

                with t1:
                    st.subheader("Risk Analysis (Compare)")
                    show_cols = ["Fund", "Risk", "Volatility", "Sharpe", "Max Drawdown"]
                    st.dataframe(comp_df[show_cols], use_container_width=True)

                with t2:
                    st.subheader("Comparison Table")
                    st.dataframe(comp_df, use_container_width=True)

                with t3:
                    st.subheader("NAV Trend (All Funds)")
                    fig = go.Figure()
                    for name, df_nav in nav_map.items():
                        fig.add_trace(go.Scatter(
                            x=df_nav["date"],
                            y=df_nav["nav"],
                            mode="lines",
                            name=name
                        ))
                    fig.update_layout(
                        title="NAV Trend Comparison",
                        xaxis_title="Date",
                        yaxis_title="NAV",
                        height=460
                    )
                    st.plotly_chart(fig, use_container_width=True)

    elif st.session_state["mf_run_report"]:
        if not chosen_code:
            st.info("Search a fund, select a variant, then click **Generate Fund Report**.")
        else:
            with st.spinner("Fetching NAV history..."):
                df_nav = fetch_scheme_nav_history(chosen_code)

            if df_nav.empty or len(df_nav) < 20:
                st.error("Not enough NAV data found for this scheme.")
            else:
                df_nav = df_nav.tail(days_limit).copy()
                rets = compute_returns(df_nav["nav"])

                vol = float(rets.std() * math.sqrt(252)) if len(rets) > 2 else float("nan")
                sr = float(sharpe_ratio(rets, rf_annual)) if len(rets) > 2 else float("nan")
                mdd = float(max_drawdown(df_nav["nav"]))

                latest_nav = float(df_nav["nav"].iloc[-1])
                latest_date = df_nav["date"].iloc[-1].strftime("%Y-%m-%d")

                r_1m = trailing_return(df_nav, 30)
                r_3m = trailing_return(df_nav, 90)
                r_1y = trailing_return(df_nav, 365)
                cagr = cagr_from_nav(df_nav)

                tab0, tab1, tab2, tab3 = st.tabs(["Overview", "Risk Analysis", "Benchmark Comparison", "Chart + CSV"])

                with tab0:
                    st.subheader("Fund Overview")
                    st.caption(chosen_display)

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Scheme Code", chosen_code)
                    m2.metric("Latest NAV", f"{latest_nav:.2f}")
                    m3.metric("NAV Date", latest_date)
                    m4.metric("Risk Label", risk_label(vol))

                    m5, m6, m7, m8 = st.columns(4)
                    m5.metric("1M Return", f"{r_1m:.2%}" if not np.isnan(r_1m) else "N/A")
                    m6.metric("3M Return", f"{r_3m:.2%}" if not np.isnan(r_3m) else "N/A")
                    m7.metric("1Y Return", f"{r_1y:.2%}" if not np.isnan(r_1y) else "N/A")
                    m8.metric("CAGR (overall)", f"{cagr:.2%}" if not np.isnan(cagr) else "N/A")

                with tab1:
                    st.subheader("Risk Analysis")
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Volatility (ann.)", f"{vol:.2%}" if not np.isnan(vol) else "N/A")
                    r2.metric("Sharpe Ratio", f"{sr:.2f}" if not np.isnan(sr) else "N/A")
                    r3.metric("Max Drawdown", f"{mdd:.2%}" if not np.isnan(mdd) else "N/A")
                    r4.metric("Data points", str(len(df_nav)))

                with tab2:
                    st.subheader("Benchmark Comparison (NIFTY 50 + SENSEX)")

                    if not YF_AVAILABLE:
                        st.info("Benchmark comparison needs yfinance installed.")
                    else:
                        start_date = df_nav["date"].iloc[0]
                        end_date = df_nav["date"].iloc[-1]

                        nifty = fetch_benchmark_prices("^NSEI", start_date, end_date)
                        sensex = fetch_benchmark_prices("^BSESN", start_date, end_date)

                        if nifty.empty or sensex.empty:
                            st.info("Benchmark data not available right now. Try again later.")
                        else:
                            m1 = align_series_on_dates(df_nav, nifty)
                            m2 = align_series_on_dates(df_nav, sensex)

                            if m1.empty or m2.empty:
                                st.info("Benchmarks could not align with NAV dates.")
                            else:
                                fund_growth = normalize_growth(m1["fund"])
                                nifty_growth = normalize_growth(m1["bench"])

                                fund_growth2 = normalize_growth(m2["fund"])
                                sensex_growth = normalize_growth(m2["bench"])

                                fund_cagr = cagr_from_nav(df_nav)

                                nifty_cagr = float((nifty["close"].iloc[-1] / nifty["close"].iloc[0]) ** (365.25 / max(1, (nifty["date"].iloc[-1] - nifty["date"].iloc[0]).days)) - 1)
                                sensex_cagr = float((sensex["close"].iloc[-1] / sensex["close"].iloc[0]) ** (365.25 / max(1, (sensex["date"].iloc[-1] - sensex["date"].iloc[0]).days)) - 1)

                                s1, s2, s3 = st.columns(3)
                                s1.metric("Fund CAGR", f"{fund_cagr:.2%}" if not np.isnan(fund_cagr) else "N/A")
                                s2.metric("NIFTY 50 CAGR", f"{nifty_cagr:.2%}" if not np.isnan(nifty_cagr) else "N/A")
                                s3.metric("SENSEX CAGR", f"{sensex_cagr:.2%}" if not np.isnan(sensex_cagr) else "N/A")

                                figb = go.Figure()
                                figb.add_trace(go.Scatter(x=m1["date"], y=fund_growth, mode="lines", name="Fund (Growth Index)"))
                                figb.add_trace(go.Scatter(x=m1["date"], y=nifty_growth, mode="lines", name="NIFTY 50"))
                                figb.add_trace(go.Scatter(x=m2["date"], y=sensex_growth, mode="lines", name="SENSEX"))

                                figb.update_layout(
                                    title="Benchmark Growth Comparison (Normalized)",
                                    xaxis_title="Date",
                                    yaxis_title="Growth Index (Start = 1.0)",
                                    height=460
                                )
                                st.plotly_chart(figb, use_container_width=True)

                                if not np.isnan(fund_cagr) and not np.isnan(nifty_cagr):
                                    alpha_nifty = fund_cagr - nifty_cagr
                                else:
                                    alpha_nifty = float("nan")

                                if not np.isnan(fund_cagr) and not np.isnan(sensex_cagr):
                                    alpha_sensex = fund_cagr - sensex_cagr
                                else:
                                    alpha_sensex = float("nan")

                                st.caption(
                                    f"Outperformance vs NIFTY50: {alpha_nifty:+.2%} | "
                                    f"vs SENSEX: {alpha_sensex:+.2%}"
                                )

                with tab3:
                    st.subheader("NAV Chart")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_nav["date"],
                        y=df_nav["nav"],
                        mode="lines",
                        name="NAV"
                    ))
                    fig.update_layout(
                        title="NAV Trend",
                        xaxis_title="Date",
                        yaxis_title="NAV",
                        height=460
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.write("")
                    st.subheader("Export NAV CSV")
                    st.download_button(
                        "Download nav_history.csv",
                        data=df_nav.to_csv(index=False),
                        file_name="nav_history.csv",
                        mime="text/csv"
                    )

    else:
        st.info("Search a fund and click **Generate Fund Report**, or add multiple funds and click **Run Comparison**.")

    st.markdown("</div>", unsafe_allow_html=True)



st.caption("FinSight • Mutual Funds • AMFI NAV + mfapi.in + Benchmarks")


