import streamlit as st

from utils.auth import require_login
from utils.payments import create_order
from utils.trading_db import get_balance, get_topups


# =========================
# AUTH
# =========================
require_login()
user_id = st.session_state["user"]["id"]


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="FinSight | Buy Virtual Cash",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================
# STYLE (FIX WHITE BOXES + THEME)
# =========================
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at 10% 0%, #0b1220 0%, #050814 45%, #050814 100%);
    font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
    color: #ffffff !important;
}

.block-container {
    max-width: 1400px;
    padding-top: 3.0rem;
    padding-bottom: 2.5rem;
}

h1, h2, h3 {
    color: #ffffff !important;
    font-weight: 950 !important;
}

.stMarkdown, .stMarkdown * {
    color: rgba(255,255,255,0.95) !important;
    opacity: 1 !important;
}

/* ✅ Fix white inputs everywhere */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
div[data-baseweb="base-input"] > div,
textarea {
    background: rgba(10, 15, 35, 0.95) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 16px !important;
}

div[data-baseweb="input"] input,
textarea {
    color: #ffffff !important;
    font-weight: 900 !important;
    background: transparent !important;
}

::placeholder {
    color: rgba(255,255,255,0.55) !important;
}

/* ✅ Cards */
.package-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 22px;
    padding: 26px;
    text-align: center;
    box-shadow: 0 18px 60px rgba(0,0,0,0.45);
    transition: all 0.25s ease;
}

.package-card:hover {
    transform: translateY(-6px);
    border-color: rgba(34,197,94,0.5);
}

/* Prices */
.price {
    font-size: 28px;
    font-weight: 950;
    margin-bottom: 6px;
}
.virtual {
    font-size: 22px;
    font-weight: 900;
    color: #22c55e;
    margin-bottom: 14px;
}
.note {
    color: rgba(255,255,255,0.65);
    font-size: 14px;
}

/* ✅ Buttons */
div.stButton > button {
    background: linear-gradient(135deg, #22C55E, #3B82F6) !important;
    border: none !important;
    border-radius: 16px !important;
    font-weight: 950 !important;
    height: 52px !important;
    color: white !important;
    width: 100%;
    box-shadow: 0 16px 60px rgba(0,0,0,0.35);
}

div.stButton > button:hover {
    transform: translateY(-2px);
    filter: brightness(1.06);
}

/* Table */
div[data-testid="stDataFrame"] {
    border-radius: 16px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
}
</style>
""", unsafe_allow_html=True)


# =========================
# HEADER
# =========================
st.markdown("## 💳 Buy Virtual Cash")
st.caption("Recharge your trading wallet with virtual money to practice stock trading using real market prices.")


# =========================
# WALLET BALANCE
# =========================
balance = get_balance(user_id)
st.metric("Current Wallet Balance", f"₹{balance:,.2f}")


# =========================
# PACKAGES
# =========================
packages = [
    (50, 50000),
    (100, 120000),
    (250, 350000),
    (500, 800000),
]

rows = [packages[:2], packages[2:]]

for row in rows:
    cols = st.columns(2, gap="large")
    for col, pkg in zip(cols, row):
        real_price, virtual_amount = pkg

        with col:
            st.markdown(f"""
            <div class="package-card">
                <div class="price">₹{real_price}</div>
                <div class="virtual">₹{virtual_amount:,} Virtual Cash</div>
                <div class="note">Instant wallet credit after successful payment</div>
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"Buy ₹{virtual_amount:,}", key=f"buy_{virtual_amount}"):

                # ✅ create Razorpay order (notes include user_id + virtual_amount)
                order = create_order(real_price, user_id, virtual_amount)

                st.session_state["pending_payment"] = {
                    "order_id": order["id"],
                    "real": real_price,
                    "virtual": virtual_amount,
                }

                st.success("✅ Order created. Click below to pay 👇")


# =========================
# PAYMENT BUTTON
# =========================
if "pending_payment" in st.session_state:
    p = st.session_state["pending_payment"]

    st.write("")
    st.markdown("### ✅ Complete Payment")
    st.caption("After payment, the webhook will credit your wallet automatically.")

    payment_html = f"""
    <script src="https://checkout.razorpay.com/v1/checkout.js"></script>

    <button id="rzp-button" style="
        width:100%;
        padding:16px 24px;
        font-size:18px;
        border-radius:14px;
        background:#22c55e;
        color:white;
        border:none;
        cursor:pointer;
        font-weight:900;">
        Pay ₹{p['real']}
    </button>

    <script>
    const options = {{
        key: "{st.secrets['RAZORPAY_KEY_ID']}",
        amount: "{int(p['real'] * 100)}",
        currency: "INR",
        name: "FinSight",
        description: "Buy Virtual Cash",
        order_id: "{p['order_id']}",
        theme: {{
            color: "#22c55e"
        }},
        handler: function (response) {{
            alert("✅ Payment successful! Wallet will update in a moment.");
            window.location.reload();
        }},
    }};

    const rzp1 = new Razorpay(options);

    document.getElementById("rzp-button").onclick = function(e) {{
        rzp1.open();
        e.preventDefault();
    }};
    </script>
    """

    # more height so popup overlay UI is not cramped
    st.components.v1.html(payment_html, height=650)

    if st.button("❌ Cancel Pending Payment", use_container_width=True):
        del st.session_state["pending_payment"]
        st.rerun()


# =========================
# INFO
# =========================
st.info("🕒 After payment, wallet will be credited once Razorpay webhook confirms payment.")


# =========================
# PURCHASE HISTORY
# =========================
st.write("")
st.markdown("## Purchase History")

history_df = get_topups(user_id)
if history_df.empty:
    st.info("No purchases yet.")
else:
    st.dataframe(history_df, use_container_width=True)

st.caption("Virtual cash is for trading practice only.")
