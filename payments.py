import razorpay
import streamlit as st
import hmac
import hashlib


def get_client():
    return razorpay.Client(auth=(
        st.secrets["RAZORPAY_KEY_ID"],
        st.secrets["RAZORPAY_KEY_SECRET"]
    ))


def create_order(amount_inr: float, user_id: int, virtual_amount: float):
    """
    Create Razorpay order with metadata in notes.
    amount_inr -> real money user pays (₹)
    virtual_amount -> how much virtual wallet you will credit (₹)
    """
    client = get_client()

    data = {
        "amount": int(float(amount_inr) * 100),  # paise
        "currency": "INR",
        "payment_capture": 1,
        "notes": {
            "user_id": str(user_id),
            "virtual_amount": str(virtual_amount)
        }
    }

    return client.order.create(data=data)


def verify_signature(order_id: str, payment_id: str, signature: str) -> bool:
    """
    Verify signature returned from checkout:
    expected signature = HMAC_SHA256(order_id|payment_id, key_secret)
    """
    secret = st.secrets["RAZORPAY_KEY_SECRET"]

    msg = f"{order_id}|{payment_id}".encode("utf-8")

    expected = hmac.new(
        secret.encode("utf-8"),
        msg,
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(expected, signature)
