import os
import sqlite3
import pandas as pd

DB_PATH = os.path.join("data", "finsight.db")


def get_conn():
    os.makedirs("data", exist_ok=True)
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_trading_tables():
    con = get_conn()
    cur = con.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS wallets (
        user_id INTEGER PRIMARY KEY,
        balance_inr REAL NOT NULL DEFAULT 100000
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS holdings (
        user_id INTEGER NOT NULL,
        ticker TEXT NOT NULL,
        qty REAL NOT NULL,
        avg_buy_price_inr REAL NOT NULL,
        PRIMARY KEY (user_id, ticker)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        ticker TEXT NOT NULL,
        side TEXT NOT NULL,
        qty REAL NOT NULL,
        price_inr REAL NOT NULL,
        total_inr REAL NOT NULL,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    cur.execute("""
    CREATE TABLE IF NOT EXISTS wallet_topups (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        real_amount REAL NOT NULL,
        virtual_amount REAL NOT NULL,
        status TEXT DEFAULT 'SUCCESS',
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)

    con.commit()
    con.close()


def ensure_wallet(user_id: int):
    init_trading_tables()
    con = get_conn()
    cur = con.cursor()

    cur.execute("SELECT user_id FROM wallets WHERE user_id=?", (user_id,))
    row = cur.fetchone()

    if not row:
        cur.execute(
            "INSERT INTO wallets (user_id, balance_inr) VALUES (?, ?)",
            (user_id, 100000)
        )
        con.commit()

    con.close()


def get_balance(user_id: int) -> float:
    ensure_wallet(user_id)
    con = get_conn()
    cur = con.cursor()
    cur.execute("SELECT balance_inr FROM wallets WHERE user_id=?", (user_id,))
    bal = cur.fetchone()[0]
    con.close()
    return float(bal)


def update_balance(user_id: int, new_balance: float):
    con = get_conn()
    cur = con.cursor()
    cur.execute("UPDATE wallets SET balance_inr=? WHERE user_id=?", (float(new_balance), user_id))
    con.commit()
    con.close()


def get_holdings(user_id: int) -> pd.DataFrame:
    init_trading_tables()
    con = get_conn()
    df = pd.read_sql_query(
        "SELECT ticker, qty, avg_buy_price_inr FROM holdings WHERE user_id=?",
        con,
        params=(user_id,)
    )
    con.close()
    return df


def add_trade(user_id: int, ticker: str, side: str, qty: float, price_inr: float):
    total = float(qty * price_inr)

    con = get_conn()
    cur = con.cursor()
    cur.execute("""
        INSERT INTO trades (user_id, ticker, side, qty, price_inr, total_inr)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (user_id, ticker, side, float(qty), float(price_inr), total))

    con.commit()
    con.close()


def get_trades(user_id: int) -> pd.DataFrame:
    init_trading_tables()
    con = get_conn()
    df = pd.read_sql_query("""
        SELECT timestamp, ticker, side, qty, price_inr, total_inr
        FROM trades
        WHERE user_id=?
        ORDER BY id DESC
        LIMIT 200
    """, con, params=(user_id,))
    con.close()
    return df


def buy_stock(user_id: int, ticker: str, qty: float, price_inr: float):
    ensure_wallet(user_id)

    qty = float(qty)
    price_inr = float(price_inr)

    if qty <= 0 or price_inr <= 0:
        return False, "Invalid quantity or price."

    bal = get_balance(user_id)
    cost = qty * price_inr

    if cost > bal:
        return False, "Not enough wallet balance."

    # Update wallet
    update_balance(user_id, bal - cost)

    # Update holdings
    con = get_conn()
    cur = con.cursor()

    cur.execute("SELECT qty, avg_buy_price_inr FROM holdings WHERE user_id=? AND ticker=?",
                (user_id, ticker))
    row = cur.fetchone()

    if row:
        old_qty, old_avg = row
        new_qty = float(old_qty + qty)
        new_avg = ((old_qty * old_avg) + (qty * price_inr)) / new_qty

        cur.execute("""
            UPDATE holdings
            SET qty=?, avg_buy_price_inr=?
            WHERE user_id=? AND ticker=?
        """, (new_qty, float(new_avg), user_id, ticker))

    else:
        cur.execute("""
            INSERT INTO holdings (user_id, ticker, qty, avg_buy_price_inr)
            VALUES (?, ?, ?, ?)
        """, (user_id, ticker, qty, price_inr))

    con.commit()
    con.close()

    add_trade(user_id, ticker, "BUY", qty, price_inr)
    return True, "✅ Buy executed successfully."


def sell_stock(user_id: int, ticker: str, qty: float, price_inr: float):
    ensure_wallet(user_id)

    qty = float(qty)
    price_inr = float(price_inr)

    if qty <= 0 or price_inr <= 0:
        return False, "Invalid quantity or price."

    con = get_conn()
    cur = con.cursor()

    cur.execute("SELECT qty FROM holdings WHERE user_id=? AND ticker=?",
                (user_id, ticker))
    row = cur.fetchone()

    if not row:
        con.close()
        return False, "You don't own this stock."

    owned_qty = float(row[0])
    if qty > owned_qty:
        con.close()
        return False, "Trying to sell more than holding quantity."

    remaining = owned_qty - qty

    if remaining <= 0:
        cur.execute("DELETE FROM holdings WHERE user_id=? AND ticker=?",
                    (user_id, ticker))
    else:
        cur.execute("""
            UPDATE holdings
            SET qty=?
            WHERE user_id=? AND ticker=?
        """, (remaining, user_id, ticker))

    con.commit()
    con.close()

    # Add money back to wallet
    bal = get_balance(user_id)
    update_balance(user_id, bal + (qty * price_inr))

    add_trade(user_id, ticker, "SELL", qty, price_inr)
    return True, "✅ Sell executed successfully."


# ✅ NEW FEATURE: Reset virtual account
def reset_user_account(user_id: int):
    init_trading_tables()
    con = get_conn()
    cur = con.cursor()

    cur.execute("DELETE FROM holdings WHERE user_id=?", (user_id,))
    cur.execute("DELETE FROM trades WHERE user_id=?", (user_id,))
    cur.execute("UPDATE wallets SET balance_inr=100000 WHERE user_id=?", (user_id,))

    con.commit()
    con.close()
# =========================
# VIRTUAL CASH PURCHASE
# =========================
def add_virtual_cash(user_id: int, real_amount: float, virtual_amount: float):
    ensure_wallet(user_id)

    con = get_conn()
    cur = con.cursor()

    cur.execute("""
        UPDATE wallets
        SET balance_inr = balance_inr + ?
        WHERE user_id = ?
    """, (float(virtual_amount), user_id))

    cur.execute("""
        INSERT INTO wallet_topups (user_id, real_amount, virtual_amount, status)
        VALUES (?, ?, ?, ?)
    """, (user_id, float(real_amount), float(virtual_amount), "SUCCESS"))

    con.commit()
    con.close()


def get_topups(user_id: int):
    init_trading_tables()

    con = get_conn()
    df = pd.read_sql_query("""
        SELECT timestamp, real_amount, virtual_amount, status
        FROM wallet_topups
        WHERE user_id = ?
        ORDER BY id DESC
    """, con, params=(user_id,))

    con.close()
    return df