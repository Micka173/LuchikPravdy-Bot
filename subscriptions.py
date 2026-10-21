import sqlite3

DB_PATH = "children.db"


def init_subs_table():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS subscribers (
            user_id INTEGER PRIMARY KEY,
            subscribed_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def subscribe(user_id):
    from datetime import datetime
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "INSERT OR IGNORE INTO subscribers (user_id, subscribed_at) VALUES (?, ?)",
            (user_id, datetime.now().isoformat())
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"⚠️ Ошибка подписки: {e}")
        return False
    finally:
        conn.close()


def unsubscribe(user_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM subscribers WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()


def is_subscribed(user_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT user_id FROM subscribers WHERE user_id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    return row is not None


def get_all_subscribers():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT user_id FROM subscribers")
    rows = cur.fetchall()
    conn.close()
    return [r[0] for r in rows]