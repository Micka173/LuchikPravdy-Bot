import sqlite3
from datetime import datetime

DB_PATH = "children.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS children (
            user_id INTEGER PRIMARY KEY,
            age_months INTEGER,
            gender TEXT,
            updated_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_child(user_id, age_months=None, gender=None):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT user_id FROM children WHERE user_id = ?", (user_id,))
    row = cur.fetchone()
    now = datetime.now().isoformat()
    if row:
        updates, values = [], []
        if age_months is not None:
            updates.append("age_months = ?"); values.append(age_months)
        if gender is not None:
            updates.append("gender = ?"); values.append(gender)
        updates.append("updated_at = ?"); values.append(now)
        values.append(user_id)
        cur.execute(f"UPDATE children SET {', '.join(updates)} WHERE user_id = ?", values)
    else:
        cur.execute(
            "INSERT INTO children (user_id, age_months, gender, updated_at) VALUES (?, ?, ?, ?)",
            (user_id, age_months, gender, now)
        )
    conn.commit()
    conn.close()


def get_child(user_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT age_months, gender FROM children WHERE user_id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    if row:
        return {"age_months": row[0], "gender": row[1]}
    return None