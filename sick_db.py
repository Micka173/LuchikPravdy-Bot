import sqlite3
from datetime import datetime, timedelta

DB_PATH = "children.db"


def init_sick_table():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS symptoms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            symptom TEXT,
            value TEXT,
            is_temperature INTEGER DEFAULT 0,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()


def add_symptom(user_id, symptom, value="", is_temperature=False):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO symptoms (user_id, symptom, value, is_temperature, timestamp) VALUES (?, ?, ?, ?, ?)",
        (user_id, symptom, value, 1 if is_temperature else 0, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def get_symptoms_last_hours(user_id, hours=24):
    conn = sqlite3.connect(DB_PATH)
    since = (datetime.now() - timedelta(hours=hours)).isoformat()
    cur = conn.execute(
        "SELECT symptom, value, is_temperature, timestamp FROM symptoms "
        "WHERE user_id = ? AND timestamp >= ? ORDER BY timestamp",
        (user_id, since)
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def get_symptoms_older(user_id, hours=24):
    """Проверяет, были ли записи о температуре раньше, чем N часов назад."""
    conn = sqlite3.connect(DB_PATH)
    since = (datetime.now() - timedelta(hours=hours)).isoformat()
    cur = conn.execute(
        "SELECT COUNT(*) FROM symptoms WHERE user_id = ? AND is_temperature = 1 AND timestamp < ?",
        (user_id, since)
    )
    row = cur.fetchone()
    conn.close()
    return row[0] if row else 0