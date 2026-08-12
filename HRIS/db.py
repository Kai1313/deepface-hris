"""SQLite storage for employees (with face embeddings) and punches. Stdlib only."""
import json
import sqlite3
import datetime

DB = "hris.db"


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE IF NOT EXISTS employees("
        " id INTEGER PRIMARY KEY,"
        " name TEXT UNIQUE NOT NULL,"
        " embeddings TEXT NOT NULL)"  # JSON list of embedding lists
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS punches("
        " id INTEGER PRIMARY KEY,"
        " employee_id INTEGER NOT NULL REFERENCES employees(id),"
        " punch_type TEXT NOT NULL CHECK(punch_type IN ('IN','OUT')),"
        " at TEXT NOT NULL)"
    )
    return conn


def add_embedding(name: str, embedding: list) -> tuple:
    """Append one embedding to an employee (creates them if new). Returns (id, photo_count)."""
    conn = connect()
    row = conn.execute("SELECT id, embeddings FROM employees WHERE name=?", (name,)).fetchone()
    if row is None:
        cur = conn.execute("INSERT INTO employees(name, embeddings) VALUES (?, ?)", (name, json.dumps([embedding])))
        eid = cur.lastrowid
        count = 1
    else:
        eid = row["id"]
        emb = json.loads(row["embeddings"]) + [embedding]
        count = len(emb)
        conn.execute("UPDATE employees SET embeddings=? WHERE id=?", (json.dumps(emb), eid))
    conn.commit()
    conn.close()
    return eid, count


def all_embeddings() -> list:
    """[(id, name, [embedding, ...]), ...] for every employee."""
    conn = connect()
    rows = conn.execute("SELECT id, name, embeddings FROM employees").fetchall()
    conn.close()
    return [(r["id"], r["name"], json.loads(r["embeddings"])) for r in rows]


def punch(employee_id: int, punch_type: str) -> str:
    conn = connect()
    at = datetime.datetime.now().isoformat(timespec="seconds")
    conn.execute(
        "INSERT INTO punches(employee_id, punch_type, at) VALUES (?,?,?)",
        (employee_id, punch_type, at),
    )
    conn.commit()
    conn.close()
    return at


def list_employees() -> list:
    conn = connect()
    rows = conn.execute("SELECT id, name, embeddings FROM employees").fetchall()
    conn.close()
    return [{"id": r["id"], "name": r["name"], "photos": len(json.loads(r["embeddings"]))} for r in rows]


def list_punches(limit: int = 20) -> list:
    conn = connect()
    rows = conn.execute(
        "SELECT p.id, e.name, p.punch_type, p.at FROM punches p"
        " JOIN employees e ON e.id = p.employee_id"
        " ORDER BY p.id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
