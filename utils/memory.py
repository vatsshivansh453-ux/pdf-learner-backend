import sqlite3
import os
import uuid

DB_PATH = "database/chat_history.db"

# Create database folder if it doesn't exist
os.makedirs("database", exist_ok=True)


def get_connection():
    return sqlite3.connect(DB_PATH)


def create_table():

    conn = get_connection()
    cursor = conn.cursor()

    # Chat history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT
        )
    """)

    # Chat sessions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions(
            session_id TEXT PRIMARY KEY,
            user_id TEXT,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id TEXT PRIMARY KEY,
            provider TEXT,
            provider_id TEXT,
            email TEXT,
            name TEXT,
            avatar_url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(provider, provider_id)
        )
    """)

    conn.commit()

    # --- lightweight migration for existing DBs created before user_id existed ---
    cursor.execute("PRAGMA table_info(chat_sessions)")
    cols = [row[1] for row in cursor.fetchall()]
    if "user_id" not in cols:
        cursor.execute("ALTER TABLE chat_sessions ADD COLUMN user_id TEXT")
        conn.commit()

    conn.close()


# Create tables automatically
create_table()


def get_chat_history(session_id):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT role, content
        FROM chat_history
        WHERE session_id=?
        ORDER BY id
        """,
        (session_id,)
    )

    rows = cursor.fetchall()

    conn.close()

    history = []

    for row in rows:
        history.append(
            {
                "role": row[0],
                "content": row[1]
            }
        )

    return history


def add_message(session_id, role, content):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO chat_history(session_id, role, content)
        VALUES (?, ?, ?)
        """,
        (session_id, role, content)
    )

    conn.commit()
    conn.close()


def clear_memory(session_id):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        DELETE FROM chat_history
        WHERE session_id=?
        """,
        (session_id,)
    )

    conn.commit()
    conn.close()
    
def create_session(user_id):

    session_id = str(uuid.uuid4())

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO chat_sessions(session_id, user_id, title)
        VALUES (?, ?, ?)
        """,
        (
            session_id,
            user_id,
            "New Chat"
        )
    )

    conn.commit()
    conn.close()

    return session_id

def get_sessions(user_id):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT session_id, title, created_at
        FROM chat_sessions
        WHERE user_id = ?
        ORDER BY created_at DESC
    """, (user_id,))

    rows = cursor.fetchall()

    conn.close()

    sessions = []

    for row in rows:

        sessions.append(
            {
                "session_id": row[0],
                "title": row[1],
                "created_at": row[2]
            }
        )

    return sessions


def get_session_owner(session_id):
    """Returns the user_id that owns a session, or None if it doesn't exist."""

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT user_id FROM chat_sessions WHERE session_id = ?",
        (session_id,)
    )

    row = cursor.fetchone()
    conn.close()

    return row[0] if row else None


##############################################################
# USERS
##############################################################

def get_or_create_user(provider, provider_id, email, name, avatar_url):
    """
    Looks up a user by (provider, provider_id). Creates one if it
    doesn't exist yet. Returns the full user record as a dict.
    """

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id, provider, provider_id, email, name, avatar_url
        FROM users
        WHERE provider = ? AND provider_id = ?
        """,
        (provider, provider_id)
    )

    row = cursor.fetchone()

    if row:
        user_id = row[0]

        # keep profile info fresh (name/avatar can change on the provider side)
        cursor.execute(
            """
            UPDATE users
            SET email = ?, name = ?, avatar_url = ?
            WHERE id = ?
            """,
            (email, name, avatar_url, user_id)
        )
        conn.commit()
        conn.close()

        return {
            "id": user_id,
            "provider": provider,
            "provider_id": provider_id,
            "email": email,
            "name": name,
            "avatar_url": avatar_url
        }

    user_id = str(uuid.uuid4())

    cursor.execute(
        """
        INSERT INTO users(id, provider, provider_id, email, name, avatar_url)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (user_id, provider, provider_id, email, name, avatar_url)
    )

    conn.commit()
    conn.close()

    return {
        "id": user_id,
        "provider": provider,
        "provider_id": provider_id,
        "email": email,
        "name": name,
        "avatar_url": avatar_url
    }


def get_user_by_id(user_id):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id, provider, provider_id, email, name, avatar_url
        FROM users
        WHERE id = ?
        """,
        (user_id,)
    )

    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "id": row[0],
        "provider": row[1],
        "provider_id": row[2],
        "email": row[3],
        "name": row[4],
        "avatar_url": row[5]
    }


def delete_session(session_id):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
    cursor.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))

    conn.commit()
    conn.close()

def update_session_title(session_id, title):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE chat_sessions
        SET title = ?
        WHERE session_id = ?
        """,
        (
            title,
            session_id
        )
    )

    conn.commit()
    conn.close()