import sqlite3

conn = sqlite3.connect("database/chat_history.db")

cursor = conn.cursor()

cursor.execute("SELECT * FROM chat_history")

rows = cursor.fetchall()

for row in rows:
    print(row)

conn.close()