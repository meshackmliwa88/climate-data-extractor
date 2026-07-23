#!/usr/bin/env python3
from __future__ import annotations

import argparse
import secrets
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from werkzeug.security import generate_password_hash

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / 'storage' / 'db' / 'cde_app.db'


def generate_otp(length: int = 6) -> str:
    alphabet = '0123456789'
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            phone TEXT,
            role TEXT NOT NULL DEFAULT 'user',
            station_name TEXT,
            password_hash TEXT NOT NULL,
            status INTEGER NOT NULL DEFAULT 1,
            force_password_change INTEGER NOT NULL DEFAULT 0,
            otp TEXT,
            otp_generated_at TEXT,
            otp_expires_at TEXT,
            last_login_at TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cols = {row[1] for row in conn.execute('PRAGMA table_info(users)').fetchall()}
    for col, sql in {
        'station_name': 'ALTER TABLE users ADD COLUMN station_name TEXT',
        'otp': 'ALTER TABLE users ADD COLUMN otp TEXT',
        'otp_generated_at': 'ALTER TABLE users ADD COLUMN otp_generated_at TEXT',
        'otp_expires_at': 'ALTER TABLE users ADD COLUMN otp_expires_at TEXT',
    }.items():
        if col not in cols:
            conn.execute(sql)


def main():
    parser = argparse.ArgumentParser(description='Create or reset a CDE admin account')
    parser.add_argument('--email', default='admin@cde.local')
    parser.add_argument('--name', default='System Administrator')
    parser.add_argument('--password', default=None, help='New password. If omitted, a random password is generated.')
    parser.add_argument('--otp', default=None, help='Optional fixed OTP. If omitted, one is generated.')
    args = parser.parse_args()
    password = args.password or secrets.token_urlsafe(18)
    otp = args.otp or generate_otp()
    now = datetime.now().isoformat(timespec='seconds')
    expires = (datetime.now() + timedelta(days=30)).isoformat(timespec='seconds')
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    ensure_schema(conn)
    existing = cur.execute('SELECT id FROM users WHERE lower(email)=lower(?)', (args.email,)).fetchone()
    if existing:
        cur.execute('''
            UPDATE users
            SET full_name=?, role='admin', password_hash=?, status=1, force_password_change=1,
                otp=?, otp_generated_at=?, otp_expires_at=?, updated_at=CURRENT_TIMESTAMP
            WHERE id=?
        ''', (args.name, generate_password_hash(password), otp, now, expires, existing[0]))
    else:
        cur.execute('''
            INSERT INTO users (full_name,email,role,password_hash,status,force_password_change,otp,otp_generated_at,otp_expires_at)
            VALUES (?,?,?,?,1,1,?,?,?)
        ''', (args.name, args.email, 'admin', generate_password_hash(password), otp, now, expires))
    conn.commit()
    conn.close()
    print(f'Admin account ready: {args.email}')
    print(f'Password: {password}')
    print(f'OTP: {otp}')


if __name__ == '__main__':
    main()
