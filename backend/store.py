"""Transactional, single-user learning progress on local SQLite."""
from contextlib import contextmanager
from datetime import datetime, timedelta
import json
from pathlib import Path
import sqlite3


def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS drafts(challenge_id TEXT PRIMARY KEY, source TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS completions(challenge_id TEXT PRIMARY KEY, xp INTEGER NOT NULL CHECK(xp>0), completed_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS attempts(id INTEGER PRIMARY KEY, challenge_id TEXT NOT NULL, mode TEXT NOT NULL, passed INTEGER NOT NULL, total INTEGER NOT NULL, result_json TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS idx_attempts_challenge ON attempts(challenge_id,id DESC);
                CREATE TABLE IF NOT EXISTS activity(day TEXT PRIMARY KEY, attempts INTEGER NOT NULL DEFAULT 0, completed INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY, challenge_id TEXT NOT NULL, role TEXT NOT NULL, text TEXT NOT NULL, metadata TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS idx_messages_challenge ON messages(challenge_id,id DESC);
                CREATE TABLE IF NOT EXISTS hints(challenge_id TEXT PRIMARY KEY, level INTEGER NOT NULL DEFAULT 0);
            ''')
            for key, value in {'name': 'Developer', 'difficulty': 'beginner', 'track': 'all', 'daily_goal': 3, 'mentor_mode': 'grounded'}.items():
                db.execute('INSERT OR IGNORE INTO settings VALUES (?,?)', (key, json.dumps(value)))

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('PRAGMA synchronous=FULL')
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def profile(self):
        with self.connection() as db:
            settings = {r['key']: json.loads(r['value']) for r in db.execute('SELECT * FROM settings')}
            completed = {r['challenge_id']: dict(r) for r in db.execute('SELECT * FROM completions')}
            activity = {r['day']: dict(r) for r in db.execute('SELECT * FROM activity')}
            latest = db.execute('SELECT challenge_id FROM drafts ORDER BY updated_at DESC LIMIT 1').fetchone()
        today = datetime.now().astimezone().date()
        cursor = today if str(today) in activity else today - timedelta(days=1)
        streak = 0
        while str(cursor) in activity:
            streak += 1
            cursor -= timedelta(days=1)
        xp = sum(r['xp'] for r in completed.values())
        week = []
        for offset in range(6, -1, -1):
            day = today - timedelta(days=offset)
            week.append({'date': str(day), 'label': day.strftime('%a'), **activity.get(str(day), {'attempts': 0, 'completed': 0})})
        return {**settings, 'completed': completed, 'xp': xp, 'level': xp // 300 + 1,
                'level_xp': xp % 300, 'level_target': 300, 'streak': streak,
                'today_completed': activity.get(str(today), {}).get('completed', 0),
                'total_attempts': sum(r['attempts'] for r in activity.values()), 'activity': week,
                'last_challenge': settings.get('last_challenge', latest['challenge_id'] if latest else 'cafe-receipt')}

    def update_settings(self, changes):
        with self.connection() as db:
            for key, value in changes.items():
                db.execute('INSERT INTO settings VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value', (key, json.dumps(value)))
        return self.profile()

    def save_draft(self, challenge_id, source):
        timestamp = now()
        with self.connection() as db:
            db.execute('INSERT INTO drafts VALUES (?,?,?) ON CONFLICT(challenge_id) DO UPDATE SET source=excluded.source,updated_at=excluded.updated_at', (challenge_id, source, timestamp))
            db.execute('INSERT INTO settings VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value', ('last_challenge', json.dumps(challenge_id)))
        return timestamp

    def detail(self, challenge_id, starter):
        with self.connection() as db:
            draft = db.execute('SELECT * FROM drafts WHERE challenge_id=?', (challenge_id,)).fetchone()
            rows = list(db.execute('SELECT * FROM messages WHERE challenge_id=? ORDER BY id DESC LIMIT 40', (challenge_id,)))
            messages = [{'id': r['id'], 'role': r['role'], 'text': r['text'], **json.loads(r['metadata'])} for r in reversed(rows)]
            hint = db.execute('SELECT level FROM hints WHERE challenge_id=?', (challenge_id,)).fetchone()
            attempt = db.execute('SELECT result_json FROM attempts WHERE challenge_id=? ORDER BY id DESC LIMIT 1', (challenge_id,)).fetchone()
        return {'source': draft['source'] if draft else starter, 'draft_updated_at': draft['updated_at'] if draft else None,
                'messages': messages, 'hint_level': hint['level'] if hint else 0,
                'last_run': json.loads(attempt['result_json']) if attempt else None}

    def record_attempt(self, challenge, source, mode, result):
        timestamp = now()
        awarded = 0
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('INSERT INTO drafts VALUES (?,?,?) ON CONFLICT(challenge_id) DO UPDATE SET source=excluded.source,updated_at=excluded.updated_at', (challenge['id'], source, timestamp))
            db.execute('INSERT INTO settings VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value', ('last_challenge', json.dumps(challenge['id'])))
            if mode == 'submit' and result['compile_status'] == 'ok' and result['passed'] == result['total'] and result['total'] > 0:
                inserted = db.execute('INSERT OR IGNORE INTO completions VALUES (?,?,?)', (challenge['id'], challenge['xp'], timestamp))
                awarded = challenge['xp'] if inserted.rowcount == 1 else 0
            result = {**result, 'xp_awarded': awarded}
            db.execute('INSERT INTO attempts(challenge_id,mode,passed,total,result_json,created_at) VALUES (?,?,?,?,?,?)',
                       (challenge['id'], mode, result['passed'], result['total'], json.dumps(result), timestamp))
            db.execute('INSERT INTO activity VALUES (?,1,?) ON CONFLICT(day) DO UPDATE SET attempts=attempts+1,completed=completed+excluded.completed', (timestamp[:10], int(awarded > 0)))
            db.execute('DELETE FROM attempts WHERE id NOT IN (SELECT id FROM attempts ORDER BY id DESC LIMIT 500)')
        return {**result, 'profile': self.profile()}

    def save_messages(self, challenge_id, question, reply):
        metadata = {k: v for k, v in reply.items() if k != 'text'}
        with self.connection() as db:
            db.execute('INSERT INTO messages(challenge_id,role,text,metadata,created_at) VALUES (?,?,?,?,?)', (challenge_id, 'user', question, '{}', now()))
            result = db.execute('INSERT INTO messages(challenge_id,role,text,metadata,created_at) VALUES (?,?,?,?,?)', (challenge_id, 'assistant', reply['text'], json.dumps(metadata), now()))
            db.execute('INSERT INTO hints VALUES (?,?) ON CONFLICT(challenge_id) DO UPDATE SET level=MAX(level,excluded.level)', (challenge_id, reply.get('next_hint_level', 0)))
            db.execute('DELETE FROM messages WHERE challenge_id=? AND id NOT IN (SELECT id FROM messages WHERE challenge_id=? ORDER BY id DESC LIMIT 40)', (challenge_id, challenge_id))
            return {**reply, 'role': 'assistant', 'id': result.lastrowid}
