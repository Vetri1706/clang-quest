"""Loopback-only dashboard API and static UI server. Python stdlib; no Docker."""
import argparse
import hmac
import hashlib
from http.server import HTTPServer, BaseHTTPRequestHandler
import ipaddress
import json
import mimetypes
from pathlib import Path
import secrets
import signal
from socketserver import ThreadingMixIn
import threading
import urllib.parse

from .store import Store
from .model_bridge import model_status, experimental_reply

ROOT = Path(__file__).resolve().parents[1]


class Application:
    def __init__(self, database=None, run_function=None, mentor_function=None, health_function=None):
        self.catalog = json.loads((ROOT / 'backend/data/curriculum.json').read_text())
        self.challenges = {c['id']: c for c in self.catalog['challenges']}
        self.store = Store(database or ROOT / 'state/progress.sqlite3')
        self.csrf = secrets.token_urlsafe(32)
        self.compiler_gate = threading.BoundedSemaphore(1)
        self.mentor_gate = threading.BoundedSemaphore(2)
        if run_function is None:
            from .runner import run_cpp, compiler_health
            run_function, health_function = run_cpp, compiler_health
        if mentor_function is None:
            from .mentor import mentor_reply
            mentor_function = mentor_reply
        self.run_function, self.mentor_function = run_function, mentor_function
        self.health = health_function() if health_function else {'available': True, 'compiler': 'test'}

    def public_challenge(self, challenge, full=False):
        if not full:
            keys = ('id', 'title', 'summary', 'difficulty', 'track', 'order', 'xp', 'minutes', 'tags')
            return {k: challenge[k] for k in keys}
        result = {k: v for k, v in challenge.items() if k not in {'reference_solution', 'hints', 'tests'}}
        result['tests'] = [t for t in challenge['tests'] if not t['hidden']]
        result['hidden_count'] = sum(t['hidden'] for t in challenge['tests'])
        return result

    def bootstrap(self):
        return {'csrf': self.csrf, 'tracks': self.catalog['tracks'], 'challenges': [self.public_challenge(c) for c in self.challenges.values()],
                'profile': self.store.profile(), 'compiler': self.health, 'custom_model': model_status()}

    def challenge(self, identifier):
        if not isinstance(identifier, str) or identifier not in self.challenges:
            raise ValueError('Choose a valid mission.')
        return self.challenges[identifier]

    def detail(self, identifier):
        c = self.challenge(identifier)
        return {'challenge': self.public_challenge(c, full=True), **self.store.detail(identifier, c['starter_code'])}

    def source(self, body):
        source = body.get('source')
        if not isinstance(source, str) or len(source.encode('utf-8')) > 32768:
            raise ValueError('Keep your C++ source within 32 KiB.')
        return source

    def run(self, body):
        challenge = self.challenge(body.get('challenge_id'))
        source = self.source(body)
        mode = body.get('mode', 'run')
        if mode not in ('run', 'submit', 'custom'):
            raise ValueError('Choose run, submit, or custom input.')
        if mode == 'custom':
            stdin = body.get('stdin', '')
            if not isinstance(stdin, str) or len(stdin.encode()) > 16384:
                raise ValueError('Keep custom input within 16 KiB.')
            cases = [{'input': stdin, 'output': '', 'hidden': False}]
        else:
            cases = [t for t in challenge['tests'] if mode == 'submit' or not t['hidden']]
        if not self.compiler_gate.acquire(blocking=False):
            raise Busy('The compiler is already running a program. Try again shortly.')
        try:
            raw = self.run_function(source, cases)
        finally:
            self.compiler_gate.release()
        results = []
        for index, row in enumerate(raw.get('results', [])):
            row = dict(row)
            row.update({'index': index + 1, 'test_index': index, 'hidden': bool(cases[index].get('hidden'))})
            if row['hidden']:
                row = {k: row[k] for k in ('index', 'test_index', 'hidden', 'passed', 'status', 'time_ms') if k in row}
            elif mode == 'custom':
                row['passed'] = row.get('status') in ('passed', 'wrong_answer', 'ok')
                if row['passed']:
                    row['status'] = 'executed'
                row.pop('expected', None)
            results.append(row)
        result = {'source_sha256': hashlib.sha256(source.encode()).hexdigest(), 'compile_status': raw['compile_status'], 'diagnostics': raw.get('diagnostics', ''),
                  'duration_ms': raw.get('duration_ms', 0), 'results': results, 'mode': mode,
                  'passed': sum(bool(r.get('passed')) for r in results), 'total': len(cases)}
        return self.store.record_attempt(challenge, source, mode, result)

    def mentor(self, body):
        challenge = self.challenge(body.get('challenge_id'))
        question = body.get('question')
        if not isinstance(question, str) or not question.strip() or len(question.encode()) > 2000:
            raise ValueError('Ask a question of up to 2,000 bytes.')
        source = self.source(body)
        detail = self.store.detail(challenge['id'], challenge['starter_code'])
        mode = body.get('mode', 'grounded')
        if mode not in ('grounded', 'experimental'):
            raise ValueError('Unknown mentor mode.')
        if not self.mentor_gate.acquire(blocking=False):
            raise Busy('Your mentor is answering another question. Try again shortly.')
        try:
            if mode == 'experimental':
                reply = experimental_reply(question)
            else:
                last_run = detail['last_run']
                if last_run is not None:
                    last_run = {**last_run, 'source_changed': last_run.get('source_sha256') != hashlib.sha256(source.encode()).hexdigest()}
                reply = self.mentor_function(question, challenge, source, last_run=last_run,
                                             profile=self.store.profile(), hint_level=detail['hint_level'])
            return self.store.save_messages(challenge['id'], question, reply)
        finally:
            self.mentor_gate.release()

    def settings(self, body):
        if not isinstance(body, dict) or set(body) - {'name', 'difficulty', 'track', 'daily_goal', 'mentor_mode'}:
            raise ValueError('Unknown profile setting.')
        if 'name' in body and (not isinstance(body['name'], str) or not 1 <= len(body['name'].strip()) <= 32):
            raise ValueError('Use a name between 1 and 32 characters.')
        if 'difficulty' in body and body['difficulty'] not in ('beginner', 'intermediate', 'advanced', 'all'):
            raise ValueError('Unknown difficulty.')
        if 'track' in body and body['track'] not in ('all', 'foundations', 'systems', 'games', 'compilers'):
            raise ValueError('Unknown learning path.')
        if 'daily_goal' in body and (type(body['daily_goal']) is not int or not 1 <= body['daily_goal'] <= 10):
            raise ValueError('Choose a daily goal between 1 and 10 missions.')
        if 'mentor_mode' in body and body['mentor_mode'] not in ('grounded', 'experimental'):
            raise ValueError('Unknown mentor mode.')
        return self.store.update_settings(body)


class Busy(Exception):
    pass


class Handler(BaseHTTPRequestHandler):
    server_version = 'Questline'
    sys_version = ''
    protocol_version = 'HTTP/1.0'

    def setup(self):
        self.request.settimeout(5)
        super().setup()

    def log_message(self, format, *args):
        return

    def respond(self, status, body, content_type='application/json; charset=utf-8'):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=True, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Referrer-Policy', 'same-origin')
        self.send_header('Connection', 'close')
        self.end_headers()
        self.wfile.write(body)

    def allowed(self):
        hosts = {'127.0.0.1:' + str(self.server.server_port), 'localhost:' + str(self.server.server_port)}
        if len(self.headers.get_all('Host', [])) != 1 or self.headers.get('Host') not in hosts:
            self.respond(403, {'error': 'Use the local Questline address.'})
            return False
        origin = self.headers.get('Origin')
        origins = {f'http://{host}' for host in hosts} | {'http://127.0.0.1:5173', 'http://localhost:5173'}
        if len(self.headers.get_all('Origin', [])) > 1 or (origin and origin not in origins) or self.headers.get('Sec-Fetch-Site') == 'cross-site':
            self.respond(403, {'error': 'Cross-site requests are not accepted.'})
            return False
        return True

    def do_GET(self):
        if not self.allowed():
            return
        try:
            path = urllib.parse.urlsplit(self.path).path
            if path == '/api/bootstrap':
                return self.respond(200, self.server.app.bootstrap())
            if path == '/api/health':
                return self.respond(200, {'status': 'ok', 'compiler': self.server.app.health})
            if path.startswith('/api/challenges/'):
                return self.respond(200, self.server.app.detail(urllib.parse.unquote(path.rsplit('/', 1)[-1])))
            if path.startswith('/api/'):
                return self.respond(404, {'error': 'Unknown endpoint.'})
            relative = urllib.parse.unquote(path).lstrip('/') or 'index.html'
            static_root = self.server.static_root.resolve()
            target = (static_root / relative).resolve()
            if not target.is_relative_to(static_root):
                return self.respond(404, {'error': 'Not found.'})
            if not target.is_file():
                return self.respond(404, {'error': 'Not found.'})
            if target.stat().st_size > 16 * 1024**2:
                return self.respond(413, {'error': 'Asset too large.'})
            return self.respond(200, target.read_bytes(), mimetypes.guess_type(target.name)[0] or 'application/octet-stream')
        except ValueError as error:
            self.respond(400, {'error': str(error)})
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception:
            self.respond(500, {'error': 'Unable to load this view. Your saved progress is unchanged.'})

    def do_POST(self):
        if not self.allowed():
            return
        supplied = self.headers.get('X-Questline-Token', '')
        if not hmac.compare_digest(supplied.encode(), self.server.app.csrf.encode()):
            return self.respond(403, {'error': 'Refresh the page to reconnect your session.'})
        lengths = self.headers.get_all('Content-Length', [])
        if len(lengths) != 1 or len(lengths[0]) > 6 or not lengths[0].isascii() or not lengths[0].isdigit() or self.headers.get('Transfer-Encoding'):
            return self.respond(400, {'error': 'A single bounded request length is required.'})
        size = int(lengths[0])
        if not 0 < size <= 131072:
            return self.respond(413, {'error': 'Request too large.'})
        if self.headers.get_content_type() != 'application/json':
            return self.respond(415, {'error': 'Send JSON data.'})
        try:
            raw = self.rfile.read(size)
            if len(raw) != size:
                raise ValueError('The request body was incomplete.')
            def unique(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError('Duplicate request field.')
                    result[key] = value
                return result
            body = json.loads(raw, object_pairs_hook=unique)
            if not isinstance(body, dict):
                raise ValueError('The request must contain a JSON object.')
            path = urllib.parse.urlsplit(self.path).path
            if path == '/api/run':
                return self.respond(200, self.server.app.run(body))
            if path == '/api/mentor':
                return self.respond(200, self.server.app.mentor(body))
            if path == '/api/draft':
                challenge = self.server.app.challenge(body.get('challenge_id'))
                timestamp = self.server.app.store.save_draft(challenge['id'], self.server.app.source(body))
                return self.respond(200, {'saved_at': timestamp})
            if path == '/api/settings':
                return self.respond(200, self.server.app.settings(body))
            self.respond(404, {'error': 'Unknown endpoint.'})
        except Busy as error:
            self.respond(429, {'error': str(error)})
        except (ValueError, UnicodeError) as error:
            self.respond(400, {'error': str(error)[:500]})
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception:
            self.respond(500, {'error': 'This operation could not complete. Your previous progress is safe.'})


class Server(ThreadingMixIn, HTTPServer):
    daemon_threads = False
    allow_reuse_address = True
    request_queue_size = 8

    def __init__(self, address, app, static_root=None):
        if not ipaddress.ip_address(address[0]).is_loopback:
            raise ValueError('Questline only binds to a loopback address.')
        self.app = app
        self.static_root = Path(static_root or ROOT / 'dist/client')
        self.slots = threading.BoundedSemaphore(8)
        super().__init__(address, Handler)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            try:
                request.settimeout(.2)
                request.sendall(b'HTTP/1.0 503 Busy\r\nContent-Length: 0\r\nConnection: close\r\n\r\n')
            finally:
                request.close()
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--database', type=Path)
    args = parser.parse_args()
    application = Application(args.database)
    with Server(('127.0.0.1', args.port), application) as server:
        def stop(number, frame):
            threading.Thread(target=server.shutdown, daemon=True).start()
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        print(json.dumps({'status': 'ready', 'url': f'http://127.0.0.1:{server.server_port}', 'compiler': application.health}), flush=True)
        server.serve_forever(poll_interval=.1)


if __name__ == '__main__':
    main()
