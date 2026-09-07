"""Independent local backend regressions; all mutable fixtures live beside this file.

Run: python3 -B -m unittest tests.test_backend -v
Override QUESTLINE_ROOT to test a different checkout. The compiler is injected;
no C++ process, model checkpoint, third-party package, or network service is used.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import http.client
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
QUESTLINE_ROOT = Path(os.environ.get('QUESTLINE_ROOT', str(HERE.parent))).resolve()
sys.path.insert(0, str(QUESTLINE_ROOT))
from backend.server import Application, Busy, Server
from backend.store import Store
from backend.mentor import mentor_reply

SOURCE = '#include <iostream>\nint main() { std::cout << 0; }\n'
SECRET_INPUT = 'PRIVATE_INPUT_57fa83\n'
SECRET_EXPECTED = 'PRIVATE_EXPECTED_93d642\n'
SECRET_EXTRA = 'PRIVATE_DEBUG_16cdd1'
REFERENCE = 'int main() { /* PRIVATE_REFERENCE_d8f092 */ return 0; }'
CHALLENGE = {
    'id': 'test-mission', 'title': 'Test mission', 'summary': 'Multiply the inputs.',
    'story': 'Compute a receipt.', 'difficulty': 'beginner', 'track': 'foundations',
    'order': 1, 'xp': 50, 'minutes': 5, 'tags': ['arithmetic', 'streams'],
    'concepts': [{'title': 'Multiplication', 'body': 'Multiply the two input values.'}],
    'objectives': ['Read two values', 'Print their product'], 'prompt': 'Print their product.',
    'input_format': 'Two integers.', 'output_format': 'One integer.', 'constraints': ['0 <= n <= 100'],
    'starter_code': SOURCE, 'reference_solution': REFERENCE,
    'hints': ['PRIVATE_HINT_1', 'PRIVATE_HINT_2', 'PRIVATE_HINT_3'],
    'tests': [
        {'input': '3 2\n', 'output': '6\n', 'hidden': False},
        {'input': '4 5\n', 'output': '20\n', 'hidden': False},
        {'input': SECRET_INPUT, 'output': SECRET_EXPECTED, 'hidden': True},
    ],
}


def fake_runner(source, cases):
    return {'compile_status': 'ok', 'diagnostics': '', 'duration_ms': 3,
            'results': [{'status': 'passed', 'passed': True, 'input': c['input'],
                         'expected': c['output'], 'actual': c['output'], 'stdout': c['output'],
                         'stderr': SECRET_EXTRA if c.get('hidden') else '',
                         'extra_debug': SECRET_EXTRA if c.get('hidden') else 'public',
                         'time_ms': 1} for c in cases]}


def fake_mentor(question, challenge, source, **context):
    return {'text': 'Read one value, then compile that step.', 'kind': 'hint', 'sources': [],
            'suggestions': ['Explain the input format'],
            'next_hint_level': min(3, context['hint_level'] + 1),
            'model': {'mode': 'grounded', 'label': 'Local study mentor'}}


class Fixtures(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='backend-test-', dir=HERE)
        self.directory = Path(self.temporary.name)
        self.database = self.directory / 'progress.sqlite3'
        self.app = self.application()

    def tearDown(self):
        self.temporary.cleanup()

    def application(self, runner=fake_runner, mentor=fake_mentor):
        app = Application(self.database, runner, mentor, lambda: {'available': True, 'compiler': 'injected'})
        challenge = copy.deepcopy(CHALLENGE)
        app.challenges = {challenge['id']: challenge}
        app.catalog = {'tracks': [], 'challenges': [challenge]}
        return app

    def run_body(self, **changes):
        return {'challenge_id': CHALLENGE['id'], 'source': SOURCE, 'mode': 'submit', **changes}

    @contextmanager
    def http_server(self, app=None):
        static = self.directory / 'static'
        static.mkdir(exist_ok=True)
        (static / 'index.html').write_text('<!doctype html><title>Fixture</title>')
        server = Server(('127.0.0.1', 0), app or self.app, static)
        worker = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
        worker.start()
        try:
            yield server
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)
            self.assertFalse(worker.is_alive(), 'HTTP serving thread did not stop')

    def request(self, server, method='GET', path='/api/bootstrap', body=None, headers=None, token=True):
        connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=15)
        data = json.dumps(body).encode() if body is not None and not isinstance(body, bytes) else body
        request_headers = {}
        if method == 'POST':
            request_headers['Content-Type'] = 'application/json'
            if token:
                request_headers['X-Questline-Token'] = server.app.csrf
        request_headers.update(headers or {})
        try:
            connection.request(method, path, body=data, headers=request_headers)
            response = connection.getresponse()
            payload = response.read()
            response_headers = dict(response.getheaders())
            if 'application/json' in response_headers.get('Content-Type', ''):
                payload = json.loads(payload)
            return response.status, payload, response_headers
        finally:
            connection.close()

    def raw_request(self, server, lines, body=b''):
        with socket.create_connection(('127.0.0.1', server.server_port), timeout=3) as connection:
            connection.sendall(('\r\n'.join(lines) + '\r\n\r\n').encode('ascii') + body)
            connection.shutdown(socket.SHUT_WR)
            response = http.client.HTTPResponse(connection)
            response.begin()
            payload = response.read()
            return response.status, payload

    def raw_headers(self, server, **changes):
        values = {'Host': f'127.0.0.1:{server.server_port}', 'X-Questline-Token': self.app.csrf,
                  'Content-Type': 'application/json', 'Content-Length': '2', **changes}
        return ['POST /api/settings HTTP/1.1'] + [f'{k}: {v}' for k, v in values.items() if v is not None]


class PrivacyAndPersistenceTests(Fixtures):
    def test_catalog_and_detail_do_not_expose_answers_or_hidden_cases(self):
        summary = self.app.bootstrap()['challenges'][0]
        self.assertNotIn('tests', summary)
        detail = self.app.detail(CHALLENGE['id'])['challenge']
        self.assertEqual(detail['hidden_count'], 1)
        self.assertEqual(len(detail['tests']), 2)
        for exposed in (summary, detail):
            self.assertNotIn('hints', exposed)
            self.assertNotIn('reference_solution', exposed)
            serialized = json.dumps(exposed)
            for secret in (SECRET_INPUT.strip(), SECRET_EXPECTED.strip(), 'PRIVATE_REFERENCE', 'PRIVATE_HINT'):
                self.assertNotIn(secret, serialized)

    def test_submit_redacts_hidden_result_before_persistence_and_mentor(self):
        captured = {}
        def mentor(question, challenge, source, **context):
            captured.update(context)
            return fake_mentor(question, challenge, source, **context)
        self.app = self.application(mentor=mentor)
        result = self.app.run(self.run_body())
        private = result['results'][2]
        self.assertTrue(private['hidden'])
        self.assertTrue(set(private) <= {'index', 'test_index', 'hidden', 'passed', 'status', 'time_ms'})
        public = result['results'][0]
        self.assertEqual(public['input'], '3 2\n')
        stored = self.app.detail(CHALLENGE['id'])['last_run']
        self.app.mentor({**self.run_body(), 'mode': 'grounded', 'question': 'Why did the test fail?'})
        for exposed in (result, stored, captured['last_run']):
            serialized = json.dumps(exposed)
            for secret in (SECRET_INPUT.strip(), SECRET_EXPECTED.strip(), SECRET_EXTRA):
                self.assertNotIn(secret, serialized)

    def test_run_uses_only_visible_cases_custom_uses_only_supplied_input(self):
        captured = []
        def runner(source, cases):
            captured.append(copy.deepcopy(cases))
            return fake_runner(source, cases)
        self.app = self.application(runner=runner)
        public = self.app.run(self.run_body(mode='run'))
        custom = self.app.run(self.run_body(mode='custom', stdin='provided input\n'))
        self.assertEqual(len(captured[0]), 2)
        self.assertFalse(any(c['hidden'] for c in captured[0]))
        self.assertEqual(captured[1], [{'input': 'provided input\n', 'output': '', 'hidden': False}])
        self.assertEqual(custom['results'][0]['status'], 'executed')
        self.assertNotIn('expected', custom['results'][0])
        self.assertEqual(public['xp_awarded'], 0)
        self.assertEqual(custom['xp_awarded'], 0)

    def test_profile_draft_messages_and_hint_level_persist_after_reopen(self):
        changes = {'name': 'Ada', 'difficulty': 'advanced', 'track': 'games', 'daily_goal': 7,
                   'mentor_mode': 'grounded'}
        self.app.settings(changes)
        self.app.store.save_draft(CHALLENGE['id'], 'int main() { return 7; }')
        reply = self.app.mentor({**self.run_body(), 'mode': 'grounded', 'question': 'Give me a hint'})
        reopened = self.application()
        for key, value in changes.items():
            self.assertEqual(reopened.store.profile()[key], value)
        detail = reopened.detail(CHALLENGE['id'])
        self.assertEqual(detail['source'], 'int main() { return 7; }')
        self.assertEqual(detail['hint_level'], 1)
        self.assertEqual([m['role'] for m in detail['messages']], ['user', 'assistant'])
        self.assertEqual(detail['messages'][1]['id'], reply['id'])
        self.assertNotEqual(self.app.csrf, reopened.csrf)

    def test_last_challenge_follows_most_recent_save_with_equal_timestamps(self):
        with patch('backend.store.now', return_value='2026-09-07T12:00:00+05:30'):
            self.app.store.save_draft('first-mission', 'first')
            self.app.store.save_draft('second-mission', 'second')
            self.assertEqual(self.app.store.profile()['last_challenge'], 'second-mission')
            self.app.store.save_draft('first-mission', 'first revision')
            self.assertEqual(self.app.store.profile()['last_challenge'], 'first-mission')

    def test_mentor_respects_explicit_difficulty_and_default_track_fallback(self):
        self.app = self.application(mentor=mentor_reply)
        self.app.challenges[CHALLENGE['id']]['track'] = 'games'
        self.app.settings({'difficulty': 'advanced', 'track': 'all'})
        reply = self.app.mentor({**self.run_body(), 'mode': 'grounded', 'question': 'Where should I start?'})
        self.assertIn('correctness invariant', reply['text'])
        self.assertIn('game track', reply['text'])

    def test_first_successful_submit_awards_xp_once(self):
        public = self.app.run(self.run_body(mode='run'))
        first = self.app.run(self.run_body())
        repeat = self.app.run(self.run_body())
        reopened = self.application()
        resumed = reopened.run(self.run_body())
        self.assertEqual([r['xp_awarded'] for r in (public, first, repeat, resumed)], [0, 50, 0, 0])
        profile = reopened.store.profile()
        self.assertEqual(profile['xp'], 50)
        self.assertEqual(profile['today_completed'], 1)
        self.assertEqual(profile['total_attempts'], 4)
        self.assertEqual(len(profile['completed']), 1)

    def test_failed_and_empty_submissions_do_not_award_xp(self):
        def wrong(source, cases):
            result = fake_runner(source, cases)
            result['results'][0].update(status='wrong_answer', passed=False, actual='0\n')
            return result
        self.app = self.application(runner=wrong)
        self.assertEqual(self.app.run(self.run_body())['xp_awarded'], 0)
        self.app.run_function = lambda source, cases: {'compile_status': 'compile_error', 'diagnostics': 'failure', 'results': []}
        self.assertEqual(self.app.run(self.run_body())['xp_awarded'], 0)
        self.assertEqual(self.app.store.profile()['xp'], 0)

    def test_concurrent_submissions_award_one_completion_transactionally(self):
        workers = 8
        barrier = threading.Barrier(workers)
        def runner(source, cases):
            barrier.wait(timeout=3)
            return fake_runner(source, cases)
        # Separate app instances model a process restart overlapping another client.
        # Each has its own CPU gate, while all share the same SQLite database.
        apps = [self.application(runner=runner) for _ in range(workers)]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(lambda app: app.run(self.run_body()), apps))
        self.assertEqual(sum(r['xp_awarded'] for r in results), 50)
        self.assertEqual(sum(r['xp_awarded'] > 0 for r in results), 1)
        profile = self.app.store.profile()
        self.assertEqual((profile['xp'], profile['today_completed'], profile['total_attempts']), (50, 1, workers))
        with self.app.store.connection() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM completions').fetchone()[0], 1)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0], workers)

    def test_failure_rolls_back_draft_completion_attempt_and_activity(self):
        self.app.store.save_draft(CHALLENGE['id'], 'previous draft')
        invalid_result = {'compile_status': 'ok', 'passed': 3, 'total': 3, 'not_json': object()}
        with self.assertRaises(TypeError):
            self.app.store.record_attempt(CHALLENGE, 'new draft', 'submit', invalid_result)
        profile = self.app.store.profile()
        self.assertEqual((profile['xp'], profile['total_attempts'], profile['today_completed']), (0, 0, 0))
        detail = self.app.detail(CHALLENGE['id'])
        self.assertEqual(detail['source'], 'previous draft')
        self.assertIsNone(detail['last_run'])

    def test_hint_level_monotonic_and_message_retention_bounded(self):
        for index in range(23):
            self.app.store.save_messages(CHALLENGE['id'], f'question {index}',
                {'text': f'answer {index}', 'next_hint_level': 3 if index == 0 else 1})
        detail = self.app.detail(CHALLENGE['id'])
        self.assertEqual(detail['hint_level'], 3)
        self.assertEqual(len(detail['messages']), 40)
        self.assertEqual(detail['messages'][0]['text'], 'question 3')
        self.assertEqual(detail['messages'][-1]['text'], 'answer 22')

    def test_visible_second_test_failure_reaches_actual_mentor(self):
        def runner(source, cases):
            result = fake_runner(source, cases)
            result['results'][1].update(status='wrong_answer', passed=False, actual='19\n', stdout='19\n')
            return result
        self.app = self.application(runner=runner, mentor=mentor_reply)
        self.app.run(self.run_body(mode='submit'))
        reply = self.app.mentor({**self.run_body(), 'mode': 'grounded', 'question': 'Why did my test fail?'})
        self.assertEqual(reply['kind'], 'test_feedback')
        self.assertIn('4 5', reply['text'])
        self.assertIn('20', reply['text'])
        self.assertIn('19', reply['text'])
        self.assertNotIn('A hidden test failed', reply['text'])


class HTTPTests(Fixtures):
    def test_bootstrap_security_headers_and_successful_settings_post(self):
        with self.http_server() as server:
            status, body, headers = self.request(server)
            self.assertEqual(status, 200)
            self.assertEqual(body['csrf'], self.app.csrf)
            self.assertEqual(headers['X-Content-Type-Options'], 'nosniff')
            self.assertEqual(headers['X-Frame-Options'], 'DENY')
            self.assertEqual(headers['Cache-Control'], 'no-store')
            self.assertEqual(headers['Connection'], 'close')
            status, body, _ = self.request(server, 'POST', '/api/settings', {'name': 'Ada'})
            self.assertEqual(status, 200)
            self.assertEqual(body['name'], 'Ada')

    def test_missing_wrong_and_stale_csrf_tokens_rejected_without_mutation(self):
        with self.http_server() as server:
            for token in (None, '', 'bad-token', self.application().csrf):
                with self.subTest(token=token):
                    headers = {} if token is None else {'X-Questline-Token': token}
                    status, _, _ = self.request(server, 'POST', '/api/settings', {'name': 'Changed'}, headers, token=False)
                    self.assertEqual(status, 403)
            self.assertEqual(self.app.store.profile()['name'], 'Developer')

    def test_host_origin_fetch_site_enforced(self):
        with self.http_server() as server:
            cases = [({'Host': 'attacker.example'}, 403),
                     ({'Host': f'127.0.0.1.attacker.example:{server.server_port}'}, 403),
                     ({'Origin': 'https://attacker.example'}, 403),
                     ({'Origin': 'null'}, 403), ({'Sec-Fetch-Site': 'cross-site'}, 403),
                     ({'Host': f'localhost:{server.server_port}'}, 200),
                     ({'Origin': f'http://localhost:{server.server_port}'}, 200),
                     ({'Origin': 'http://127.0.0.1:5173'}, 200)]
            for headers, expected in cases:
                with self.subTest(headers=headers):
                    status, _, _ = self.request(server, headers=headers)
                    self.assertEqual(status, expected)
            for headers in ({'Host': 'attacker.example'}, {'Origin': 'https://attacker.example'}, {'Sec-Fetch-Site': 'cross-site'}):
                status, _, _ = self.request(server, 'POST', '/api/settings', {'name': 'Changed'}, headers)
                self.assertEqual(status, 403)
            self.assertEqual(self.app.store.profile()['name'], 'Developer')

    def test_missing_and_ambiguous_host_origin_headers_rejected(self):
        with self.http_server() as server:
            host = f'127.0.0.1:{server.server_port}'
            cases = [[], [f'Host: {host}', 'Host: attacker.example'],
                     [f'Host: {host}', f'Origin: http://{host}', 'Origin: https://attacker.example']]
            for headers in cases:
                with self.subTest(headers=headers):
                    status, _ = self.raw_request(server, ['GET /api/bootstrap HTTP/1.1', *headers])
                    self.assertIn(status, (400, 403))

    def test_invalid_settings_reject_all_fields_atomically(self):
        cases = [{'name': ''}, {'name': ' '}, {'name': 'x' * 33}, {'name': 1},
                 {'difficulty': 'expert'}, {'track': 'private'}, {'mentor_mode': 'magic'},
                 {'daily_goal': True}, {'daily_goal': 0}, {'daily_goal': 11}, {'daily_goal': 3.0},
                 {'daily_goal': '3'}, {'unknown': True}, {'name': 'Changed', 'daily_goal': 0}]
        baseline = self.app.store.profile()
        with self.http_server() as server:
            for body in cases:
                with self.subTest(body=body):
                    status, _, _ = self.request(server, 'POST', '/api/settings', body)
                    self.assertEqual(status, 400)
                    self.assertEqual(self.app.store.profile(), baseline)

    def test_bad_json_shapes_duplicate_fields_and_utf8_rejected(self):
        with self.http_server() as server:
            for raw in (b'{', b'[]', b'null', b'42', b'"text"', b'{"name":"one","name":"two"}', b'{"name":"\xff"}'):
                with self.subTest(raw=raw):
                    status, _, _ = self.request(server, 'POST', '/api/settings', raw)
                    self.assertEqual(status, 400)
            self.assertEqual(self.app.store.profile()['name'], 'Developer')

    def test_invalid_challenge_identifier_types_return_client_error(self):
        with self.http_server() as server:
            for path in ('/api/draft', '/api/run', '/api/mentor'):
                for identifier in (None, 'unknown', [], {}):
                    with self.subTest(path=path, identifier=identifier):
                        body = self.run_body(challenge_id=identifier, question='Help', mode='grounded' if path.endswith('mentor') else 'submit')
                        status, _, _ = self.request(server, 'POST', path, body)
                        self.assertEqual(status, 400)
            self.assertEqual(self.app.store.profile()['total_attempts'], 0)

    def test_content_length_transfer_encoding_and_media_type_limits(self):
        with self.http_server() as server:
            cases = [({'Content-Length': None}, 400), ({'Content-Length': '-1'}, 400),
                     ({'Content-Length': '+2'}, 400), ({'Content-Length': '2, 2'}, 400),
                     ({'Content-Length': '0'}, 413), ({'Content-Length': '131073'}, 413),
                     ({'Content-Length': '9' * 5000}, 400),
                     ({'Transfer-Encoding': 'chunked'}, 400), ({'Content-Type': 'text/plain'}, 415)]
            for changes, expected in cases:
                with self.subTest(changes={key: (value[:80] + "...") if isinstance(value, str) and len(value) > 80 else value for key, value in changes.items()}):
                    status, _ = self.raw_request(server, self.raw_headers(server, **changes), b'{}')
                    self.assertEqual(status, expected)
            duplicate = self.raw_headers(server) + ['Content-Length: 2']
            self.assertEqual(self.raw_request(server, duplicate, b'{}')[0], 400)
            truncated = self.raw_headers(server, **{'Content-Length': '50'})
            self.assertEqual(self.raw_request(server, truncated, b'{}')[0], 400)

    def test_source_stdin_question_utf8_byte_limits(self):
        with self.http_server() as server:
            valid_source = 'x' * 32768
            status, _, _ = self.request(server, 'POST', '/api/draft', self.run_body(source=valid_source))
            self.assertEqual(status, 200)
            for source in ('x' * 32769, '\u00e9' * 16385, 3, None):
                status, _, _ = self.request(server, 'POST', '/api/draft', self.run_body(source=source))
                self.assertEqual(status, 400)
            self.assertEqual(self.app.detail(CHALLENGE['id'])['source'], valid_source)
            for stdin, expected in (('x' * 16384, 200), ('x' * 16385, 400), ('\u00e9' * 8193, 400), ([], 400)):
                status, _, _ = self.request(server, 'POST', '/api/run', self.run_body(mode='custom', stdin=stdin))
                self.assertEqual(status, expected)
            for question, expected in (('x' * 2000, 200), ('x' * 2001, 400), ('\u00e9' * 1001, 400), (' ', 400), (None, 400)):
                status, _, _ = self.request(server, 'POST', '/api/mentor', self.run_body(mode='grounded', question=question))
                self.assertEqual(status, expected)

    def test_static_traversal_symlink_escape_and_asset_limit(self):
        secret = self.directory / 'secret.txt'
        secret.write_text('OUTSIDE_STATIC_ROOT')
        with self.http_server() as server:
            (server.static_root / 'outside.txt').symlink_to(secret)
            with (server.static_root / 'large.bin').open('wb') as handle:
                handle.truncate(16 * 1024**2 + 1)
            for path in ('/../secret.txt', '/%2e%2e/secret.txt', '/outside.txt', '/api/nonexistent'):
                with self.subTest(path=path):
                    status, body, _ = self.request(server, path=path)
                    self.assertEqual(status, 404)
                    self.assertNotIn('OUTSIDE_STATIC_ROOT', str(body))
            self.assertEqual(self.request(server, path='/large.bin')[0], 413)
            self.assertEqual(self.request(server, path='/')[0], 200)

    def test_server_rejects_nonloopback_binding(self):
        with self.assertRaises(ValueError):
            Server(('0.0.0.0', 0), self.app, self.directory)

    def test_concurrent_http_compile_is_rejected_and_gate_recovers(self):
        entered, release = threading.Event(), threading.Event()
        def runner(source, cases):
            entered.set()
            if not release.wait(timeout=10):
                raise TimeoutError('Test runner did not receive release')
            return fake_runner(source, cases)
        self.app = self.application(runner=runner)
        with self.http_server() as server, ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(self.request, server, 'POST', '/api/run', self.run_body())
            try:
                self.assertTrue(entered.wait(timeout=5))
                self.assertEqual(self.request(server, 'POST', '/api/run', self.run_body())[0], 429)
                self.assertEqual(self.app.store.profile()['total_attempts'], 0)
            finally:
                release.set()
            self.assertEqual(pending.result(timeout=15)[0], 200)
            self.assertEqual(self.request(server, 'POST', '/api/run', self.run_body())[0], 200)
            self.assertEqual(self.app.store.profile()['total_attempts'], 2)

    def test_runner_exception_returns_generic_error_and_releases_gate(self):
        def broken(source, cases):
            raise RuntimeError('PRIVATE_PROCESS_DETAILS')
        self.app = self.application(runner=broken)
        with self.http_server() as server:
            status, body, _ = self.request(server, 'POST', '/api/run', self.run_body())
            self.assertEqual(status, 500)
            self.assertNotIn('PRIVATE_PROCESS_DETAILS', json.dumps(body))
            self.assertEqual(self.app.store.profile()['total_attempts'], 0)
            self.app.run_function = fake_runner
            self.assertEqual(self.request(server, 'POST', '/api/run', self.run_body())[0], 200)


if __name__ == '__main__':
    unittest.main(verbosity=2)
