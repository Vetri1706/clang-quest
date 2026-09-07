"""Real compiler/API integration and model-bound regressions with isolated state."""
import http.client
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from backend.server import Application, Server
from backend import model_bridge
from backend.runner import run_cpp


class LiveJourney(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='questline-integration-')
        cls.path = Path(cls.temp.name) / 'progress.sqlite3'
        cls.app = Application(cls.path, run_function=run_cpp, health_function=lambda: {'available': True})
        cls.server = Server(('127.0.0.1', 0), cls.app)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.challenge = cls.app.challenges['cafe-receipt']

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(5)
        cls.temp.cleanup()

    def request(self, path, body=None):
        client = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=95)
        payload = json.dumps(body) if body is not None else None
        headers = {'Content-Type': 'application/json', 'X-Questline-Token': self.app.csrf}
        client.request('POST' if body is not None else 'GET', path, payload, headers)
        response = client.getresponse()
        data = json.loads(response.read())
        client.close()
        self.assertEqual(response.status, 200, data)
        return data

    def test_editor_to_mentor_to_completed_mission(self):
        c = self.challenge
        initial = self.request('/api/challenges/' + c['id'])
        self.assertNotIn('reference_solution', initial['challenge'])
        self.assertEqual(len(initial['challenge']['tests']), 2)
        failed = self.request('/api/run', {'challenge_id': c['id'], 'source': c['starter_code'], 'mode': 'run'})
        self.assertEqual(failed['compile_status'], 'ok', failed)
        self.assertLess(failed['passed'], failed['total'])
        self.assertEqual(failed['profile']['xp'], 0)
        hint = self.request('/api/mentor', {'challenge_id': c['id'], 'source': c['starter_code'], 'question': 'Give me a small hint'})
        self.assertEqual(hint['kind'], 'hint')
        self.assertEqual(hint['model']['mode'], 'grounded')
        self.assertNotIn(c['reference_solution'], hint['text'])
        done = self.request('/api/run', {'challenge_id': c['id'], 'source': c['reference_solution'], 'mode': 'submit'})
        self.assertEqual((done['compile_status'], done['passed'], done['total']), ('ok', 5, 5), done)
        self.assertEqual(done['xp_awarded'], c['xp'])
        for row in done['results']:
            if row['hidden']:
                self.assertTrue(set(row).isdisjoint({'input', 'expected', 'actual', 'stdout', 'stderr'}))
        restored = Application(self.path, run_function=run_cpp, health_function=lambda: {'available': True}).detail(c['id'])
        self.assertEqual(restored['source'], c['reference_solution'])
        self.assertEqual(len(restored['messages']), 2)
        self.assertEqual(restored['last_run']['passed'], 5)

    def test_custom_input_is_executed_without_grading_or_xp(self):
        c = self.challenge
        result = self.request('/api/run', {'challenge_id': c['id'], 'source': c['reference_solution'], 'mode': 'custom', 'stdin': '7 90\n'})
        self.assertEqual(result['results'][0]['status'], 'executed', result)
        self.assertEqual(result['results'][0]['actual'].strip(), '630')
        self.assertNotIn('expected', result['results'][0])
        self.assertEqual(result['xp_awarded'], 0)

    def test_real_compiler_diagnostic_reaches_mentor(self):
        c = self.challenge
        source = '#include <iostream>\nint main(){ std::cout << missing_name; }'
        result = self.request('/api/run', {'challenge_id': c['id'], 'source': source, 'mode': 'run'})
        self.assertNotEqual(result['compile_status'], 'ok')
        self.assertIn('missing_name', result['diagnostics'])
        reply = self.request('/api/mentor', {'challenge_id': c['id'], 'source': source, 'question': 'Explain the compiler error from my last run'})
        self.assertEqual(reply['kind'], 'diagnostic')
        self.assertIn('missing_name', reply['text'])
        stale = self.request('/api/mentor', {'challenge_id': c['id'], 'source': 'int unrelated_current_line = 42;', 'question': 'Explain the compiler error from my last run'})
        self.assertIn('earlier draft', stale['text'])
        self.assertNotIn('unrelated_current_line', stale['text'])


class ModelBounds(unittest.TestCase):
    def test_small_context_does_not_loop_forever(self):
        tokenizer = SimpleNamespace(encode=lambda text, add_bos=False: [0] + list(text))
        fake_model = SimpleNamespace(config=SimpleNamespace(max_seq_len=16))
        seen = {}
        def generate(model, tokenizer, prompt, **kwargs):
            seen.update({'prompt': prompt, **kwargs})
            return 'experimental output'
        fake_module = SimpleNamespace(generate=generate)
        with patch.object(model_bridge, '_loaded', (fake_module, fake_model, tokenizer)), patch.object(model_bridge, 'model_status', return_value={'available': True}):
            response = model_bridge.experimental_reply('A prompt that must be shortened.')
        self.assertLessEqual(len(seen['prompt']) + 1 + seen['max_tokens'], 16)
        self.assertEqual(response['model']['mode'], 'experimental')
        self.assertIn('not a verified C++ explanation', response['text'])

    def test_unusable_context_releases_lock(self):
        with patch.object(model_bridge, '_loaded', (None, SimpleNamespace(config=SimpleNamespace(max_seq_len=1)), None)), patch.object(model_bridge, 'model_status', return_value={'available': True}):
            with self.assertRaises(ValueError):
                model_bridge.experimental_reply('test')
        self.assertTrue(model_bridge._lock.acquire(blocking=False))
        model_bridge._lock.release()


class ShutdownDrain(unittest.TestCase):
    def test_server_close_drains_active_request(self):
        entered, finish = threading.Event(), threading.Event()
        def runner(source, cases):
            entered.set()
            finish.wait(4)
            return {'compile_status': 'ok', 'diagnostics': '', 'results': []}
        with tempfile.TemporaryDirectory() as folder:
            app = Application(Path(folder)/'progress.db', run_function=runner, health_function=lambda: {'available': True})
            server = Server(('127.0.0.1', 0), app)
            serving = threading.Thread(target=server.serve_forever, daemon=True)
            serving.start()
            errors = []
            def client():
                try:
                    connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=8)
                    connection.request('POST','/api/run',json.dumps({'challenge_id':'cafe-receipt','source':'int main(){}','mode':'run'}),{'X-Questline-Token':app.csrf,'Content-Type':'application/json'})
                    connection.getresponse().read()
                    connection.close()
                except Exception as error:
                    errors.append(error)
            request = threading.Thread(target=client)
            request.start()
            self.assertTrue(entered.wait(4))
            server.shutdown()
            closing = threading.Thread(target=server.server_close)
            closing.start()
            closing.join(.15)
            self.assertTrue(closing.is_alive(), 'Server closed before the active request could clean up')
            finish.set()
            closing.join(6)
            request.join(6)
            serving.join(6)
            self.assertFalse(closing.is_alive())
            self.assertFalse(errors)


if __name__ == '__main__':
    unittest.main()
