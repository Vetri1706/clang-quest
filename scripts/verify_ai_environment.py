#!/usr/bin/env python3
"""Exercise the actual Questline HTTP app, NumPy checkpoint, and C++ sandbox.

No mock model, mock runner, personal database, network model, or API key is used.
A passing exit code certifies this bounded integration journey, not AI teaching
competence. The model's existing quality gate remains a separate report field.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import http.client
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import threading
import time

PROMPTS = [
    'Write a complete C++20 program that reads two integers and prints their product.',
    'Explain RAII and give one C++ example using std::unique_ptr.',
    'How does LLVM SSA differ from C++ source variables?',
]
DIAGNOSTIC_SOURCE = '#include <iostream>\nint main(){ std::cout << missing_name; }\n'
EXPERIMENTAL_PREFIX = 'Experimental model output — not a verified C++ explanation.\n\n'
PRIVATE_FIELDS = {'input', 'expected', 'actual', 'stdout', 'stderr'}
THREAD_KEYS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS')
MAX_RESPONSE_BYTES = 1024 * 1024


class VerificationFailure(RuntimeError):
    pass


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(65536), b''):
            digest.update(chunk)
    return digest.hexdigest()


def scrub(value):
    """Never persist tokens, profile details, or hidden-case payloads."""
    if isinstance(value, dict):
        excluded = {'csrf', 'profile', 'headers', 'draft_updated_at', 'saved_at'}
        if value.get('hidden') is True:
            excluded |= PRIVATE_FIELDS
        return {key: scrub(item) for key, item in value.items() if key not in excluded}
    if isinstance(value, list):
        return [scrub(item) for item in value]
    return value


def command_value(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5,
                                env={'PATH': '/usr/bin:/bin', 'LANG': 'C', 'LC_ALL': 'C'})
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def environment():
    version = command_value(['/usr/bin/clang++', '--version'])
    compiler_lines = [line for line in (version or '').splitlines()
                      if line.startswith(('Apple clang version', 'Target:', 'Thread model:'))]
    try:
        numpy_version = importlib.metadata.version('numpy')
    except importlib.metadata.PackageNotFoundError:
        numpy_version = None
    memory = command_value(['/usr/sbin/sysctl', '-n', 'hw.memsize'])
    return {
        'os': platform.system(), 'os_release': platform.release(),
        'macos_version': platform.mac_ver()[0], 'architecture': platform.machine(),
        'cpu_model': command_value(['/usr/sbin/sysctl', '-n', 'machdep.cpu.brand_string']),
        'logical_cpu_count': os.cpu_count(),
        'physical_memory_bytes': int(memory) if memory and memory.isdigit() else None,
        'python_version': platform.python_version(), 'numpy_version': numpy_version,
        'compiler': compiler_lines, 'sandbox_executable_present': Path('/usr/bin/sandbox-exec').is_file(),
        'thread_limits': {key: os.environ[key] for key in THREAD_KEYS},
        'docker_used': False, 'model_network_requests': False,
        'submitted_code_path': 'Apple clang++ -> macOS Seatbelt -> local C++ executable',
    }


def observe_runner_launches(records):
    """Passively observe actual trusted-helper arguments; never retain private paths."""
    def observe(event, arguments):
        if event != 'subprocess.Popen':
            return
        try:
            argv = arguments[1]
            if not isinstance(argv, (list, tuple)) or len(argv) != 4 or argv[2] != '_child':
                return
            config = json.loads(argv[3])
            command = config.get('command')
            if not isinstance(command, list) or not isinstance(config.get('compile'), bool):
                return
            environment_keys = sorted((arguments[3] or {}).keys())
            records.append({
                'phase': 'compile' if config['compile'] else 'run',
                'command_with_paths_reduced_to_basenames': [Path(word).name if word.startswith('/') else word for word in command],
                'helper_cpu_soft_seconds': config['seconds'],
                'parent_rss_limit_bytes': config['memory_bytes'],
                'inherited_environment_keys': environment_keys,
                'profile_default_deny': '(deny default)' in config['profile'],
                'profile_explicit_other_process_info_denial': '(deny process-info*)' in config['profile'],
            })
        except Exception:
            # Evidence collection never changes subprocess execution behavior.
            return
    sys.addaudithook(observe)


def bridge_generated_text(response):
    text = response['text'][len(EXPERIMENTAL_PREFIX):]
    return '' if text == '[The model ended without producing text.]' else text


def generation_flags(bridge_source):
    """Extract constants from the deployed bridge rather than inventing flags."""
    tree = ast.parse(bridge_source)
    values = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'generate':
            for keyword in node.keywords:
                if isinstance(keyword.value, ast.Constant) and keyword.arg:
                    values[keyword.arg] = keyword.value.value
    return values


def source_inventory(project):
    paths = [
        'backend/server.py', 'backend/model_bridge.py', 'backend/runner.py',
        'backend/mentor.py', 'backend/store.py', 'backend/knowledge.json',
        'backend/data/curriculum.json', 'ai/generate.py', 'ai/rawllm/model.py',
        'ai/rawllm/cache.py', 'ai/rawllm/tokenizer.py', 'ai/rawllm/safeio.py',
    ]
    return {name: file_hash(project / name) for name in paths if (project / name).is_file()}


def model_evidence(bridge, project):
    loaded = bridge._loaded
    if loaded is None:
        raise VerificationFailure('The real HTTP request did not load the checkpoint.')
    _, model, tokenizer = loaded
    parameters = model.params
    digest = hashlib.sha256()
    count = 0
    for name in sorted(parameters):
        array = parameters[name].data
        count += array.size
        metadata = json.dumps({'name': name, 'shape': list(array.shape), 'dtype': str(array.dtype)},
                              sort_keys=True, separators=(',', ':')).encode()
        digest.update(metadata + b'\0' + array.tobytes(order='C') + b'\0')
    run = bridge.RUN.resolve()
    try:
        run_relative = str(run.relative_to(project))
    except ValueError as exc:
        raise VerificationFailure('The model must load from the bundled project ai directory.') from exc
    manifest_path = run / 'checkpoint/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    filename = manifest.get('payload')
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise VerificationFailure('Checkpoint payload name is not a bounded local filename.')
    payload_hash = file_hash(run / 'checkpoint' / filename)
    if payload_hash != manifest.get('sha256'):
        raise VerificationFailure('The checkpoint payload hash did not match the manifest.')
    config = vars(model.config)
    context = config['max_seq_len']
    return {
        'run_directory': run_relative, 'parameter_count': int(count),
        'named_parameter_tensors': len(parameters), 'architecture': config,
        'architecture_parameter_count': int(model.config.parameter_count),
        'tokenizer_vocab_size': tokenizer.vocab_size,
        'checkpoint_manifest_sha256': file_hash(manifest_path),
        'checkpoint_payload_file': filename, 'checkpoint_payload_sha256': payload_hash,
        'loaded_weights_sha256': digest.hexdigest(),
        'loaded_weights_hash_encoding': 'sorted parameter names; compact sorted JSON(name,shape,dtype), NUL, C-order bytes, NUL',
        'settings_sha256': file_hash(run / 'settings.json'),
        'tokenizer_sha256': file_hash(run / 'tokenizer.json'),
        'generation_flags': {
            **generation_flags((project / 'backend/model_bridge.py').read_text()),
            'max_output_tokens': min(24, context - 2),
            'question_character_cap': 120, 'decoding': 'greedy when temperature is zero',
        },
        'quality_gate_reported_by_application': bridge.model_status().get('quality_gate'),
        'training_performed_by_this_verification': False,
    }


class Journey:
    def __init__(self, server, report, seconds):
        self.server, self.report = server, report
        self.deadline = time.monotonic() + seconds
        self.token = None

    def check(self, condition, name):
        self.report['mechanics_checks'].append({'name': name, 'passed': bool(condition)})
        if not condition:
            raise VerificationFailure(name)

    def request(self, name, path, body=None, record=True):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise VerificationFailure('The verification journey exceeded its time budget.')
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port,
                                                timeout=min(95, remaining))
        headers = {'Content-Type': 'application/json',
                   'Origin': f'http://127.0.0.1:{self.server.server_port}'}
        if self.token:
            headers['X-Questline-Token'] = self.token
        started = time.monotonic()
        try:
            connection.request('POST' if body is not None else 'GET', path,
                               json.dumps(body) if body is not None else None, headers)
            response = connection.getresponse()
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise VerificationFailure('An HTTP response exceeded the evidence byte budget.')
            payload = json.loads(raw)
            status = response.status
        finally:
            connection.close()
        if record:
            request = {key: value for key, value in (body or {}).items() if key != 'source'}
            if body and 'source' in body:
                request['source_sha256'] = sha256_bytes(body['source'].encode())
                request['source_bytes'] = len(body['source'].encode())
            self.report['http_evidence'].append({
                'step': name, 'method': 'POST' if body is not None else 'GET',
                'path': path, 'request': request, 'status': status,
                'duration_ms': round((time.monotonic() - started) * 1000, 2),
                'response': scrub(payload),
            })
        self.check(status == 200, name + ': HTTP 200')
        return payload

    def mentor(self, name, source, question, mode):
        return self.request(name, '/api/mentor', {'challenge_id': 'cafe-receipt',
                            'source': source, 'question': question, 'mode': mode})

    def run(self, name, source, mode='run'):
        return self.request(name, '/api/run', {'challenge_id': 'cafe-receipt',
                            'source': source, 'mode': mode})


def fenced(text, language='text'):
    longest = max((len(x) for x in re.findall(r'`+', text)), default=0)
    fence = '`' * max(3, longest + 1)
    return fence + language + '\n' + text + '\n' + fence


def markdown(report):
    summary = report.get('summary', {})
    model = report.get('model', {})
    checks = report['mechanics_checks']
    lines = [
        '# AI inside the Questline environment — actual execution evidence', '',
        f"Recorded UTC: {report['started_at_utc']}. Total wall time: {report.get('duration_seconds', 0):.2f} seconds.", '',
        f"**Integration mechanics: {report['mechanics_status'].upper()}.** "
        f"{sum(c['passed'] for c in checks)} of {len(checks)} recorded checks passed.", '',
        '**Neural teaching competence: not established.** This report does not convert an inference smoke test '
        'or a curated reference solution into a neural C++ quality-gate pass.', '',
        '## What actually ran', '',
        'The script started the real Application and HTTP Server on an ephemeral loopback port with a fresh '
        'temporary SQLite database. HTTP requests invoked the default experimental NumPy model, grounded '
        'mentor, and sandboxed Apple C++ compiler. No model or execution mocks were injected. '
        'Personal progress, session tokens, hostnames, account names, and private test payloads are excluded.', '',
        f"The loaded checkpoint contains **{model.get('parameter_count', 'unavailable')} parameters**. "
        f"Application quality gate: **{model.get('quality_gate_reported_by_application', 'unavailable')}**.", '',
        f"Checkpoint payload SHA-256: `{model.get('checkpoint_payload_sha256', 'unavailable')}`.",
        f"Loaded numerical weights SHA-256: `{model.get('loaded_weights_sha256', 'unavailable')}`.", '',
        '## Observed neural outputs', '',
    ]
    for index, item in enumerate(report.get('neural_outputs', []), 1):
        lines.extend([f'### Prompt {index}', '', item['prompt'], '',
                      fenced(item['generated_text']), '',
                      f"Generated UTF-8 bytes: {item['utf8_bytes']}. C++ code fence present: {item['cpp_fence_present']}.", ''])
    lines.extend([
        f"Distinct outputs for the three different prompts: **{summary.get('distinct_neural_outputs', 'unavailable')}**. "
        f"Greedy repeat matched: **{summary.get('greedy_repeat_equal', 'unavailable')}**.", '',
        'The first model output was submitted to the real compiler exactly as generated, including any prose '
        'or Markdown. The verifier did not add includes, a main function, corrected syntax, or a replacement '
        'solution. A compiler failure here is recorded as a neural capability observation, not an integration failure.', '',
        f"Neural output compile status: **{summary.get('neural_compile_status', 'unavailable')}**; "
        f"visible tests passed: **{summary.get('neural_visible_tests_passed', 0)}**.", '',
        '## Grounded mentoring and a real learning journey', '',
        'A deliberately broken original fixture referenced `missing_name`. The real compiler rejected it; '
        'the grounded mentor then explained that actual diagnostic. A separate hint request returned a hint. '
        'An explicit “show solution” request returned the curated café reference, which the verifier extracted '
        'unchanged from that response and submitted to all tests.', '',
        f"Curated reference test result: **{summary.get('reference_passed', 0)}/{summary.get('reference_total', 0)}**. "
        f"Hidden test payloads redacted: **{summary.get('hidden_tests_redacted', False)}**.", '',
        'This passing program came from authored curriculum content. It was not generated by the neural model '
        'and is not evidence that the tiny model can independently solve the mission.', '',
    ])
    for step in ('broken_source_compilation', 'grounded_compiler_explanation', 'grounded_hint',
                 'explicit_curated_reference', 'curated_reference_submission', 'neural_program_submission'):
        event = next((event for event in report['http_evidence'] if event['step'] == step), None)
        if not event:
            continue
        lines.extend([f'### {step.replace("_", " ").capitalize()}', '',
                      fenced(json.dumps(event['response'], indent=2, ensure_ascii=True), 'json'), ''])
    lines.extend(['## Reproducibility', '',
                  fenced(json.dumps({'environment': report['environment'], 'model': model,
                                     'source_sha256': report['source_sha256'],
                                     'verification_script_sha256': report['verification_script_sha256'],
                                     'runtime_limits': report.get('runtime_limits', {}),
                                     'observed_process_launches': report.get('observed_process_launches', [])},
                                    indent=2, ensure_ascii=True), 'json'), '',
                  'Run from the repository root:', '',
                  fenced('python3 scripts/verify_ai_environment.py --output reports/ai', 'sh'), '',
                  'The JSON report contains safe HTTP transcripts, checks, hashes, and runtime settings. '
                  'Exit status 0 means this bounded integration journey passed; exit status 1 means a '
                  'mechanical check or execution requirement failed. Neural answer quality is reported '
                  'separately and never silently treated as passed.', ''])
    if report.get('failure'):
        lines.extend(['## Failure', '', report['failure'], ''])
    return '\n'.join(lines)


def verify(project, output, max_seconds):
    started = time.monotonic()
    for key in THREAD_KEYS:
        os.environ[key] = '1'
    report = {
        'format': 'questline-ai-environment-validation', 'version': 1,
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'mechanics_status': 'failed', 'mechanics_checks': [], 'http_evidence': [],
        'neural_outputs': [], 'summary': {}, 'environment': environment(),
        'observed_process_launches': [],
        'source_sha256': source_inventory(project),
        'verification_script_sha256': file_hash(__file__),
        'test_policy': {'real_http': True, 'real_model': True, 'real_sandboxed_compiler': True,
                        'mocks': False, 'database': 'fresh temporary SQLite, deleted after the run',
                        'concurrent_compiler_jobs': 1, 'maximum_journey_seconds': max_seconds,
                        'model_quality_pass_is_not_an_exit_condition': True},
    }
    server = worker = None
    try:
        if not (project / 'backend/server.py').is_file() or not (project / 'ai/generate.py').is_file():
            raise VerificationFailure('Run against a Questline project with the bundled ai directory.')
        sys.path.insert(0, str(project))
        from backend.server import Application, Server
        from backend import model_bridge
        observe_runner_launches(report['observed_process_launches'])
        with tempfile.TemporaryDirectory(prefix='questline-ai-verification-') as temp:
            app = Application(database=Path(temp) / 'verification.sqlite3')
            server = Server(('127.0.0.1', 0), app)
            worker = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .05}, daemon=False)
            worker.start()
            journey = Journey(server, report, max(1, max_seconds - (time.monotonic() - started)))
            boot = journey.request('bootstrap', '/api/bootstrap', record=False)
            journey.token = boot['csrf']
            journey.check(bool(boot['compiler'].get('available')), 'Real compiler health smoke test passed')
            journey.check(bool(boot['custom_model'].get('available')), 'Bundled experimental model is available')
            report['runtime_limits'] = boot['compiler'].get('limits', {})
            report['compiler_health'] = scrub(boot['compiler'])
            report['model_status'] = scrub(boot['custom_model'])
            detail = journey.request('public_challenge', '/api/challenges/cafe-receipt')
            challenge = detail['challenge']
            starter = detail['source']
            journey.check('reference_solution' not in challenge and 'hints' not in challenge,
                          'Ordinary challenge endpoint withholds reference and private hints')
            journey.check(len(challenge['tests']) == 2 and challenge['hidden_count'] == 3,
                          'Café fixture has two public and three hidden cases')
            for index, prompt in enumerate(PROMPTS, 1):
                response = journey.mentor('experimental_prompt_' + str(index), starter, prompt, 'experimental')
                journey.check(response.get('kind') == 'experimental' and response.get('model', {}).get('mode') == 'experimental',
                              'Prompt ' + str(index) + ' used the experimental neural route')
                journey.check(response['text'].startswith(EXPERIMENTAL_PREFIX),
                              'Prompt ' + str(index) + ' retained the unvalidated-output label')
                generated = bridge_generated_text(response)
                report['neural_outputs'].append({'prompt': prompt, 'generated_text': generated,
                    'utf8_bytes': len(generated.encode()),
                    'cpp_fence_present': bool(re.search(r'```(?:cpp|c\+\+)\b', generated, re.IGNORECASE))})
            report['model'] = model_evidence(model_bridge, project)
            journey.check(report['model']['parameter_count'] == report['model']['architecture_parameter_count'],
                          "The bundled checkpoint loaded the architecture's complete parameter inventory")
            repeat = journey.mentor('experimental_greedy_repeat', starter, PROMPTS[0], 'experimental')
            repeated_text = bridge_generated_text(repeat)
            report['summary']['greedy_repeat_equal'] = repeated_text == report['neural_outputs'][0]['generated_text']
            journey.check(report['summary']['greedy_repeat_equal'], 'The same greedy prompt repeats deterministically')
            report['summary']['distinct_neural_outputs'] = len({item['generated_text'] for item in report['neural_outputs']})
            neural_source = report['neural_outputs'][0]['generated_text']
            neural_run = journey.run('neural_program_submission', neural_source)
            journey.check(neural_run['compile_status'] in {'ok', 'compile_error'},
                          'Actual neural output reached a working compiler')
            report['summary']['neural_compile_status'] = neural_run['compile_status']
            report['summary']['neural_visible_tests_passed'] = neural_run['passed']
            report['summary']['neural_teaching_quality'] = 'not established; existing failed application gate retained'
            broken = journey.run('broken_source_compilation', DIAGNOSTIC_SOURCE)
            journey.check(broken['compile_status'] == 'compile_error' and 'missing_name' in broken['diagnostics'],
                          'The real compiler diagnosed the deliberate unknown identifier')
            explained = journey.mentor('grounded_compiler_explanation', DIAGNOSTIC_SOURCE,
                                        'Explain the compiler error from my last run', 'grounded')
            journey.check(explained['kind'] == 'diagnostic' and 'missing_name' in explained['text'] and
                          explained.get('model', {}).get('mode') == 'grounded' and bool(explained.get('sources')),
                          'Grounded mentoring explained the actual compiler diagnostic with a source')
            hint = journey.mentor('grounded_hint', starter, 'Give me a small hint', 'grounded')
            journey.check(hint['kind'] == 'hint', 'A hint request returned a hint rather than a solution')
            reference = journey.mentor('explicit_curated_reference', starter, 'show solution', 'grounded')
            match = re.search(r'(`{3,})cpp\n(.*?)\n\1', reference['text'], re.DOTALL)
            journey.check(reference['kind'] == 'solution' and match is not None,
                          'An explicit request returned a fenced curated reference solution')
            reference_source = match.group(2)
            report['reference_source_sha256'] = sha256_bytes(reference_source.encode())
            journey.check(reference_source not in hint['text'], 'The earlier hint did not disclose the complete reference')
            completed = journey.run('curated_reference_submission', reference_source, 'submit')
            journey.check((completed['compile_status'], completed['passed'], completed['total']) == ('ok', 5, 5),
                          'The unchanged curated reference passed all five real C++ executions')
            hidden = [row for row in completed['results'] if row.get('hidden')]
            redacted = len(hidden) == 3 and all(not PRIVATE_FIELDS.intersection(row) for row in hidden)
            journey.check(redacted, 'All three hidden result payloads stayed redacted at the HTTP boundary')
            journey.check(completed.get('xp_awarded') == challenge['xp'], 'The successful submission awarded the fixture XP once')
            report['summary'].update(reference_passed=completed['passed'], reference_total=completed['total'],
                                     hidden_tests_redacted=redacted,
                                     curated_reference_origin='authored curriculum returned after an explicit HTTP mentor request')
            final_model = model_evidence(model_bridge, project)
            journey.check(final_model['loaded_weights_sha256'] == report['model']['loaded_weights_sha256'],
                          'Inference and mentoring did not mutate the loaded neural weights')
            launches = report['observed_process_launches']
            journey.check(any(item['phase'] == 'compile' for item in launches) and
                          any(item['phase'] == 'run' for item in launches) and
                          all(item['profile_default_deny'] for item in launches),
                          'Passive audit observed real helper launches with default-deny sandbox profiles')
            server.shutdown()
            server.server_close()
            worker.join(5)
            journey.check(not worker.is_alive(), 'The temporary HTTP server stopped cleanly')
            server = worker = None
            report['mechanics_status'] = 'passed'
    except VerificationFailure as exc:
        report['failure'] = str(exc)
    except Exception as exc:
        report['failure'] = 'Verification stopped because of ' + type(exc).__name__ + '; inspect the last recorded step.'
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if worker is not None:
            worker.join(5)
        report['duration_seconds'] = round(time.monotonic() - started, 3)
        output.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(scrub(report), indent=2, ensure_ascii=True, allow_nan=False) + '\n'
        (output / 'environment-validation.json').write_text(serialized, encoding='utf-8')
        (output / 'environment-validation.md').write_text(markdown(scrub(report)), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-seconds', type=int, default=180)
    args = parser.parse_args()
    if not 30 <= args.max_seconds <= 300:
        parser.error('--max-seconds must be between 30 and 300')
    project = args.project.resolve()
    output = (args.output or project / 'reports/ai').resolve()
    report = verify(project, output, args.max_seconds)
    print(json.dumps({'mechanics_status': report['mechanics_status'],
                      'mechanics_checks_passed': sum(c['passed'] for c in report['mechanics_checks']),
                      'neural_teaching_quality': report['summary'].get('neural_teaching_quality', 'not established'),
                      'duration_seconds': report['duration_seconds'],
                      'reports': ['environment-validation.json', 'environment-validation.md']}, indent=2))
    return 0 if report['mechanics_status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
