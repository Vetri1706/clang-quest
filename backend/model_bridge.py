"""Opt-in access to the user's actual NumPy model, never a trusted answer source."""
import importlib.util
import os
from pathlib import Path
import sys
import threading

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')

_lock = threading.Lock()
_loaded = None
FRAMEWORK = Path(__file__).resolve().parents[1] / 'ai'
RUN = FRAMEWORK / 'runs/hardened_dpo'


def model_status():
    available = (RUN / 'checkpoint/manifest.json').is_file() and (FRAMEWORK / 'generate.py').is_file() and importlib.util.find_spec('numpy') is not None
    return {'available': available, 'mode': 'experimental',
            'name': 'Your NumPy model', 'quality_gate': 'failed', 'trained_parameters': 29656,
            'inference_path': 'ai/generate.py', 'checkpoint_path': 'ai/runs/hardened_dpo/checkpoint/manifest.json',
            'training_basis_path': 'ai/TRAINING_BASIS.md'}


def experimental_reply(question):
    global _loaded
    if not _lock.acquire(blocking=False):
        raise ValueError('The experimental model is busy. Try again in a moment.')
    try:
        if not model_status()['available']:
            raise ValueError('The bundled NumPy checkpoint or NumPy dependency is unavailable. Follow the AI environment setup in README.md.')
        if _loaded is None:
            sys.path.insert(0, str(FRAMEWORK))
            spec = importlib.util.spec_from_file_location('questline_numpy_generate', FRAMEWORK / 'generate.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            from rawllm.model import Config
            from rawllm.safeio import read_json
            configuration = Config(**read_json(RUN / 'settings.json')['config'])
            if configuration.parameter_count > 2_000_000 or configuration.max_seq_len > 512:
                raise ValueError('This dashboard only loads bounded research checkpoints up to 2 million parameters and 512 context tokens.')
            model, tokenizer = module.load_run(RUN)
            _loaded = module, model, tokenizer
        module, model, tokenizer = _loaded
        context = model.config.max_seq_len
        if type(context) is not int or context < 3:
            raise ValueError('The checkpoint context window is too small for generation.')
        budget = min(24, context - 2)
        prompt = question[:120]
        while prompt and len(tokenizer.encode(prompt, add_bos=True)) + budget > context:
            prompt = prompt[:-8]
        if len(tokenizer.encode(prompt, add_bos=True)) + budget > context:
            raise ValueError('The tokenizer cannot fit this checkpoint context window.')
        answer = module.generate(model, tokenizer, prompt, max_tokens=budget, temperature=0, seed=17)
        answer = ''.join(c if c.isprintable() or c in '\n\t' else '\ufffd' for c in answer)
        return {'text': 'Experimental model output — not a verified C++ explanation.\n\n' + (answer or '[The model ended without producing text.]'),
                'kind': 'experimental', 'sources': [], 'suggestions': ['Switch to guided mentoring for a sourced explanation'],
                'next_hint_level': 0, 'model': {'mode': 'experimental', 'label': 'Your NumPy model · unvalidated'}}
    finally:
        _lock.release()
