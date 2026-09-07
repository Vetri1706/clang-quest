"""Regression checks for the in-repository model artifacts and dashboard bridge."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

from backend import model_bridge

ROOT = Path(__file__).resolve().parents[1]
AI = ROOT / 'ai'


class BundledArtifacts(unittest.TestCase):
    def test_model_resolves_inside_repository(self):
        self.assertEqual(model_bridge.FRAMEWORK, AI)
        self.assertTrue(model_bridge.RUN.is_relative_to(AI))
        status = model_bridge.model_status()
        self.assertEqual(status['checkpoint_path'], 'ai/runs/hardened_dpo/checkpoint/manifest.json')
        self.assertEqual(status['quality_gate'], 'failed')

    def test_all_three_training_stages_have_matching_payloads_and_datasets(self):
        for stage in ('pretrain', 'sft', 'dpo'):
            with self.subTest(stage=stage):
                folder = AI / 'runs' / ('hardened_' + stage)
                settings = json.loads((folder / 'settings.json').read_text())
                manifest = json.loads((folder / 'checkpoint/manifest.json').read_text())
                payload = folder / 'checkpoint' / manifest['payload']
                self.assertEqual(Path(manifest['payload']).name, manifest['payload'])
                self.assertEqual(hashlib.sha256(payload.read_bytes()).hexdigest(), manifest['sha256'])
                self.assertEqual(hashlib.sha256((folder / 'tokenizer.json').read_bytes()).hexdigest(), settings['tokenizer_sha256'])
                self.assertEqual(hashlib.sha256((AI / settings['dataset']).read_bytes()).hexdigest(), settings['dataset_sha256'])
                self.assertFalse(Path(settings['dataset']).is_absolute())
                self.assertEqual(manifest['counters']['updates'], 3)
                self.assertEqual(manifest['counters']['skipped_updates'], 0)
                self.assertEqual(manifest['format'], 'rawllm-checkpoint')
                if stage == 'dpo':
                    self.assertEqual(hashlib.sha256((folder / 'reference.npz').read_bytes()).hexdigest(), settings['reference_sha256'])

    def test_historical_weight_selection_is_identical_to_import_provenance(self):
        provenance = json.loads((AI / 'IMPORT_PROVENANCE.json').read_text())
        payloads = [item for item in provenance['files'] if item['path'].endswith('.npz')]
        self.assertEqual(len(payloads), 4)
        for item in payloads:
            self.assertFalse(item['transformed'])
            self.assertEqual(hashlib.sha256((AI / item['path']).read_bytes()).hexdigest(), item['sha256'])


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'Install requirements-ai.txt for neural inference tests')
class ActualBundledInference(unittest.TestCase):
    def test_actual_parameters_and_deterministic_greedy_dashboard_inference(self):
        first = model_bridge.experimental_reply('C++: 5 / 2? Answer:')
        second = model_bridge.experimental_reply('C++: 5 / 2? Answer:')
        self.assertEqual(first['text'], second['text'])
        self.assertEqual(first['model']['mode'], 'experimental')
        self.assertIn('not a verified C++ explanation', first['text'])
        self.assertEqual(first['sources'], [])
        _, model, tokenizer = model_bridge._loaded
        count = sum(parameter.data.size for parameter in model.params.values())
        self.assertEqual(count, 29656)
        self.assertEqual(count, model.config.parameter_count)
        self.assertEqual(model.config.vocab_size, tokenizer.vocab_size)
        self.assertEqual(model.config.max_seq_len, 256)


if __name__ == '__main__':
    unittest.main()
