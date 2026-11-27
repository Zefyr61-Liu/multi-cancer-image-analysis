import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cancer_lab.engine import adapted_evaluation, manifest_hash, manifest_identity


class AdaptedEvaluationTests(unittest.TestCase):
    def test_historical_checkpoint_requires_registered_protocol_and_test_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / 'candidate.pt'
            protocol = candidate.with_name('protocol.json')
            protocol.write_text('{}', encoding='utf-8')
            digest = manifest_hash(protocol)
            checkpoint_digest = 'example-checkpoint'
            manifest = Path(directory) / 'manifest.json'
            manifest.write_text('{"root":"original","classes":["x"]}', encoding='utf-8')
            identity = manifest_identity(manifest)
            registry = {checkpoint_digest: dict(
                protocol_sha256=digest, epoch=12, manifests={identity: ['test']},
                scope='observed test')}
            saved = {'protocol_sha256': digest}
            with patch('cancer_lab.engine.read_json', return_value=registry):
                epoch, provenance = adapted_evaluation(candidate, saved, checkpoint_digest, identity, 'test')
                self.assertEqual(epoch, 12)
                self.assertEqual(provenance['protocol_sha256'], digest)
                registry[checkpoint_digest]['protocol_field'] = 'adaptation_protocol_sha256'
                self.assertEqual(adapted_evaluation(candidate, {'adaptation_protocol_sha256': digest},
                    checkpoint_digest, identity, 'test')[0], 12)
                del registry[checkpoint_digest]['protocol_field']
                for manifest_id, split in [('unregistered', 'test'), (identity, 'val')]:
                    with self.assertRaisesRegex(ValueError, '未授权'):
                        adapted_evaluation(candidate, saved, checkpoint_digest, manifest_id, split)
                with self.assertRaisesRegex(ValueError, '没有已登记'):
                    adapted_evaluation(candidate, saved, 'different-checkpoint', identity, 'test')
                with self.assertRaisesRegex(ValueError, '协议校验失败'):
                    adapted_evaluation(candidate, {'protocol_sha256': 'different'}, checkpoint_digest, identity, 'test')
                protocol.write_text(json.dumps({'changed': True}), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, '协议校验失败'):
                    adapted_evaluation(candidate, saved, checkpoint_digest, identity, 'test')
