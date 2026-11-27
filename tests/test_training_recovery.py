import tempfile
import unittest
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from cancer_lab.data import prepare, write_json, read_json
from cancer_lab.engine import train, evaluate, predict_batch


class TrainingRecoveryTests(unittest.TestCase):
    def test_resize_cache_preserves_training_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rng = np.random.default_rng(23)
            data = root / 'images'
            for label in ('a', 'b'):
                folder = data / label
                folder.mkdir(parents=True)
                for index in range(12):
                    Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)).save(folder / f'{index}.png')
            manifest = root / 'manifest.json'
            prepare(data, manifest, val_ratio=.2, test_ratio=.2)
            cfg = read_json(Path(__file__).resolve().parents[1] / 'configs/baseline.json')
            cfg.update(epochs=2, patience=4, image_size=32, batch_size=4, num_threads=2, device='cpu')
            config = root / 'config.json'
            write_json(config, cfg)
            train(manifest, config, root / 'normal')
            stop = root / 'STOP'
            stop.touch()
            train(manifest, config, root / 'cached', stop_file=stop, cache_images=True)
            stop.unlink()
            train(manifest, config, root / 'cached', resume=True, cache_images=True)
            normal = torch.load(root / 'normal/last.pt', weights_only=True)
            cached = torch.load(root / 'cached/last.pt', weights_only=True)
            for key in normal['state_dict']:
                self.assertTrue(torch.equal(normal['state_dict'][key], cached['state_dict'][key]), key)
            self.assertTrue(read_json(root / 'cached/execution.json')['cache_resized_images'])

    def test_interrupted_finalization_and_partial_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / 'images'
            rng = np.random.default_rng(11)
            for label in ('a', 'b'):
                folder = data / label
                folder.mkdir(parents=True)
                for index in range(12):
                    Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(folder / f'{index}.png')
            manifest = root / 'manifest.json'
            prepare(data, manifest, val_ratio=.2, test_ratio=.2)
            cfg = read_json(Path(__file__).resolve().parents[1] / 'configs/baseline.json')
            cfg.update(epochs=1, patience=1, image_size=32, batch_size=4, num_threads=2, device='cpu')
            config = root / 'config.json'
            write_json(config, cfg)
            run = root / 'run'
            train(manifest, config, run)
            original_weights = (run / 'last.pt').read_bytes()
            (run / 'training_status.json').unlink()
            with patch('cancer_lab.engine.run_epoch', side_effect=AssertionError('Finalization must not train')):
                result = train(manifest, config, run, resume=True)
            self.assertEqual(result['epochs_completed'], 1)
            self.assertEqual((run / 'last.pt').read_bytes(), original_weights)
            # 模拟早停权重落盘后，状态文件还停留在之前的轮次。
            saved = torch.load(run / 'last.pt', weights_only=True)
            saved['stale'] = 1
            torch.save(saved, run / 'last.pt')
            write_json(run / 'training_status.json', dict(epochs_completed=0, status='stop_requested'))
            write_json(config, dict(cfg, epochs=3))
            with patch('cancer_lab.engine.run_epoch', side_effect=AssertionError('Early stop must be finalized')):
                result = train(manifest, config, run, resume=True)
            self.assertEqual(result['status'], 'early_stopping')
            self.assertEqual(result['epochs_completed'], 1)
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
            import run_grouped_comparison
            partial = run / 'val'
            partial.mkdir()
            (partial / 'metrics.json').write_text('{')
            report = run_grouped_comparison.evaluate_once(manifest, run, 'val', data)
            self.assertEqual(len(list(run.glob('val_incomplete_*'))), 1)
            self.assertTrue((partial / 'confusion_matrix.png').is_file())
            with patch.object(run_grouped_comparison, 'evaluate', side_effect=AssertionError('Reuse complete evaluation')):
                cached = run_grouped_comparison.evaluate_once(manifest, run, 'val', data)
            self.assertEqual(cached, report)

    def test_epoch_pause_resume_and_data_relocation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root/'images'
            rng = np.random.default_rng(3)
            for label in ('a','b'):
                folder = data/label
                folder.mkdir(parents=True)
                for index in range(12):
                    Image.fromarray(rng.integers(0,255,(32,32,3),dtype=np.uint8)).save(folder/f'{index}.png')
            manifest = root/'manifest.json'
            prepare(data,manifest,val_ratio=.2,test_ratio=.2)
            cfg = read_json(Path(__file__).resolve().parents[1]/'configs/baseline.json')
            cfg.update(epochs=2,patience=4,image_size=32,batch_size=4,num_threads=2,device='cpu')
            config = root/'config.json'
            write_json(config,cfg)
            full = root/'full'
            train(manifest,config,full)
            stop = root/'STOP'
            stop.touch()
            paused = root/'paused'
            result = train(manifest,config,paused,stop_file=stop)
            self.assertEqual(result['status'],'stop_requested')
            self.assertEqual(result['epochs_completed'],1)
            stop.unlink()
            train(manifest,config,paused,resume=True)
            first = torch.load(full/'last.pt',weights_only=True)
            second = torch.load(paused/'last.pt',weights_only=True)
            for key in first['state_dict']:
                self.assertTrue(torch.equal(first['state_dict'][key],second['state_dict'][key]),key)
            relocated = root/'relocated'
            shutil.move(str(data),str(relocated))
            report = evaluate(manifest,paused/'best.pt',root/'evaluation','cpu','val',relocated)
            self.assertEqual(int(report['macro avg']['support']),len(read_json(manifest)['splits']['val']))
            moved_run = root/'moved_run'
            train(manifest,config,moved_run,data_root=relocated,stop_file=root/'unused_stop')
            snapshot = moved_run/'manifest.json'
            evaluate(snapshot,moved_run/'best.pt',root/'snapshot_evaluation','cpu','val')
            write_json(config,dict(cfg,epochs=3))
            train(snapshot,config,moved_run,resume=True)
            corrupted = read_json(snapshot)
            corrupted['splits']['val'][0]['sha256'] = '0'*64
            changed = root/'changed.json'
            write_json(changed,corrupted)
            with self.assertRaisesRegex(ValueError,'清单'):
                evaluate(changed,moved_run/'best.pt',root/'changed_evaluation','cpu')
            self.assertEqual(read_json(root/'snapshot_evaluation/evaluation_provenance.json')['split'],'val')
            (relocated/'a/broken.jpg').write_bytes(b'broken')
            result = predict_batch(paused/'best.pt',relocated,root/'predictions.json','cpu')
            self.assertEqual(result['samples'],25)
            self.assertEqual(result['errors'],1)
            self.assertEqual(len(read_json(root/'predictions.json')),25)
            with self.assertRaises(ValueError):
                predict_batch(paused/'best.pt',relocated,root/'wrong.txt','cpu')
