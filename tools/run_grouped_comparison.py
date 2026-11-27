"""在固定分组清单上训练 CancerCNN 和 ResNet-18，并按验证结果选择模型。"""
import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cancer_lab.data import read_json, write_json
from cancer_lab.engine import train, evaluate, manifest_identity, manifest_hash
from cancer_lab.split_check import check_splits


def evaluate_once(manifest, run, split, data_root):
    target = run / split
    files = ['metrics.json', 'predictions.csv', 'confusion_matrix.png', 'evaluation_provenance.json']
    complete = all((target / name).is_file() for name in files)
    if complete:
        try:
            with Image.open(target / 'confusion_matrix.png') as image:
                image.verify()
            provenance = read_json(target / 'evaluation_provenance.json')
            metrics = read_json(target / 'metrics.json')
            complete = (provenance.get('checkpoint_sha256') == manifest_hash(run / 'best.pt') and
                        provenance.get('manifest_identity_v1') == manifest_identity(manifest) and
                        provenance.get('split') == split and metrics.get('classification_report') is not None)
        except (OSError, ValueError):
            complete = False
    if complete:
        return metrics['classification_report']
    if target.exists():
        backup = run / f'{split}_incomplete_{time.time_ns()}'
        if run.resolve() not in target.resolve().parents or run.resolve() not in backup.resolve().parents:
            raise ValueError('Partial evaluation is outside the intended run directory')
        target.rename(backup)
    return evaluate(manifest, run / 'best.pt', target, split=split, data_root=data_root)


def run_comparison(manifest, data_root, configurations, output, scope, resume=False, cache_images=False):
    check_splits(manifest)
    configs = {name: read_json(path) for name, path in configurations.items()}
    controlled = [{k: v for k, v in cfg.items() if k != 'architecture'} for cfg in configs.values()]
    if controlled[0] != controlled[1]:
        raise ValueError('Configurations differ beyond architecture')
    protocol = dict(manifest_identity_v1=manifest_identity(manifest), configurations=configs,
                    initialization='random, no pretrained weights', within_run_selection='lowest validation loss',
                    between_model_selection='highest validation macro F1', scope=scope,
                    development_history='Related enhanced images were previously used in the 26-class exploration')
    if resume:
        previous = read_json(output / 'protocol.json')
        if any(previous.get(k) != v for k, v in protocol.items()):
            raise ValueError('Resume requires the same protocol, manifest, and configurations')
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / 'protocol.json', dict(protocol, registered_utc=datetime.now(timezone.utc).isoformat()))
    results = []
    for name, config in configurations.items():
        run = output / name
        try:
            status = read_json(run / 'training_status.json')
        except (OSError, ValueError):
            status = {}
        if status.get('status') not in ('epochs_completed', 'early_stopping'):
            if resume and run.exists() and not (run / 'last.pt').exists():
                backup = output / f'{name}_incomplete_{time.time_ns()}'
                if output.resolve() not in run.resolve().parents or output.resolve() not in backup.resolve().parents:
                    raise ValueError('Unfinished run is outside the intended output directory')
                run.rename(backup)
            train(manifest, config, run, data_root=data_root, resume=resume and (run / 'last.pt').exists(),
                  cache_images=cache_images)
        validation = evaluate_once(manifest, run, 'val', data_root)
        results.append(dict(model=name, validation_accuracy=validation['accuracy'],
                            validation_macro_f1=validation['macro avg']['f1-score']))
    selected = max(results, key=lambda r: r['validation_macro_f1'])['model']
    write_json(output / 'selection.json', dict(model=selected, criterion='highest validation macro F1'))
    for row in results:
        run = output / row['model']
        report = evaluate_once(manifest, run, 'test', data_root)
        row.update(test_accuracy=report['accuracy'], test_macro_f1=report['macro avg']['f1-score'])
    write_json(output / 'metrics.json', results)
    print('GROUPED COMPARISON COMPLETE', results, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=['breakhis', 'lc25000'])
    parser.add_argument('--output')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--cache-images', action='store_true')
    args = parser.parse_args()
    datasets = {
        'breakhis': ('breakhis_patient_v1', 'breakhis_original', 'Two breast classes; conservative case-prefix grouping across magnifications'),
        'lc25000': ('lc25000_tile_v1', 'lc25000_grouped', 'Five lung/colon classes; published reconstructed tile groups, pixel duplicates excluded'),
    }
    split, directory, scope = datasets[args.dataset]
    configs = {name: ROOT / f'configs/grouped_{name}.json' for name in ('cnn', 'resnet18')}
    run_comparison(ROOT / f'data/splits/{split}.json', ROOT / f'data/{directory}/images', configs,
                   ROOT / (args.output or f'outputs/{args.dataset}_comparison_v1'), scope, args.resume, args.cache_images)


if __name__ == '__main__':
    main()
