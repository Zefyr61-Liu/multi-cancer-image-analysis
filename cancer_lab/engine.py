"""训练只使用 train/val；evaluate 独立读取 test 并生成报告。"""
import csv
import hashlib
import platform
import random
import time
import importlib.metadata
import math
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch import nn
from torch.utils.data import DataLoader

from .data import CancerDataset, read_json, write_json, transform, load_manifest, EXTENSIONS
from .model import build_model
from .split_check import check_splits


def device_for(name):
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu') if name == 'auto' else torch.device(name)


def manifest_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_identity(path):
    """v1 内容身份忽略存储根目录，保留类别、划分、图片哈希和其他元数据。"""
    value = read_json(path)
    value.pop('root', None)
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


def matches_manifest(saved, path):
    if 'manifest_identity_v1' in saved:
        return saved['manifest_identity_v1'] == manifest_identity(path)
    # 旧权重继续按原字节身份严格验证。
    return 'manifest_sha256' in saved and saved['manifest_sha256'] == manifest_hash(path)


def adapted_evaluation(checkpoint, saved, checkpoint_digest, manifest_id, split):
    """Verify the release record for a historical adaptation checkpoint."""
    registry = read_json(Path(__file__).with_name('adapted_evaluation.json'))
    record = registry.get(checkpoint_digest)
    if record is None:
        raise ValueError('该适配权重没有已登记的通用评估记录。')
    protocol = Path(checkpoint).with_name('protocol.json')
    expected = record['protocol_sha256']
    protocol_field = record.get('protocol_field', 'protocol_sha256')
    if (saved.get(protocol_field) != expected or not protocol.is_file()
            or manifest_hash(protocol) != expected):
        raise ValueError('适配协议校验失败，请使用权重同目录下的原始 protocol.json。')
    if split not in record['manifests'].get(manifest_id, []):
        raise ValueError('该适配权重未授权评估此清单或划分。')
    return record['epoch'], dict(protocol_sha256=expected, scope=record['scope'])


def save_checkpoint(path, value):
    """同目录临时文件完成写入后再替换，避免中断损坏已有权重。"""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def run_epoch(model, loader, criterion, device, optimizer=None):
    model.train(optimizer is not None)
    loss_sum = correct = count = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(images)
            # unreduced loss allows consistent sample-averaged weighted loss.
            loss = criterion(logits, targets).mean()
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        loss_sum += loss.item() * len(targets)
        correct += (logits.argmax(1) == targets).sum().item()
        count += len(targets)
    return loss_sum / count, correct / count


def train(manifest_path, config_path, output, data_root=None, resume=False, stop_file=None, cache_images=False):
    check_splits(manifest_path)
    cfg, manifest = read_json(config_path), load_manifest(manifest_path, data_root)
    runtime_threads = int(os.environ.get('CANCER_LAB_NUM_THREADS', cfg.get('num_threads', torch.get_num_threads())))
    if not 1 <= runtime_threads <= (os.cpu_count() or 1):
        raise ValueError('Runtime thread count outside available CPU range')
    torch.set_num_threads(runtime_threads)
    if cfg['image_size'] < 16 or cfg['epochs'] < 1 or cfg['patience'] < 1 or cfg['batch_size'] < 1:
        raise ValueError('image_size >= 16，epochs/patience/batch_size 必须为正数。')
    if (not math.isfinite(cfg['learning_rate']) or cfg['learning_rate'] <= 0
            or not math.isfinite(cfg['weight_decay']) or cfg['weight_decay'] < 0
            or not 0 <= cfg['dropout'] < 1):
        raise ValueError('学习率应为正数，weight_decay 应非负，dropout 应在 [0,1) 内。')
    output = Path(output)
    resumed = None
    if resume:
        resumed = torch.load(output / 'last.pt', map_location='cpu', weights_only=True)
        previous = {k:v for k,v in resumed['config'].items() if k != 'epochs'}
        current = {k:v for k,v in cfg.items() if k != 'epochs'}
        if previous != current or not matches_manifest(resumed, manifest_path):
            raise ValueError('恢复训练需要相同配置和清单，仅允许增加 epochs。')
        if cfg['epochs'] < resumed['epoch']:
            raise ValueError('epochs 不能小于已完成轮数。')
    else:
        output.mkdir(parents=True, exist_ok=False)
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = device_for(cfg['device'])
    loaders = {split: DataLoader(CancerDataset(manifest, split, cfg['image_size'], cache_images=cache_images,
                                             normalization=cfg.get('normalization', 'symmetric')),
                                batch_size=cfg['batch_size'], shuffle=split == 'train',
                                num_workers=cfg['num_workers']) for split in ('train', 'val')}
    model = build_model(len(manifest['classes']), cfg, initialize_pretrained=resumed is None).to(device)
    counts = np.bincount([r['label'] for r in manifest['splits']['train']], minlength=len(manifest['classes']))
    weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights if cfg['class_weight'] else None, reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    write_json(output / 'config.json', cfg)
    write_json(output / 'execution.json', dict(
        runtime_num_threads=runtime_threads,
        cache_resized_images=cache_images, resume=resume,
        requested_from_epoch=resumed['epoch'] + 1 if resumed is not None else 1,
        cache_scope='deterministic RGB resize; random augmentation is sampled on every access'))
    write_json(output / 'manifest.json', manifest)
    write_json(output / 'environment.json', dict(python=platform.python_version(), torch=str(torch.__version__),
                                               device=str(device), numpy=np.__version__,
                                               packages={name:importlib.metadata.version(name) for name in
                                                         ['torch','torchvision','numpy','Pillow','scikit-learn','matplotlib','requests']}))
    (output / 'model.txt').write_text(str(model) + '\nParameters: ' + str(sum(p.numel() for p in model.parameters())), encoding='utf-8')
    best, stale, history = float('inf'), 0, []
    start_epoch = 1
    if resumed is not None:
        model.load_state_dict(resumed['state_dict'])
        optimizer.load_state_dict(resumed['optimizer'])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        best, stale, history = resumed['best_loss'], resumed['stale'], resumed['history']
        start_epoch = resumed['epoch'] + 1
        torch.set_rng_state(resumed['torch_rng'])
        random.setstate(resumed['python_rng'])
        state = resumed['numpy_rng']
        np.random.set_state((state[0], np.array(state[1], dtype=np.uint32), state[2], state[3], state[4]))
        if device.type == 'cuda' and resumed.get('cuda_rng'):
            torch.cuda.set_rng_state_all(resumed['cuda_rng'])
    stopped = 'epochs_completed'
    # 最后一次权重已写入、汇总状态尚未落盘时，恢复只补齐曲线与状态。
    if resumed is not None and stale >= cfg['patience']:
        try:
            status_epoch = read_json(output / 'training_status.json').get('epochs_completed')
        except (OSError, ValueError):
            status_epoch = None
        if status_epoch != resumed['epoch']:
            start_epoch = cfg['epochs'] + 1
            stopped = 'early_stopping'
    for epoch in range(start_epoch, cfg['epochs'] + 1):
        started = time.perf_counter()
        train_loss, train_acc = run_epoch(model, loaders['train'], criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, loaders['val'], criterion, device)
        row = dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss, train_accuracy=train_acc, val_accuracy=val_acc,
                   elapsed_seconds=time.perf_counter() - started)
        history.append(row)
        with (output / 'history.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(f'Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}', flush=True)
        if val_loss < best:
            best, stale = val_loss, 0
            save_checkpoint(output / 'best.pt', dict(state_dict=model.state_dict(), config=cfg, classes=manifest['classes'],
                            epoch=epoch, val_loss=val_loss, manifest_sha256=manifest_hash(manifest_path),
                            manifest_identity_v1=manifest_identity(manifest_path)))
        else:
            stale += 1
        numpy_rng = np.random.get_state()
        save_checkpoint(output / 'last.pt', dict(state_dict=model.state_dict(), config=cfg, classes=manifest['classes'],
                        epoch=epoch, val_loss=val_loss, manifest_sha256=manifest_hash(manifest_path),
                        manifest_identity_v1=manifest_identity(manifest_path),
                        optimizer=optimizer.state_dict(), best_loss=best, stale=stale, history=history,
                        torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                        numpy_rng=(numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]),
                        cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else []))
        if stop_file and Path(stop_file).exists():
            stopped = 'stop_requested'
            break
        if stale >= cfg['patience']:
            stopped = 'early_stopping'
            break
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ('loss', 'accuracy')):
        for split in ('train', 'val'):
            ax.plot([r['epoch'] for r in history], [r[f'{split}_{metric}'] for r in history], label=split)
        ax.set(xlabel='Epoch', ylabel=metric)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'learning_curves.png', dpi=160)
    plt.close(fig)
    result = dict(output=str(output), best_val_loss=best, epochs_completed=history[-1]['epoch'], status=stopped)
    write_json(output / 'training_status.json', result)
    return result


def load_model(checkpoint, device):
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    if 'num_threads' in saved['config']:
        torch.set_num_threads(int(os.environ.get('CANCER_LAB_NUM_THREADS', saved['config']['num_threads'])))
    model = build_model(len(saved['classes']), saved['config']).to(device)
    model.load_state_dict(saved['state_dict'])
    model.eval()
    return model, saved


def predict_loader(model, loader, device):
    """共享的有标签批量推理，保持清单顺序并返回完整类别概率。"""
    model.eval()
    actual, predicted, probabilities = [], [], []
    with torch.inference_mode():
        for images, targets in loader:
            probs = model(images.to(device)).softmax(1).cpu()
            actual.extend(targets.tolist())
            predicted.extend(probs.argmax(1).tolist())
            probabilities.extend(probs.tolist())
    return actual, predicted, probabilities


def evaluate(manifest_path, checkpoint, output, device_name='auto', split='test', data_root=None):
    if split not in ('val', 'test'):
        raise ValueError('Evaluation split must be val or test')
    check_splits(manifest_path)
    device = device_for(device_name)
    model, saved = load_model(checkpoint, device)
    manifest = load_manifest(manifest_path, data_root)
    if manifest['classes'] != saved['classes']:
        raise ValueError('测试清单的类别及顺序必须与权重一致。')
    checkpoint_digest = manifest_hash(checkpoint)
    manifest_id = manifest_identity(manifest_path)
    adaptation = {}
    if 'protocol_sha256' in saved or 'adaptation_protocol_sha256' in saved:
        epoch, adaptation = adapted_evaluation(checkpoint, saved, checkpoint_digest, manifest_id, split)
    elif not matches_manifest(saved, manifest_path):
        raise ValueError('测试清单必须与训练所用清单完全一致。')
    else:
        epoch = saved['epoch']
    dataset = CancerDataset(manifest, split, saved['config']['image_size'],
                            normalization=saved['config'].get('normalization', 'symmetric'))
    loader = DataLoader(dataset, batch_size=saved['config']['batch_size'])
    actual, predicted, probabilities = predict_loader(model, loader, device)
    classes = saved['classes']
    labels = list(range(len(classes)))
    report = classification_report(actual, predicted, labels=labels, target_names=classes, output_dict=True, zero_division=0)
    matrix = confusion_matrix(actual, predicted, labels=labels)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'evaluation_provenance.json', dict(
        recorded_utc=datetime.now(timezone.utc).isoformat(), split=split,
        checkpoint_sha256=checkpoint_digest,
        input_manifest_sha256=manifest_hash(manifest_path), manifest_identity_v1=manifest_id,
        strategy=manifest['strategy'], group_basis=manifest.get('group_basis','unspecified_legacy_manifest'),
        scope=adaptation.get('scope', manifest.get('evaluation_scope','independence requires source evidence')),
        **{key: value for key, value in adaptation.items() if key != 'scope'},
        interpretation='records this evaluation access; prior accesses and group authenticity require separate evidence'))
    write_json(output / 'metrics.json', dict(classification_report=report, confusion_matrix=matrix.tolist(), classes=classes,
                                            checkpoint_epoch=epoch, samples=len(actual), split=split))
    with (output / 'predictions.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['path', 'actual', 'predicted'] + [f'prob_{c}' for c in classes])
        for row, truth, prediction, probs in zip(dataset.rows, actual, predicted, probabilities):
            writer.writerow([row['path'], classes[truth], classes[prediction]] + probs)
    fig, ax = plt.subplots(figsize=(max(7, len(classes) * .65), max(6, len(classes) * .6)))
    ConfusionMatrixDisplay(matrix, display_labels=classes).plot(ax=ax, xticks_rotation=90, colorbar=False)
    fig.tight_layout()
    fig.savefig(output / 'confusion_matrix.png', dpi=160)
    plt.close(fig)
    return report


def predict_image(model, saved, image, device):
    with Image.open(image) as im:
        tensor = transform(saved['config']['image_size'], normalization=saved['config'].get('normalization', 'symmetric'))(im.convert('RGB')).unsqueeze(0)
    with torch.inference_mode():
        probs = model(tensor.to(device)).softmax(1)[0].cpu().tolist()
    return sorted([dict(label=label, probability=p) for label, p in zip(saved['classes'], probs)],
                  key=lambda row: row['probability'], reverse=True)


def predict(checkpoint, image, device_name='auto'):
    device = device_for(device_name)
    model, saved = load_model(checkpoint, device)
    return predict_image(model, saved, image, device)


def predict_batch(checkpoint, source, output, device_name='auto'):
    source, output = Path(source), Path(output)
    if output.suffix.lower() not in ('.json', '.csv'):
        raise ValueError('预测输出格式应为 .json 或 .csv。')
    files = sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS) if source.is_dir() else [source]
    if not files or not source.exists():
        raise ValueError('输入路径不存在或没有支持的图片。')
    if output.resolve() in {p.resolve() for p in files}:
        raise ValueError('预测输出不能覆盖输入图片。')
    device = device_for(device_name)
    model, saved = load_model(checkpoint, device)
    results = []
    for path in files:
        try:
            ranking = predict_image(model, saved, path, device)
            results.append(dict(path=str(path), status='ok', predicted=ranking[0]['label'],
                                probability=ranking[0]['probability'], top3=ranking[:3]))
        except (OSError, ValueError) as error:
            results.append(dict(path=str(path), status='error', error=str(error)))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == '.json':
        write_json(output, results)
    else:
        with output.open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=['path','status','predicted','probability','error'])
            writer.writeheader()
            writer.writerows({k:v for k,v in row.items() if k != 'top3'} for row in results)
    return dict(output=str(output), samples=len(results), errors=sum(row['status']=='error' for row in results))
