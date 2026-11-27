"""生成随机合成图，验证完整流程；结果不能作为医学模型表现。"""
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cancer_lab.data import prepare, read_json, write_json
from cancer_lab.engine import train, evaluate, predict


def main():
    torch.set_num_threads(2)
    output = Path('outputs') / ('smoke_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    rng = np.random.default_rng(42)
    group_rows = []
    for label in range(3):
        folder = output / 'synthetic' / f'synthetic_class_{label}'
        folder.mkdir(parents=True)
        for index in range(24):
            pixels = rng.integers(0, 90, (32, 32, 3), dtype=np.uint8)
            pixels[:, :, label] += 120
            Image.fromarray(pixels).save(folder / f'{index}.png')
            group_rows.append([f'synthetic_class_{label}/{index}.png', f'group_{index // 2}'])
    manifest_path = output / 'manifest.json'
    prepare(output / 'synthetic', manifest_path, val_ratio=.2, test_ratio=.2)
    manifest = read_json(manifest_path)
    sets = [set(r['sha256'] for r in rows) for rows in manifest['splits'].values()]
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i))
    with (output / 'groups.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['path', 'group_id'])
        writer.writerows(group_rows)
    prepare(output / 'synthetic', output / 'group_manifest.json', groups_csv=output / 'groups.csv')
    grouped = read_json(output / 'group_manifest.json')
    groups = [set(r['group'] for r in rows) for rows in grouped['splits'].values()]
    assert all(not groups[i] & groups[j] for i in range(3) for j in range(i))
    cfg = read_json('configs/baseline.json')
    cfg.update(image_size=32, batch_size=8, epochs=2, num_workers=0, device='cpu')
    write_json(output / 'config.json', cfg)
    train(manifest_path, output / 'config.json', output / 'run')
    metrics = evaluate(manifest_path, output / 'run/best.pt', output / 'test', 'cpu')
    result = predict(output / 'run/best.pt', output / 'synthetic/synthetic_class_0/0.png', 'cpu')
    assert len(result) == 3 and abs(sum(r['probability'] for r in result) - 1) < 1e-5
    assert int(metrics['macro avg']['support']) == len(manifest['splits']['test'])
    write_json(output / 'SMOKE_ONLY.json', dict(status='passed', purpose='synthetic pipeline verification only'))
    print(f'SMOKE PASSED: {output.resolve()}')


if __name__ == '__main__':
    main()
