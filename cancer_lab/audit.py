"""只读数据审查：不删除、修复或移动原始图片。"""
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .data import class_folders, write_json


def audit(root, output, class_layout='leaf'):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not root.is_dir():
        raise ValueError('数据目录不存在。')
    if output == root or root in output.parents:
        raise ValueError('审查输出目录不能位于数据目录内。')
    folders = [(name, files) for name, files in class_folders(root, class_layout) if files]
    if not folders:
        raise ValueError('数据目录没有支持的图片。')
    output.mkdir(parents=True, exist_ok=False)
    rows, broken, counts = [], [], Counter()
    exact, pixels, ahashes = defaultdict(list), defaultdict(list), defaultdict(list)
    for class_index, (class_name, files) in enumerate(folders, start=1):
        thumbs = []
        for path in files:
            relative = path.relative_to(root).as_posix()
            try:
                file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                with Image.open(path) as image:
                    image.load()  # 实际解码，不能只检查文件头
                    width, height, mode = image.width, image.height, image.mode
                    rgb = image.convert('RGB')
                    pixel_hash = hashlib.sha256(f'{width}x{height}:'.encode() + rgb.tobytes()).hexdigest()
                    tiny = np.asarray(rgb.resize((8, 8)).convert('L'))
                    ahash = np.packbits(tiny >= tiny.mean()).tobytes().hex()
                    std = float(np.asarray(rgb.resize((64, 64)), dtype=float).std())
                    if len(thumbs) < 6:
                        thumbs.append((path.name, ImageOps.contain(rgb, (180, 160)).copy()))
                row = dict(path=relative, label=class_name, width=width, height=height, mode=mode,
                           bytes=path.stat().st_size, sha256=file_hash, pixel_sha256=pixel_hash,
                           average_hash=ahash, intensity_std=std, low_variance=std < 2)
                rows.append(row)
                counts[class_name] += 1
                exact[file_hash].append(relative)
                pixels[pixel_hash].append(relative)
                ahashes[ahash].append(relative)
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                broken.append(dict(path=relative, label=class_name, error=str(exc)))
        sheet = Image.new('RGB', (1200, 220), 'white')
        draw = ImageDraw.Draw(sheet)
        draw.text((10, 5), class_name, fill='black')
        for index, (name, thumb) in enumerate(thumbs):
            sheet.paste(thumb, (index * 200 + 10, 30))
            draw.text((index * 200 + 10, 195), name[:26], fill='black')
        sheet.save(output / f'samples_{class_index:02d}.jpg')
    def duplicates(mapping):
        return [paths for paths in mapping.values() if len(paths) > 1]
    duplicate_files, duplicate_pixels = duplicates(exact), duplicates(pixels)
    candidates = duplicates(ahashes)
    report = dict(root=str(root), class_layout=class_layout, total_files=sum(len(f) for _, f in folders),
                  valid_images=len(rows), corrupt_images=broken, class_counts=dict(counts),
                  dimensions=dict(Counter(f"{r['width']}x{r['height']}" for r in rows)),
                  modes=dict(Counter(r['mode'] for r in rows)),
                  exact_duplicate_groups=duplicate_files, pixel_duplicate_groups=duplicate_pixels,
                  perceptual_candidate_groups=candidates,
                  low_variance_images=[r['path'] for r in rows if r['low_variance']],
                  limitations=['Average-hash collisions are review candidates, not proof of duplication.',
                               'No patient identity or augmentation ancestry is inferred from filenames.',
                               'Statistics describe only the supplied local directory.'])
    write_json(output / 'audit.json', report)
    if rows:
        with (output / 'images.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    text = ['# 数据质量审查', '', f'扫描范围：`{root}`', '',
            f'图片总数：{report["total_files"]}；可解码：{len(rows)}；损坏：{len(broken)}。',
            f'字节重复组：{len(duplicate_files)}；像素重复组：{len(duplicate_pixels)}；感知哈希候选组：{len(candidates)}。',
            '', '感知哈希相同仅作为人工复核线索，不自动删除；未验证患者独立性。', '',
            '| 类别 | 可解码图片数 |', '|---|---:|']
    text += [f'| {name} | {counts[name]} |' for name, _ in folders]
    (output / 'report.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    return dict(output=str(output), total=report['total_files'], valid=len(rows), corrupt=len(broken),
                classes=len(folders), exact_duplicate_groups=len(duplicate_files), pixel_duplicate_groups=len(duplicate_pixels),
                perceptual_candidate_groups=len(candidates))
