"""生成固定划分清单；增强只作用于训练集。"""
import csv
import hashlib
import json
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from torch.utils.data import Dataset
from torchvision import transforms

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_manifest(path, data_root=None):
    """路径迁移只覆盖运行时根目录，保留原始清单及权重关联哈希。"""
    manifest = read_json(path)
    if data_root is not None:
        manifest['root'] = str(Path(data_root).resolve())
    return manifest


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def class_folders(root, layout='top'):
    """top: 首层目录为类别；leaf: 图片所在目录的相对路径为类别。"""
    root = Path(root)
    if layout == 'leaf':
        folders = sorted({p.parent for p in root.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS})
        return [(p.relative_to(root).as_posix(), sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() in EXTENSIONS)) for p in folders]
    if layout != 'top':
        raise ValueError('class_layout must be top or leaf')
    return [(p.name, sorted(f for f in p.rglob('*') if f.is_file() and f.suffix.lower() in EXTENSIONS)) for p in sorted(root.iterdir()) if p.is_dir()]


def prepare(root, output, seed=42, val_ratio=0.15, test_ratio=0.15, groups_csv=None, class_layout='top'):
    root, output = Path(root).resolve(), Path(output)
    if output.exists():
        raise ValueError('划分文件已存在，请使用新的输出路径，避免覆盖实验划分。')
    if not root.is_dir() or not 0 < val_ratio + test_ratio < 1 or min(val_ratio, test_ratio) <= 0:
        raise ValueError('请提供有效数据目录和正数验证/测试比例，其和必须小于 1。')
    group_map = {}
    if groups_csv:
        with open(groups_csv, encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                key = row['path'].replace('\\', '/')
                if key in group_map or not row['group_id'].strip():
                    raise ValueError('分组表存在重复路径或空 group_id。')
                group_map[key] = row['group_id'].strip()
    records, hashes, classes = [], {}, []
    for class_name, files in class_folders(root, class_layout):
        if not files:
            continue
        label = len(classes)
        classes.append(class_name)
        for path in files:
            relative = path.relative_to(root).as_posix()
            with Image.open(path) as im:
                im.verify()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in hashes:
                raise ValueError(f'发现完全重复图片，请先审核去重：{relative} 与 {hashes[digest]}')
            hashes[digest] = relative
            if groups_csv and relative not in group_map:
                raise ValueError(f'分组表缺少：{relative}')
            records.append(dict(path=relative, label=label, sha256=digest,
                                group=group_map.get(relative, relative)))
    if len(classes) < 2:
        raise ValueError('至少需要两个类别；目录格式为 data/raw/类别/图片。')
    indices = list(range(len(records)))
    if groups_csv:
        # 患者/切片分组优先于类别比例；任何组不能跨训练、验证、测试集。
        def split(ids, fraction, random_state):
            splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=random_state)
            left, right = next(splitter.split(ids, groups=[records[i]['group'] for i in ids]))
            return [ids[i] for i in left], [ids[i] for i in right]
    else:
        def split(ids, fraction, random_state):
            return train_test_split(ids, test_size=fraction, random_state=random_state,
                                    stratify=[records[i]['label'] for i in ids])
    train_val, test = split(indices, test_ratio, seed)
    train, val = split(train_val, val_ratio / (1 - test_ratio), seed + 1)
    splits = {}
    for name, ids in [('train', train), ('val', val), ('test', test)]:
        splits[name] = [records[i] for i in ids]
        if {r['label'] for r in splits[name]} != set(range(len(classes))):
            raise ValueError(f'{name} 缺少类别，请增加样本/患者组或调整划分方案。')
    result = dict(root=str(root), classes=classes, seed=seed, class_layout=class_layout,
                  group_basis='provided_csv' if groups_csv else 'image_path',
                  strategy='group' if groups_csv else 'stratified_image', splits=splits)
    write_json(output, result)
    return {k: len(v) for k, v in splits.items()}


def transform(image_size, training=False, normalization='symmetric'):
    if normalization not in ('symmetric', 'imagenet'):
        raise ValueError('Unknown normalization')
    mean, std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) if normalization == 'imagenet' else ([0.5]*3, [0.5]*3)
    steps = [transforms.Resize((image_size, image_size))]
    if training:
        steps += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)]
    return transforms.Compose(steps + [transforms.ToTensor(),
                                      transforms.Normalize(mean, std)])


class CancerDataset(Dataset):
    def __init__(self, manifest, split, image_size, verify=True, cache_images=False, normalization='symmetric'):
        self.rows = manifest['splits'][split]
        self.root = Path(manifest['root'])
        self.transform = transform(image_size, split == 'train', normalization)
        self._resized_images = {} if cache_images else None
        self._resize = transforms.Resize((image_size, image_size))
        if verify:
            for row in self.rows:
                if hashlib.sha256((self.root / row['path']).read_bytes()).hexdigest() != row['sha256']:
                    raise ValueError(f"图片发生变化，请重新准备实验：{row['path']}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if self._resized_images is not None:
            if index not in self._resized_images:
                with Image.open(self.root / row['path']) as im:
                    self._resized_images[index] = self._resize(im.convert('RGB'))
            # 只缓存确定性的 RGB/缩放结果，每次访问仍重新采样训练增强。
            return self.transform(self._resized_images[index]), row['label']
        with Image.open(self.root / row['path']) as im:
            tensor = self.transform(im.convert('RGB'))
        return tensor, row['label']
