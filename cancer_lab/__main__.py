import argparse
import json

from .data import prepare
from .engine import train, evaluate, predict, predict_batch
from .audit import audit
from .split_check import check_splits


def main():
    parser = argparse.ArgumentParser(description='多类型癌症分析：数据准备、训练、测试、单图预测')
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('check-splits')
    p.add_argument('--manifest', required=True)
    p.add_argument('--output')
    p = commands.add_parser('prepare')
    p.add_argument('--root', default='data/raw')
    p.add_argument('--output', default='data/splits/manifest.json')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val-ratio', type=float, default=.15)
    p.add_argument('--test-ratio', type=float, default=.15)
    p.add_argument('--groups-csv', help='可选 CSV：path,group_id；path 为相对数据根目录路径')
    p.add_argument('--class-layout', choices=['top', 'leaf'], default='top')
    p = commands.add_parser('audit')
    p.add_argument('--root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--class-layout', choices=['top', 'leaf'], default='leaf')
    p = commands.add_parser('train')
    p.add_argument('--manifest', default='data/splits/manifest.json')
    p.add_argument('--config', default='configs/baseline.json')
    p.add_argument('--output', default='outputs/baseline')
    p.add_argument('--data-root')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--cache-images', action='store_true', help='缓存缩放后的 RGB 图片，增强仍在每次训练访问时采样')
    p.add_argument('--stop-file', help='检测到文件存在后，在完成当前 epoch 后保存并停止')
    p = commands.add_parser('evaluate')
    p.add_argument('--manifest', default='data/splits/manifest.json')
    p.add_argument('--checkpoint', default='outputs/baseline/best.pt')
    p.add_argument('--output', default='outputs/baseline/test')
    p.add_argument('--split', choices=['val', 'test'], default='test')
    p.add_argument('--device', default='auto')
    p.add_argument('--data-root')
    p = commands.add_parser('predict')
    p.add_argument('--checkpoint', default='outputs/baseline/best.pt')
    p.add_argument('--image', required=True)
    p.add_argument('--device', default='auto')
    p = commands.add_parser('predict-batch')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--source', required=True, help='单张图片或递归扫描的图片目录')
    p.add_argument('--output', required=True, help='.json 或 .csv')
    p.add_argument('--device', default='auto')
    args = parser.parse_args()
    if args.command == 'check-splits':
        result = check_splits(args.manifest, args.output)
    elif args.command == 'prepare':
        result = prepare(args.root, args.output, args.seed, args.val_ratio, args.test_ratio, args.groups_csv, args.class_layout)
        if not args.groups_csv:
            print('提示：当前按图片分层划分；正式医学实验应审核患者/切片关联并提供分组表。')
    elif args.command == 'audit':
        result = audit(args.root, args.output, args.class_layout)
    elif args.command == 'train':
        result = train(args.manifest, args.config, args.output, args.data_root, args.resume, args.stop_file, args.cache_images)
    elif args.command == 'evaluate':
        result = evaluate(args.manifest, args.checkpoint, args.output, args.device, args.split, args.data_root)
    elif args.command == 'predict-batch':
        result = predict_batch(args.checkpoint, args.source, args.output, args.device)
    else:
        result = predict(args.checkpoint, args.image, args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.command == 'predict-batch' and result['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
