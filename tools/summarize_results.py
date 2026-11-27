"""Validate aggregate experiment records and regenerate portfolio figures."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
SHORT_LABELS = {
    'all_benign': 'Non-ALL',
    'all_early': 'Early pre-B ALL',
    'all_pre': 'Pre-B ALL',
    'all_pro': 'Pro-B ALL',
    'kidney_normal': 'Kidney normal',
    'kidney_tumor': 'Kidney tumor',
    'lymph_cll': 'Lymphoma CLL',
    'lymph_fl': 'Lymphoma FL',
    'lymph_mcl': 'Lymphoma MCL',
    'oral_normal': 'Oral normal',
    'oral_scc': 'Oral SCC',
}
BLUE = '#28638A'
ORANGE = '#AE562B'


def read_run(name):
    result = json.loads((ROOT / 'results' / f'{name}_test_metrics.json').read_text(encoding='utf-8'))
    with (ROOT / 'results' / f'{name}_history.csv').open(encoding='utf-8-sig', newline='') as stream:
        history = list(csv.DictReader(stream))
    matrix = np.asarray(result['confusion_matrix'], dtype=int)
    labels = result['classes']
    if matrix.shape != (len(labels), len(labels)) or (matrix < 0).any() or matrix.sum() != result['samples']:
        raise ValueError(f'{name}: invalid confusion matrix or sample count')
    support = matrix.sum(axis=1)
    positives = matrix.diagonal()
    denominator = support + matrix.sum(axis=0)
    class_f1 = np.divide(2 * positives, denominator, out=np.zeros(len(labels)), where=denominator > 0)
    accuracy = np.trace(matrix) / matrix.sum()
    report = result['classification_report']
    if not np.allclose(class_f1, [report[label]['f1-score'] for label in labels]) or not np.isclose(
        accuracy, report['accuracy']
    ) or not np.isclose(class_f1.mean(), report['macro avg']['f1-score']):
        raise ValueError(f'{name}: recorded metrics differ from the confusion matrix')
    lowest_loss = min(history, key=lambda row: float(row['val_loss']))
    if int(lowest_loss['epoch']) != result['checkpoint_epoch']:
        raise ValueError(f'{name}: selected epoch does not match validation loss')
    return labels, support, class_f1, accuracy, history, result['checkpoint_epoch']


def save(fig, output, filename):
    fig.savefig(output / filename, dpi=170, facecolor='white')
    plt.close(fig)


def plot_class_comparison(output, labels, support, cnn_f1, resnet_f1):
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.8, 5.4), layout='constrained')
    ax.barh(y - .18, cnn_f1, height=.32, color=BLUE, label='CancerCNN')
    ax.barh(y + .18, resnet_f1, height=.32, color=ORANGE, label='ResNet-18')
    ax.set(yticks=y, yticklabels=[f'{name} (n={n})' for name, n in zip(labels, support)],
           xlim=(0, 1), xlabel='Class F1', title='Two architectures on the same source-grouped test split')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=.22)
    ax.legend(loc='lower right')
    save(fig, output, 'per_class_f1.png')


def plot_training(output, runs):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), layout='constrained')
    for row, (name, history, selected_epoch) in enumerate(runs):
        epochs = [int(record['epoch']) for record in history]
        for column, (key, label) in enumerate((('loss', 'Weighted cross-entropy'), ('accuracy', 'Accuracy'))):
            ax = axes[row, column]
            ax.plot(epochs, [float(record[f'train_{key}']) for record in history],
                    color=BLUE, marker='o', markersize=3, label='Train')
            ax.plot(epochs, [float(record[f'val_{key}']) for record in history],
                    color=ORANGE, linestyle='--', marker='s', markersize=3, label='Validation')
            ax.axvline(selected_epoch, color='#67747E', linestyle=':', label=f'Selected epoch {selected_epoch}')
            ax.set(xticks=epochs, xlabel='Epoch', ylabel=label, title=f'{name} · {label}')
            ax.grid(axis='y', alpha=.22)
            ax.legend(frameon=False, fontsize=8)
            if key == 'accuracy':
                ax.set_ylim(0, 1)
                ax.yaxis.set_major_formatter(PercentFormatter(1))
    save(fig, output, 'training_comparison.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'portfolio_figures')
    args = parser.parse_args()
    labels, support, cnn_f1, cnn_acc, cnn_history, cnn_epoch = read_run('cancer_cnn')
    baseline_labels, baseline_support, resnet_f1, resnet_acc, resnet_history, resnet_epoch = read_run('resnet18')
    if baseline_labels != labels or not np.array_equal(baseline_support, support):
        raise ValueError('The recorded test runs disagree on label order or class support')
    display = [SHORT_LABELS[label.rsplit('/', 1)[-1]] for label in labels]
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.axisbelow': True})
    plot_class_comparison(args.output, display, support, cnn_f1, resnet_f1)
    plot_training(args.output, [('CancerCNN', cnn_history, cnn_epoch), ('ResNet-18', resnet_history, resnet_epoch)])
    with (args.output / 'per_class_f1.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['class', 'test_images', 'cancercnn_f1', 'resnet18_f1'])
        for label, count, cnn, resnet in zip(labels, support, cnn_f1, resnet_f1):
            writer.writerow([label, count, f'{cnn:.6f}', f'{resnet:.6f}'])
    print(f'CancerCNN: accuracy {cnn_acc:.2%}, macro F1 {cnn_f1.mean():.4f}, selected epoch {cnn_epoch}')
    print(f'ResNet-18: accuracy {resnet_acc:.2%}, macro F1 {resnet_f1.mean():.4f}, selected epoch {resnet_epoch}')


if __name__ == '__main__':
    main()
