import yaml
import argparse
import os
import re
import numpy as np
from utils.dataloader import *
from utils.tools import *
from utils.logger import get_result_name
from supervisor import (
    EIN_BiGCN_supervisor,
    EIN_BiGCN_Uncertainty_supervisor,
    EIN_ResGCN_supervisor,
    EIN_ResGCN_Uncertainty_supervisor,
    EIN_LIRS_supervisor,
    EIN_BiGCN_StateAuxSameDiff_supervisor,
    EIN_ResGCN_StateAuxSameDiff_supervisor,
    EIN_RAGCL_BiGCN_supervisor,
    EIN_RAGCL_ResGCN_supervisor,
    EIN_Plain_BiGCN_supervisor,
    EIN_Plain_ResGCN_supervisor,
)


def _safe_filename_part(value):
    value = str(value).strip()
    value = re.sub(r'[^A-Za-z0-9_.-]+', '_', value)
    value = value.strip('._-')
    return value or 'unknown'


def _selection_metric_part(args):
    metric = getattr(args, 'selection_metric', 'val_loss')
    if metric is None:
        metric = 'val_loss'
    return _safe_filename_part(metric)


def _summary_model_parts(args):
    base_model = str(getattr(args, 'base_model', 'unknown')).strip()

    if base_model.startswith('Plain_'):
        return 'Base', base_model[len('Plain_'):]
    if base_model.startswith('RAGCL_'):
        return 'Ragcl', base_model[len('RAGCL_'):]
    if 'StateAuxSameDiff' in base_model:
        return 'Ours', base_model.replace('_StateAuxSameDiff', '')
    if base_model == 'LIRS':
        return 'LIRS', None

    return str(getattr(args, 'model_name', 'Model')).strip(), base_model


def build_summary_filename(args):
    summary_name = getattr(args, 'summary_name', None)
    if summary_name is not None and str(summary_name).strip():
        summary_name = _safe_filename_part(summary_name)
        if not summary_name.endswith('.txt'):
            summary_name += '.txt'
        return summary_name

    method, backbone = _summary_model_parts(args)
    parts = [method]
    if backbone:
        parts.append(backbone)

    if hasattr(args, 'undirected'):
        parts.append('undirected' if getattr(args, 'undirected') else 'directed')

    parts.append(_selection_metric_part(args))

    embedding = str(getattr(args, 'word_embedding', 'unknown')).strip()
    if embedding == 'multilingual-e5-base':
        embedding = 'e5'
    parts.append(embedding)

    safe_parts = [_safe_filename_part(part) for part in parts]
    return 'summary_{}.txt'.format('_'.join(safe_parts))


def summarize_results(results, args):
    metrics = ['acc', 'auc', 'f1']
    lines = []

    lines.append('Experiment setting:')
    lines.append('Target dataset: {}'.format(args.dataset))
    lines.append('Mode: {}'.format(getattr(args, 'experiment_mode', 'id')))
    lines.append('OOD source datasets: {}'.format(getattr(args, 'ood_source_datasets', [])))
    lines.append('Validation domain: {}'.format(getattr(args, 'ood_val_domain', 'source')))
    lines.append('Checkpoint selection metric: {}'.format(getattr(args, 'selection_metric', 'val_loss')))
    lines.append('')

    lines.append('Seed results:')
    for result in results:
        lines.append(
            'Seed {seed}: Acc {acc:.4f} | AUC {auc:.4f} | F1 {f1:.4f}'.format(**result)
        )

    lines.append('')
    lines.append('Average results over {} runs:'.format(len(results)))
    for metric in metrics:
        values = np.array([result[metric] for result in results])
        lines.append(
            '{}: {:.2f}+/-{:.2f} (%)'.format(
                metric.upper(), values.mean() * 100, values.std() * 100
            )
        )

    summary = '\n'.join(lines)
    print(summary)

    result_name = get_result_name(args)
    summary_dir = os.path.join('experiments', args.model_name, args.dataset)
    if result_name:
        summary_dir = os.path.join(summary_dir, result_name)
    os.makedirs(summary_dir, exist_ok=True)
    summary_filename = (
        'summary_{}.txt'.format(_selection_metric_part(args))
        if result_name
        else build_summary_filename(args)
    )
    summary_path = os.path.join(summary_dir, summary_filename)
    with open(summary_path, 'w', encoding='utf-8') as file_obj:
        file_obj.write(summary + '\n')
    print('Summary saved to: {}'.format(summary_path))


if __name__ == '__main__':

    dataset = 'DRWeibo'
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_filename', default='configs/EIN/' + dataset +'.yaml', 
                    type=str, help='the configuration to use')
    args = parser.parse_args()

    print(f'Starting experiment with configurations in {args.config_filename}...')
    
    configs = yaml.load(
        open(args.config_filename), 
        Loader=yaml.FullLoader
    )
    
    args = argparse.Namespace(**configs)

    results = []
    supervisor = globals()['EIN_' + args.base_model + '_supervisor']
    for i in range(5):
        args.seed = i
        result = supervisor(args)
        if result is not None:
            result['seed'] = i
            results.append(result)

    if results:
        summarize_results(results, args)
