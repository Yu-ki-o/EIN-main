"""Average validation-selected checkpoints from one training trajectory."""

from pathlib import Path

import torch


DEFAULT_TARGETS = dict(acc=.83, auc=.815, f1=.76)
AVERAGE_SELECTORS = ('loss', 'acc', 'auc', 'f1', 'target_score')


def average_checkpoints(directory, metadata, selectors, device):
    """Give each unique epoch equal weight, preserving state keys and shapes."""
    directory = Path(directory)
    sources, epochs, states = [], set(), []
    for selector in selectors:
        if selector not in AVERAGE_SELECTORS:
            raise ValueError(f'Unknown checkpoint averaging selector: {selector}')
        epoch = metadata['best'][selector]['epoch']
        if epoch in epochs:
            continue
        epochs.add(epoch)
        path = directory / f'best_val_{selector}.pth'
        states.append(torch.load(path, weights_only=True, map_location=device))
        sources.append(dict(selector=selector, epoch=epoch, checkpoint=str(path)))
    if not states or any(state.keys() != states[0].keys() for state in states):
        raise ValueError('Checkpoint parameter keys do not match')
    result = {}
    for key, value in states[0].items():
        if value.is_floating_point():
            result[key] = torch.stack([state[key] for state in states]).mean(0)
        else:
            if any(not torch.equal(state[key], value) for state in states):
                raise ValueError(f'Nonfloating buffer differs: {key}')
            result[key] = value.clone()
    return result, sources
