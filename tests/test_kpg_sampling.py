"""Exercise the real sampling helper without requiring PyG to be installed."""
import ast
from pathlib import Path

import pytest
import torch


def _load_sampler():
    path = Path(__file__).resolve().parents[1] / 'model' / 'kpg.py'
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == '_sample_with_replacement')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['_sample_with_replacement']


sample = _load_sampler()


@pytest.fixture(autouse=True)
def deterministic_mode():
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous, warn_only=warn_only)


@pytest.mark.parametrize('device', ['cpu', pytest.param(
    'cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_multiple_draws_reproducible_and_weighted(device):
    weights = torch.tensor([0., 1., 3., 0.], device=device)
    generator = torch.Generator(device=device).manual_seed(123)
    saved = generator.get_state()
    first = sample(weights, 10000, generator)
    generator.set_state(saved)
    assert torch.equal(first, sample(weights, 10000, generator))
    assert first.device == weights.device
    assert first.dtype == torch.long
    assert bool(((first == 1) | (first == 2)).all())
    assert abs(float((first == 2).float().mean()) - .75) < .03
    assert torch.are_deterministic_algorithms_enabled()


def test_multi_draw_strict_path_does_not_call_multinomial(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError('Strict multi-draw must avoid CUDA multinomial.')
    monkeypatch.setattr(torch, 'multinomial', fail)
    assert sample(torch.tensor([0., 1., 0.]), 6).tolist() == [1] * 6


def test_original_single_draw_path_is_unchanged():
    weights = torch.tensor([1., 2.])
    expected = torch.multinomial(weights, 1, replacement=True,
                                 generator=torch.Generator().manual_seed(17))
    actual = sample(weights, 1, torch.Generator().manual_seed(17))
    assert torch.equal(expected, actual)


@pytest.mark.parametrize('weights', [[0., 0.], [-1., 2.], [float('nan'), 1.]])
def test_invalid_weights_fail_clearly(weights):
    with pytest.raises(ValueError):
        sample(torch.tensor(weights), 3)
