"""Unit tests of the pruning primitives (CPU, small model)."""

import io

import pytest
import torch
import torch.nn as nn

from compress.pruning import (apply_masks, global_magnitude_masks, masks_to_cpu,
                              prunable_weights, sparsity_report)


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Linear(32, 10))


@pytest.mark.parametrize("target", [0.5, 0.7, 0.9])
def test_global_sparsity_matches_target(target):
    m = tiny_model()
    masks = global_magnitude_masks(m, target)
    apply_masks(m, masks)
    rep = sparsity_report(m, masks)
    assert abs(rep["sparsity_prunable"] - target) < 1e-3          # ±0,1 p.p.
    assert set(masks) == set(prunable_weights(m))
    assert all("bn" not in k and "bias" not in k for k in masks)


def test_threshold_is_global_not_per_layer():
    m = tiny_model()
    masks = global_magnitude_masks(m, 0.7)
    per = [1 - v.float().mean().item() for v in masks.values()]
    assert max(per) - min(per) > 0.05        # layers differ -> one global threshold


def test_smallest_magnitudes_are_the_ones_removed():
    m = tiny_model()
    masks = global_magnitude_masks(m, 0.5)
    w = m[0].weight.detach().abs()
    kept, removed = w[masks["0.weight"]], w[~masks["0.weight"]]
    assert removed.max() <= kept.min() + 1e-12


def test_masks_survive_an_optimizer_step():
    m = tiny_model()
    masks = global_magnitude_masks(m, 0.6)
    apply_masks(m, masks)
    opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    for _ in range(3):
        opt.zero_grad()
        m(torch.randn(4, 3, 8, 8)).sum().backward()
        opt.step()                       # gradients flow into masked weights ...
        apply_masks(m, masks)            # ... and are zeroed again
    rep = sparsity_report(m, masks)
    assert abs(rep["sparsity_prunable"] - 0.6) < 1e-3
    for k, mask in masks.items():
        assert (dict(m.named_parameters())[k][~mask] == 0).all()


def test_masks_roundtrip_through_checkpoint():
    m = tiny_model()
    masks = global_magnitude_masks(m, 0.5)
    apply_masks(m, masks)
    buf = io.BytesIO()
    torch.save({"model": m.state_dict(), "prune_masks": masks_to_cpu(masks)}, buf)
    buf.seek(0)
    ck = torch.load(buf, weights_only=False)
    m2 = tiny_model(seed=1)
    m2.load_state_dict(ck["model"])
    for k, mask in ck["prune_masks"].items():
        assert torch.equal(mask, masks[k].cpu())
    x = torch.randn(2, 3, 8, 8)
    assert torch.allclose(m(x), m2(x))
    assert abs(sparsity_report(m2)["sparsity_prunable"] - 0.5) < 1e-3


def test_zero_sparsity_is_identity():
    m = tiny_model()
    before = [p.clone() for p in m.parameters()]
    masks = global_magnitude_masks(m, 0.0)
    apply_masks(m, masks)
    assert all(torch.equal(a, b) for a, b in zip(before, m.parameters()))


def test_invalid_sparsity():
    with pytest.raises(ValueError):
        global_magnitude_masks(tiny_model(), 1.0)
