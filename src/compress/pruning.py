"""Global unstructured magnitude pruning with explicit, persisted masks.

Why explicit masks instead of torch.nn.utils.prune: the reparametrisation it
installs (weight_orig + weight_mask forward hooks) leaks into checkpoints,
ONNX export and quantisation. Here a mask is a plain bool tensor per prunable
weight; the model keeps ordinary parameters, zeros are re-imposed with
apply_masks() after every optimizer step, and the masks travel inside the
checkpoint under the key "prune_masks".

Scope: weights of every Conv2d and Linear. Biases and BatchNorm are excluded
(negligible parameter count; pruning them destabilises training).
"""

from __future__ import annotations

import torch
import torch.nn as nn

PRUNABLE = (nn.Conv2d, nn.Linear)


def prunable_weights(model: nn.Module, exclude: tuple[str, ...] = ()) -> dict[str, torch.Tensor]:
    """{param_name: weight} for every Conv2d / Linear weight, in module order.
    `exclude` lists module names left dense (e.g. a randomly initialised head,
    whose magnitudes carry no information)."""
    return {f"{name}.weight": m.weight for name, m in model.named_modules()
            if isinstance(m, PRUNABLE) and name not in exclude}


def global_magnitude_masks(model: nn.Module, sparsity: float,
                           exclude: tuple[str, ...] = ()) -> dict[str, torch.Tensor]:
    """Masks that zero the `sparsity` fraction of smallest-|w| prunable weights,
    with one global threshold across all layers (L1 magnitude)."""
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    weights = prunable_weights(model, exclude)
    if sparsity == 0.0:
        return {k: torch.ones_like(w, dtype=torch.bool) for k, w in weights.items()}

    with torch.no_grad():
        flat = torch.cat([w.detach().abs().flatten().float() for w in weights.values()])
        k = int(round(sparsity * flat.numel()))
        # kthvalue is the k-th smallest; everything <= it is pruned. Ties at the
        # threshold are resolved by ranking, so the count is exact.
        order = torch.argsort(flat)
        keep = torch.ones_like(flat, dtype=torch.bool)
        keep[order[:k]] = False
        masks, offset = {}, 0
        for name, w in weights.items():
            n = w.numel()
            masks[name] = keep[offset:offset + n].view_as(w).to(w.device)
            offset += n
    return masks


@torch.no_grad()
def apply_masks(model: nn.Module, masks: dict[str, torch.Tensor]) -> None:
    """Re-impose zeros in place. Call after every optimizer step."""
    params = dict(model.named_parameters())
    for name, mask in masks.items():
        p = params[name]
        p.mul_(mask.to(p.device, p.dtype))


def sparsity_report(model: nn.Module, masks: dict[str, torch.Tensor] | None = None,
                    exclude: tuple[str, ...] = ()) -> dict:
    """Achieved sparsity from the actual zeros in the weights (not the masks),
    globally over prunable weights, over all parameters, and per layer."""
    weights = prunable_weights(model, exclude)
    per_layer, zeros, total = [], 0, 0
    for name, w in weights.items():
        n = w.numel()
        z = int((w == 0).sum().item())
        zeros += z
        total += n
        row = {"name": name, "numel": n, "zeros": z, "sparsity": z / n}
        if masks is not None:
            row["mask_sparsity"] = 1.0 - masks[name].float().mean().item()
        per_layer.append(row)
    all_total = sum(p.numel() for p in model.parameters())
    all_zeros = sum(int((p == 0).sum().item()) for p in model.parameters())
    return {"sparsity_prunable": zeros / total, "prunable_numel": total, "prunable_zeros": zeros,
            "sparsity_all_params": all_zeros / all_total, "params_total": all_total,
            "params_nonzero": all_total - all_zeros, "per_layer": per_layer}


def masks_to_cpu(masks: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().to("cpu", torch.bool) for k, v in masks.items()}


def masks_to_device(masks: dict[str, torch.Tensor], device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in masks.items()}
