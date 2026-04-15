from __future__ import annotations

import torch


def _safe_mean(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_tensor(0.0)
    return values.mean()


def _sorted_second_difference(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.numel() < 3:
        return y.new_tensor(0.0)
    order = torch.argsort(x.view(-1))
    y_sorted = y.view(-1)[order]
    second_diff = y_sorted[2:] - (2.0 * y_sorted[1:-1]) + y_sorted[:-2]
    return _safe_mean(second_diff.square())


def compute_physics_regularization(
    outputs: dict,
    max_temp_derivative: float = 2e-2,
    short_range_factor: float = 1.0,
    long_range_fraction: float = 0.75,
) -> tuple[torch.Tensor, dict[str, float]]:
    physics_aux = outputs.get("physics_aux", {})
    device = outputs["log_gamma"].device

    zero = torch.tensor(0.0, device=device)
    total = zero
    terms = {
        "smoothness": 0.0,
        "short_range_repulsion": 0.0,
        "long_range_decay": 0.0,
        "charge_sign_consistency": 0.0,
        "thermo_derivative": 0.0,
    }

    for prefix in ("cross", "cross_refine"):
        dist = physics_aux.get(f"{prefix}_dist")
        pred_energy = physics_aux.get(f"{prefix}_predicted_energy")
        lj_energy = physics_aux.get(f"{prefix}_lj_energy")
        charge_product = physics_aux.get(f"{prefix}_charge_product")
        charge_response = physics_aux.get(f"{prefix}_charge_response")
        sigma = physics_aux.get(f"{prefix}_sigma")
        cutoff = physics_aux.get(f"{prefix}_cutoff")

        if dist is None:
            continue

        smoothness = _sorted_second_difference(dist, pred_energy)
        short_mask = dist < (short_range_factor * sigma)
        short_repulsion = _safe_mean(torch.relu(-lj_energy[short_mask])) if short_mask.any() else zero
        long_mask = dist > (long_range_fraction * cutoff)
        long_decay = _safe_mean(pred_energy[long_mask].abs()) if long_mask.any() else zero
        charge_consistency = _safe_mean(torch.relu(-(charge_product * charge_response)))

        total = total + smoothness + short_repulsion + long_decay + charge_consistency
        terms["smoothness"] += float(smoothness.detach().cpu())
        terms["short_range_repulsion"] += float(short_repulsion.detach().cpu())
        terms["long_range_decay"] += float(long_decay.detach().cpu())
        terms["charge_sign_consistency"] += float(charge_consistency.detach().cpu())

    k2 = outputs.get("k2")
    if k2 is not None:
        temp_k = outputs["log_gamma"].new_tensor(273.15)
        thermo_derivative = torch.abs(k2.view(-1) / (temp_k**2))
        derivative_penalty = _safe_mean(torch.relu(thermo_derivative - max_temp_derivative))
        total = total + derivative_penalty
        terms["thermo_derivative"] = float(derivative_penalty.detach().cpu())

    return total, terms
