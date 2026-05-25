"""Paired Mode 1/2 forward helper for the physics consistency loss."""

from __future__ import annotations

import torch


def add_paired_physics_predictions(
    model: torch.nn.Module,
    pred: dict,
    batch: dict,
    *,
    use_amp: bool,
) -> dict:
    """Attach same-sample Mode 1/2 predictions for D_dt consistency.

    Units are handled inside ``ml.training.losses``. SIE 표준 근사 가정:
    the helper only asks the existing model heads for Mode 1 H0 correction and
    Mode 2 SIE parameter correction; it does not add a new approximation switch.
    """

    required = {"h0_approx", "dt_lc", "theta_E_approx", "dphi_sie_rad2"}
    if not required.issubset(batch):
        return pred

    device = batch["lc"].device
    batch_size = int(batch["lc"].shape[0])
    target_mode_1 = torch.ones(batch_size, dtype=torch.long, device=device)
    target_mode_2 = torch.full((batch_size,), 2, dtype=torch.long, device=device)

    with torch.amp.autocast("cuda", enabled=use_amp):
        out1 = model(
            lc=batch["lc"],
            lc_mask=batch["lc_mask"],
            params=batch["params"],
            sigma_curve=batch["sigma_curve"],
            target_mode=target_mode_1,
        )
        out2 = model(
            lc=batch["lc"],
            lc_mask=batch["lc_mask"],
            params=batch["params"],
            sigma_curve=batch["sigma_curve"],
            target_mode=target_mode_2,
        )

    pred["physics"] = {"mode1": out1.get("mode1"), "mode2": out2.get("mode2")}
    return pred
