"""Conditional mixture posterior for an H0-blind Fermat-potential ratio."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from ml.models.encoders import ImageEncoder, ParamEncoder


class FermatRatioPosterior(nn.Module):
    """p(log(dphi_truth/dphi_SIE) | image, geometry, SIE structure)."""
    def __init__(self, d_model: int = 128, components: int = 5, dropout: float = .1) -> None:
        super().__init__()
        self.components = components
        self.param_encoder = ParamEncoder(5, d_model, dropout)
        self.image_encoder = ImageEncoder(d_model, dropout)
        self.geometry_encoder = nn.Sequential(nn.Linear(4, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model), nn.GELU())
        self.head = nn.Sequential(nn.Linear(d_model * 3, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, components * 3))

    def forward(self, *, params: torch.Tensor, geometry: torch.Tensor, image: torch.Tensor) -> dict[str, torch.Tensor]:
        h = torch.cat([self.param_encoder(params), self.geometry_encoder(geometry), self.image_encoder(image)], dim=-1)
        raw = self.head(h).view(-1, self.components, 3)
        return {"logits": raw[..., 0], "mean": raw[..., 1], "log_scale": raw[..., 2].clamp(-7., 3.)}


def mixture_nll(pred: dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    x = target.float().unsqueeze(-1)
    scale = pred["log_scale"].exp()
    log_pdf = -.5 * ((x - pred["mean"]) / scale).square() - pred["log_scale"] - .5 * math.log(2. * math.pi)
    return -torch.logsumexp(torch.log_softmax(pred["logits"], -1) + log_pdf, dim=-1).mean()


def posterior_mean_std(pred: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    w = torch.softmax(pred["logits"], -1)
    mean = (w * pred["mean"]).sum(-1)
    var = (w * (pred["log_scale"].exp().square() + pred["mean"].square())).sum(-1) - mean.square()
    return mean, var.clamp_min(1e-10).sqrt()
