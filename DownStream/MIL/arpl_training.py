"""ARPL-regularized training objective for CLAM-style MIL models.

The anomaly-detection variant keeps the original CLAM architecture and adds an
Adversarial Reciprocal Point Learning (ARPL) term during training. For the
binary slide-level experiments, ARPL is applied to the two-dimensional bag
logits:

    total_loss = bag_weight * bag_loss
               + (1 - bag_weight) * instance_loss
               + arpl_weight * arpl_loss

This file contains only the reusable training objective. Dataset paths, model
weights, experiment identifiers and evaluation-specific thresholding are
deliberately excluded.

Reference
---------
Chen et al., "Adversarial Reciprocal Points Learning for Open Set
Recognition", IEEE TPAMI (2021).
https://github.com/gary23ai/ARPL
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ReciprocalPointDistance(nn.Module):
    """Learnable reciprocal points and their class-wise distances."""

    def __init__(self, num_classes: int, feature_dim: int) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("ARPL requires at least two known classes.")
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive.")

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.centers = nn.Parameter(0.1 * torch.randn(num_classes, feature_dim))

    def forward(
        self,
        features: Tensor,
        *,
        metric: str,
        centers: Tensor | None = None,
    ) -> Tensor:
        """Return one distance or similarity value per sample and class."""
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                "features must have shape [batch, feature_dim], "
                f"but received {tuple(features.shape)}."
            )

        reference = self.centers if centers is None else centers
        if reference.shape != self.centers.shape:
            raise ValueError(
                "centers must have shape "
                f"{tuple(self.centers.shape)}, but received {tuple(reference.shape)}."
            )

        if metric == "l2":
            squared_distance = (
                features.square().sum(dim=1, keepdim=True)
                - 2.0 * features @ reference.t()
                + reference.square().sum(dim=1).unsqueeze(0)
            )
            return squared_distance / float(self.feature_dim)
        if metric == "dot":
            return features @ reference.t()
        raise ValueError("metric must be either 'l2' or 'dot'.")


class ARPLoss(nn.Module):
    """Reciprocal-point classification and open-space margin loss.

    Parameters mirror the settings used by the CLAM anomaly-detection
    experiments: one reciprocal point per known class, temperature 1.0,
    reciprocal-point regularization weight 0.1 and margin 1.0.
    """

    def __init__(
        self,
        num_classes: int,
        feature_dim: int | None = None,
        *,
        temperature: float = 1.0,
        reciprocal_weight: float = 0.1,
        margin: float = 1.0,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if reciprocal_weight < 0:
            raise ValueError("reciprocal_weight must be non-negative.")

        feature_dim = num_classes if feature_dim is None else feature_dim
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.temperature = temperature
        self.reciprocal_weight = reciprocal_weight

        self.distance = ReciprocalPointDistance(num_classes, feature_dim)
        self.radius = nn.Parameter(torch.zeros(1))
        self.margin_loss = nn.MarginRankingLoss(margin=margin)

    @property
    def reciprocal_points(self) -> Tensor:
        """Return the learnable reciprocal points."""
        return self.distance.centers

    def forward(self, representations: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        """Return ARPL logits and the complete ARPL training loss."""
        labels = labels.reshape(-1).long()
        if representations.shape[0] != labels.shape[0]:
            raise ValueError("representations and labels must share the batch size.")
        if labels.numel() == 0:
            raise ValueError("The batch must contain at least one sample.")
        if labels.min().item() < 0 or labels.max().item() >= self.num_classes:
            raise ValueError("labels contain a class index outside the known classes.")

        l2_distance = self.distance(representations, metric="l2")
        dot_similarity = self.distance(representations, metric="dot")
        arpl_logits = l2_distance - dot_similarity

        classification_loss = F.cross_entropy(
            arpl_logits / self.temperature,
            labels,
        )

        class_points = self.reciprocal_points.index_select(0, labels)
        known_distance = (representations - class_points).square().mean(dim=1)
        direction = torch.ones_like(known_distance)
        radius_loss = self.margin_loss(
            self.radius.expand_as(known_distance),
            known_distance,
            direction,
        )

        loss = classification_loss + self.reciprocal_weight * radius_loss
        return arpl_logits, loss


@dataclass(frozen=True)
class CLAMLosses:
    """Individual terms of the ARPL-regularized CLAM objective."""

    total: Tensor
    bag: Tensor
    instance: Tensor
    arpl: Tensor


def compute_clam_arpl_loss(
    bag_logits: Tensor,
    labels: Tensor,
    instance_loss: Tensor,
    arpl: ARPLoss,
    *,
    bag_weight: float = 0.7,
    arpl_weight: float = 1.0,
) -> CLAMLosses:
    """Combine the standard CLAM losses with the ARPL training term.

    ``bag_logits`` are used as the ARPL representations to reproduce the
    training adaptation used for the binary MIL anomaly-detection experiments.
    """
    if not 0.0 <= bag_weight <= 1.0:
        raise ValueError("bag_weight must be within [0, 1].")
    if arpl_weight < 0:
        raise ValueError("arpl_weight must be non-negative.")
    if bag_logits.ndim != 2 or bag_logits.shape[1] != arpl.feature_dim:
        raise ValueError(
            "bag_logits must have shape [batch, arpl.feature_dim], "
            f"but received {tuple(bag_logits.shape)}."
        )
    if instance_loss.ndim != 0:
        raise ValueError("instance_loss must be a scalar tensor.")

    labels = labels.reshape(-1).long()
    bag_loss = F.cross_entropy(bag_logits, labels)
    _, reciprocal_loss = arpl(bag_logits, labels)
    total_loss = (
        bag_weight * bag_loss
        + (1.0 - bag_weight) * instance_loss
        + arpl_weight * reciprocal_loss
    )
    return CLAMLosses(
        total=total_loss,
        bag=bag_loss,
        instance=instance_loss,
        arpl=reciprocal_loss,
    )


def joint_trainable_parameters(
    model: nn.Module,
    arpl: ARPLoss,
) -> Iterable[nn.Parameter]:
    """Yield model and ARPL parameters for one joint optimizer.

    Example:
        optimizer = torch.optim.Adam(
            joint_trainable_parameters(model, arpl),
            lr=1e-4,
            weight_decay=1e-5,
        )

    Including both parameter sets is required so that the reciprocal points and
    radius are optimized together with the MIL model.
    """
    return chain(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        (parameter for parameter in arpl.parameters() if parameter.requires_grad),
    )


__all__ = [
    "ARPLoss",
    "CLAMLosses",
    "ReciprocalPointDistance",
    "compute_clam_arpl_loss",
    "joint_trainable_parameters",
]

