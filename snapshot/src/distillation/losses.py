"""Knowledge distillation losses for later experiments."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def output_kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Binary logit KD loss using softened probabilities."""
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    student_log_prob = F.logsigmoid(student_logits / temperature)
    teacher_prob = torch.sigmoid(teacher_logits / temperature)
    loss = F.binary_cross_entropy_with_logits(student_logits / temperature, teacher_prob, reduction="mean")
    return loss * (temperature**2)


def hard_label_loss(student_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard supervised binary cross-entropy with logits."""
    return F.binary_cross_entropy_with_logits(student_logits, labels.float(), reduction="mean")


def combined_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Weighted hard-label plus output-KD loss."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")
    return alpha * output_kd_loss(student_logits, teacher_logits, temperature) + (1.0 - alpha) * hard_label_loss(student_logits, labels)
