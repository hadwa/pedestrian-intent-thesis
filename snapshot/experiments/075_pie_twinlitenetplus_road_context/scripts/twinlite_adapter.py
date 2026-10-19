"""Pinned TwinLiteNet+ Large preprocessing and inference adapter."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from types import SimpleNamespace
from types import ModuleType
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as functional

import contract


@dataclass(frozen=True)
class LetterboxMeta:
    """Exact geometry needed to restore logits to the source image."""

    original_hw: tuple[int, int]
    resized_hw: tuple[int, int]
    left: int
    top: int
    right: int
    bottom: int


def letterbox_bgr(
    image: np.ndarray,
    target: int = 640,
    stride: int = 32,
) -> tuple[np.ndarray, LetterboxMeta]:
    """Apply the official auto-stride letterbox to one BGR image."""

    values = np.asarray(image)
    if values.ndim != 3 or values.shape[2] != 3 or values.dtype != np.uint8:
        raise ValueError("image must be a nonempty uint8 HxWx3 BGR array")
    height, width = map(int, values.shape[:2])
    if height <= 0 or width <= 0 or target <= 0 or stride <= 0:
        raise ValueError("image, target, and stride dimensions must be positive")
    ratio = min(target / height, target / width)
    resized_width = int(round(width * ratio))
    resized_height = int(round(height * ratio))
    pad_width = float(np.mod(target - resized_width, stride)) / 2.0
    pad_height = float(np.mod(target - resized_height, stride)) / 2.0
    if (width, height) != (resized_width, resized_height):
        resized = cv2.resize(
            values, (resized_width, resized_height), interpolation=cv2.INTER_AREA
        )
    else:
        resized = values.copy()
    top = int(round(pad_height - 0.1))
    bottom_padding = int(round(pad_height + 0.1))
    left = int(round(pad_width - 0.1))
    right_padding = int(round(pad_width + 0.1))
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom_padding,
        left,
        right_padding,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    meta = LetterboxMeta(
        original_hw=(height, width),
        resized_hw=(resized_height, resized_width),
        left=left,
        top=top,
        right=left + resized_width,
        bottom=top + resized_height,
    )
    return np.ascontiguousarray(padded), meta


def prepare_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert official letterboxed BGR uint8 input to normalized RGB CHW."""

    values = np.asarray(image)
    if values.ndim != 3 or values.shape[2] != 3 or values.dtype != np.uint8:
        raise ValueError("letterboxed image must be uint8 HxWx3")
    rgb_chw = np.ascontiguousarray(values[:, :, ::-1].transpose(2, 0, 1))
    return torch.from_numpy(rgb_chw).float().div_(255.0)


def restore_logits(logits: torch.Tensor, meta: LetterboxMeta) -> torch.Tensor:
    """Crop letterbox padding and resize logits to exact source dimensions."""

    if logits.ndim != 4 or logits.shape[1] != 2:
        raise ValueError("TwinLiteNet logits must be Bx2xHxW")
    cropped = logits[..., meta.top : meta.bottom, meta.left : meta.right]
    if tuple(cropped.shape[-2:]) != meta.resized_hw:
        raise ValueError("logit geometry does not match letterbox metadata")
    return functional.interpolate(
        cropped, size=meta.original_hw, mode="bilinear", align_corners=False
    )


def load_large_model(
    *, map_location: str | torch.device = "cpu"
) -> tuple[torch.nn.Module, dict[str, object]]:
    """Strictly load the SHA-authenticated official Large checkpoint."""

    model_path = contract.resolve_source("twinlite_model")
    contract.resolve_source("twinlite_config")
    checkpoint_path = contract.resolve_source("twinlite_checkpoint")
    source_root = model_path.parents[1]
    source_text = str(source_root)
    while source_text in sys.path:
        sys.path.remove(source_text)
    sys.path.insert(0, source_text)
    for name in [key for key in sys.modules if key == "model" or key.startswith("model.")]:
        del sys.modules[name]
    # The upstream model directory has no __init__.py. Register it explicitly so
    # Exp075's own scripts/model.py cannot shadow the official package.
    model_package = ModuleType("model")
    model_package.__path__ = [str(source_root / "model")]
    model_package.__package__ = "model"
    sys.modules["model"] = model_package
    from model.model import TwinLiteNetPlus

    state = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    if type(state).__name__ != "OrderedDict":
        raise RuntimeError("official checkpoint must contain a plain OrderedDict")
    model = TwinLiteNetPlus(SimpleNamespace(config="large"))
    incompatible = model.load_state_dict(state, strict=True)
    parameters = sum(value.numel() for value in model.parameters())
    trainable = sum(value.numel() for value in model.parameters() if value.requires_grad)
    audit: dict[str, object] = {
        "container": type(state).__name__,
        "state_entries": len(state),
        "parameters": parameters,
        "trainable_parameters": trainable,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }
    frozen = contract.load_registration()["twinlite"]
    if (
        len(state) != int(frozen["state_entries"])
        or parameters != int(frozen["parameters"])
        or incompatible.missing_keys
        or incompatible.unexpected_keys
    ):
        raise RuntimeError("official Large checkpoint structure drifted")
    return model, audit


class TwinLiteExtractor:
    """Run the pinned two-head model and return original-coordinate masks."""

    def __init__(
        self,
        *,
        device: str | torch.device,
        half: bool,
        model: torch.nn.Module | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.half = bool(half)
        if self.half and self.device.type != "cuda":
            raise ValueError("FP16 primary inference requires CUDA")
        if model is None:
            model, self.load_audit = load_large_model(map_location="cpu")
        else:
            self.load_audit = None
        self.model = model.to(self.device)
        if self.half:
            self.model.half()
        else:
            self.model.float()
        self.model.eval()

    @property
    def model_parameters(self) -> int:
        return sum(value.numel() for value in self.model.parameters())

    def predict(
        self, images: Sequence[np.ndarray]
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if not images:
            raise ValueError("at least one image is required")
        prepared: list[torch.Tensor] = []
        metadata: list[LetterboxMeta] = []
        for image in images:
            padded, meta = letterbox_bgr(image)
            prepared.append(prepare_tensor(padded))
            metadata.append(meta)
        shapes = {tuple(value.shape) for value in prepared}
        if len(shapes) != 1:
            raise ValueError("one inference batch requires identical letterbox shapes")
        batch = torch.stack(prepared).to(self.device)
        batch = batch.half() if self.half else batch.float()
        with torch.inference_mode():
            outputs = self.model(batch)
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
            raise RuntimeError("TwinLiteNet must return drivable and lane logits")
        masks: list[list[np.ndarray]] = [[], []]
        for head_index, logits in enumerate(outputs):
            if logits.ndim != 4 or logits.shape[:2] != (len(images), 2):
                raise RuntimeError("TwinLiteNet head must have shape Bx2xHxW")
            if not torch.isfinite(logits).all():
                raise RuntimeError("TwinLiteNet produced non-finite logits")
            for index, meta in enumerate(metadata):
                restored = restore_logits(logits[index : index + 1], meta)
                foreground = restored.argmax(dim=1).eq(1).squeeze(0)
                masks[head_index].append(
                    foreground.detach().cpu().numpy().astype(bool, copy=False)
                )
        return masks[0], masks[1]
