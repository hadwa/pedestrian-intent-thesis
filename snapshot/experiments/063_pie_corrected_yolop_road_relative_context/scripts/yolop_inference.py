"""Pinned YOLOP inference with official local segmentation postprocessing."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .contracts import ExperimentConfig, sha256_file


YOLOP_ROOT = Path("/home/abdelrahman/Hadwa/MSLSTM-PID/YOLOP")
YOLOP_WEIGHTS = YOLOP_ROOT / "weights" / "End-to-end.pth"
YOLOP_NOTEBOOK = Path(
    "/home/abdelrahman/Hadwa/MSLSTM-PID/Offline Extraction/yolop-extraction.ipynb"
)
YOLOP_DEMO = YOLOP_ROOT / "tools" / "demo.py"
IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45


@dataclass(frozen=True)
class FramePerception:
    """Corrected automatic scene perception for one exact source frame."""

    detections: np.ndarray | None
    drivable_mask: np.ndarray
    lane_mask: np.ndarray


def letterbox(
    image: np.ndarray,
    new_shape: int | tuple[int, int] = IMAGE_SIZE,
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """Apply the fixed, non-auto YOLOP letterbox and return ratio and half-pad."""
    import cv2

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("YOLOP input must be an HxWx3 image")
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    source_height, source_width = image.shape[:2]
    ratio_value = min(new_shape[0] / source_height, new_shape[1] / source_width)
    ratio = (ratio_value, ratio_value)
    resized_shape = (
        int(round(source_width * ratio_value)),
        int(round(source_height * ratio_value)),
    )
    width_pad = (new_shape[1] - resized_shape[0]) / 2
    height_pad = (new_shape[0] - resized_shape[1]) / 2
    if (source_width, source_height) != resized_shape:
        image = cv2.resize(image, resized_shape, interpolation=cv2.INTER_LINEAR)
    top = int(round(height_pad - 0.1))
    bottom = int(round(height_pad + 0.1))
    left = int(round(width_pad - 0.1))
    right = int(round(width_pad + 0.1))
    prepared = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color,
    )
    if prepared.shape[:2] != new_shape:
        raise RuntimeError(
            f"Letterbox shape drifted: {prepared.shape[:2]} != {new_shape}"
        )
    return prepared, ratio, (width_pad, height_pad)


def remove_letterbox_and_argmax(
    logits,
    original_shape: tuple[int, int],
    pad: tuple[float, float],
) -> np.ndarray:
    """Crop padding, resize two-class logits, and return the class-1 mask."""
    import torch

    if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != 2:
        raise ValueError("YOLOP segmentation logits must be a 1x2 two-channel tensor")
    if len(original_shape) != 2 or min(int(value) for value in original_shape) <= 0:
        raise ValueError("original_shape must contain positive height and width")
    width_pad, height_pad = (float(pad[0]), float(pad[1]))
    top = int(round(height_pad - 0.1))
    bottom_pad = int(round(height_pad + 0.1))
    left = int(round(width_pad - 0.1))
    right_pad = int(round(width_pad + 0.1))
    tensor_height, tensor_width = logits.shape[-2:]
    bottom = tensor_height - bottom_pad if bottom_pad else tensor_height
    right = tensor_width - right_pad if right_pad else tensor_width
    if not (0 <= top < bottom <= tensor_height and 0 <= left < right <= tensor_width):
        raise ValueError("Letterbox padding removes the complete segmentation tensor")
    cropped = logits[:, :, top:bottom, left:right]
    resized = torch.nn.functional.interpolate(
        cropped,
        size=(int(original_shape[0]), int(original_shape[1])),
        mode="bilinear",
        align_corners=False,
    )
    return resized.argmax(dim=1)[0].to(torch.uint8).cpu().numpy()


def authenticate_yolop_sources() -> dict[str, str]:
    """Hash every external YOLOP artifact before model construction."""
    config = ExperimentConfig()
    expected = {
        str(YOLOP_WEIGHTS): config.yolop_checkpoint_sha256,
        str(YOLOP_NOTEBOOK): config.yolop_notebook_sha256,
        str(YOLOP_DEMO): config.yolop_demo_sha256,
    }
    for source, digest in expected.items():
        path = Path(source)
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"YOLOP source authentication failed: {path}")
    return expected


class CorrectedYolopExtractor:
    """Run the pinned checkpoint and return official foreground class masks."""

    def __init__(self, device: str = "cuda:0") -> None:
        import torch

        authenticate_yolop_sources()
        if str(YOLOP_ROOT) not in sys.path:
            sys.path.insert(0, str(YOLOP_ROOT))
        from lib.config import cfg
        from lib.models import get_net

        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not visible to PyTorch")
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        model = get_net(cfg)
        checkpoint = torch.load(YOLOP_WEIGHTS, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
        incompatibility = model.load_state_dict(state_dict, strict=True)
        if incompatibility.missing_keys or incompatibility.unexpected_keys:
            raise RuntimeError("Pinned YOLOP checkpoint did not load strictly")
        self.model = model.to(self.device).eval()
        self.parameter_count = sum(parameter.numel() for parameter in model.parameters())

    def infer_frame(self, image: np.ndarray) -> FramePerception:
        """Infer detections and corrected binary masks for one BGR source frame."""
        return self.infer_batch((image,))[0]

    def infer_batch(self, images: tuple[np.ndarray, ...] | list[np.ndarray]) -> list[FramePerception]:
        """Infer a fixed batch while retaining each frame's exact geometry."""
        import torch

        from lib.core.general import non_max_suppression, scale_coords

        if not images:
            raise ValueError("YOLOP inference batch must not be empty")
        prepared_images: list[np.ndarray] = []
        pads: list[tuple[float, float]] = []
        for image in images:
            prepared, _, pad = letterbox(image, IMAGE_SIZE)
            prepared_images.append(prepared[:, :, ::-1].transpose(2, 0, 1))
            pads.append(pad)
        tensor = torch.from_numpy(np.ascontiguousarray(np.stack(prepared_images))).to(
            self.device
        )
        tensor = tensor.float().div(255.0)
        with torch.inference_mode():
            detection_output, drivable_logits, lane_logits = self.model(tensor)
            raw_detection = (
                detection_output[0]
                if isinstance(detection_output, (list, tuple))
                else detection_output
            )
            detection_batches = non_max_suppression(
                raw_detection,
                CONFIDENCE_THRESHOLD,
                IOU_THRESHOLD,
                classes=None,
                agnostic=False,
            )
            outputs: list[FramePerception] = []
            for index, image in enumerate(images):
                detections = detection_batches[index]
                detection_array = None
                if detections is not None and len(detections):
                    detections = detections.clone()
                    scale_coords(tensor.shape[2:], detections[:, :4], image.shape)
                    detection_array = detections.cpu().numpy().astype(
                        np.float32, copy=False
                    )
                outputs.append(
                    FramePerception(
                        detections=detection_array,
                        drivable_mask=remove_letterbox_and_argmax(
                            drivable_logits[index : index + 1], image.shape[:2], pads[index]
                        ),
                        lane_mask=remove_letterbox_and_argmax(
                            lane_logits[index : index + 1], image.shape[:2], pads[index]
                        ),
                    )
                )
        return outputs


__all__ = [
    "CorrectedYolopExtractor",
    "FramePerception",
    "authenticate_yolop_sources",
    "letterbox",
    "remove_letterbox_and_argmax",
]
