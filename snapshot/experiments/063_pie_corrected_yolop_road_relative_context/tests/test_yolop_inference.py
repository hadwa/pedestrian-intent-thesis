from __future__ import annotations

import importlib

import numpy as np
import pytest


torch = pytest.importorskip("torch")


def _module():
    return importlib.import_module(
        "experiments.063_pie_corrected_yolop_road_relative_context.scripts.yolop_inference"
    )


def test_postprocess_crops_padding_before_resize_and_selects_foreground() -> None:
    inference = _module()
    logits = torch.zeros(1, 2, 8, 8)
    logits[:, 0] = 2.0
    logits[:, 1, 2:6, 2:6] = 4.0
    mask = inference.remove_letterbox_and_argmax(
        logits,
        original_shape=(4, 4),
        pad=(2.0, 2.0),
    )
    assert mask.shape == (4, 4)
    assert mask.dtype == np.uint8
    assert np.all(mask == 1)


def test_postprocess_does_not_sigmoid_channel_zero() -> None:
    inference = _module()
    logits = torch.empty(1, 2, 4, 4)
    logits[:, 0] = 10.0
    logits[:, 1] = -10.0
    mask = inference.remove_letterbox_and_argmax(
        logits,
        original_shape=(4, 4),
        pad=(0.0, 0.0),
    )
    assert np.all(mask == 0)


def test_postprocess_rejects_malformed_two_channel_tensor() -> None:
    inference = _module()
    with pytest.raises(ValueError, match="two-channel"):
        inference.remove_letterbox_and_argmax(
            torch.zeros(1, 1, 4, 4),
            original_shape=(4, 4),
            pad=(0.0, 0.0),
        )


def test_letterbox_metadata_round_trips_non_square_image() -> None:
    pytest.importorskip("cv2")
    inference = _module()
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    prepared, _, pad = inference.letterbox(image, 640)
    assert prepared.shape == (640, 640, 3)
    assert pad == pytest.approx((0.0, 160.0))

    logits = torch.zeros(1, 2, 640, 640)
    logits[:, 1, 160:480, :] = 1.0
    mask = inference.remove_letterbox_and_argmax(
        logits,
        original_shape=image.shape[:2],
        pad=pad,
    )
    assert mask.shape == image.shape[:2]
    assert np.all(mask == 1)
