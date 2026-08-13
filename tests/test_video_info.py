import importlib.util
from fractions import Fraction
from pathlib import Path

import pytest
import torch
from comfy_api.input_impl.video_types import VideoFromComponents
from comfy_api.util.video_types import VideoComponents

_IMPL = Path(__file__).resolve().parent.parent / "__init__.py"
_spec = importlib.util.spec_from_file_location("comfy_bb_nodes", _IMPL)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)

VideoInfo = _impl.VideoInfo


def test_video_info_duration_and_frame_count():
    video = VideoFromComponents(
        VideoComponents(images=torch.rand(5, 2, 2, 3), frame_rate=Fraction(30))
    )
    duration, frame_count = VideoInfo.execute(video)
    assert duration == pytest.approx(5 / 30)
    assert frame_count == 5


def test_video_info_fractional_frame_rate():
    video = VideoFromComponents(
        VideoComponents(images=torch.rand(2, 2, 2, 3), frame_rate=Fraction(24000, 1001))
    )
    duration, frame_count = VideoInfo.execute(video)
    assert duration == pytest.approx(2 / (24000 / 1001))
    assert frame_count == 2
