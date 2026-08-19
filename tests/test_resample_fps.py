import importlib.util
from fractions import Fraction
from pathlib import Path

import pytest
import torch
from comfy_api.input_impl.video_types import VideoFromComponents
from comfy_api.util.video_types import VideoComponents

_IMPL = Path(__file__).resolve().parent.parent / "video_nodes.py"
_spec = importlib.util.spec_from_file_location("comfy_bb_video_nodes", _IMPL)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)

ResampleFPS = _impl.ResampleFPS


def test_resample_30_to_24(tmp_path, monkeypatch):
    monkeypatch.setattr(_impl, "get_output_directory", lambda: str(tmp_path))
    video = VideoFromComponents(
        VideoComponents(images=torch.rand(30, 32, 32, 3), frame_rate=Fraction(30))
    )
    result = ResampleFPS.execute(video, 24.0, "test_resample")
    out = result[0]
    # Duration is preserved; frame count becomes ~24 for 1 second of video.
    assert out.get_duration() == pytest.approx(1.0, abs=0.1)
    assert out.get_frame_count() == pytest.approx(24, abs=2)
