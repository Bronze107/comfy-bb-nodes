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

ConcatVideos = _impl.ConcatVideos


def make_video(frames, fps=30):
    return VideoFromComponents(
        VideoComponents(images=torch.rand(frames, 32, 32, 3), frame_rate=Fraction(fps))
    )


def test_concat_stream_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(_impl, "get_output_directory", lambda: str(tmp_path))
    result = ConcatVideos.execute(
        {"video_1": make_video(5), "video_2": make_video(3)}, "test_concat"
    )
    video = result[0]
    assert video.get_duration() == pytest.approx(8 / 30, abs=0.1)
    assert video.get_frame_count() == 8


def test_concat_reencode_mismatched_fps(tmp_path, monkeypatch):
    monkeypatch.setattr(_impl, "get_output_directory", lambda: str(tmp_path))
    result = ConcatVideos.execute(
        {"video_1": make_video(6, fps=30), "video_2": make_video(6, fps=24)}, "test_concat"
    )
    video = result[0]
    # fps filter preserves duration: 6/30 + 6/24 seconds.
    assert video.get_duration() == pytest.approx(6 / 30 + 6 / 24, abs=0.1)
