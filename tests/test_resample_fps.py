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


def make_video(frames, fps=30):
    return VideoFromComponents(
        VideoComponents(images=torch.rand(frames, 32, 32, 3), frame_rate=Fraction(fps))
    )


def make_resizable_video(frames=30, fps=30):
    return VideoFromComponents(
        VideoComponents(images=torch.rand(frames, 180, 320, 3), frame_rate=Fraction(fps))
    )


def test_resample_30_to_24(tmp_path, monkeypatch):
    monkeypatch.setattr(_impl, "get_output_directory", lambda: str(tmp_path))
    video = make_video(30)
    result = ResampleFPS.execute(video, 24.0, 0.0, "test_resample")
    out = result[0]
    # Duration is preserved; frame count becomes ~24 for 1 second of video.
    assert out.get_duration() == pytest.approx(1.0, abs=0.1)
    assert out.get_frame_count() == pytest.approx(24, abs=2)


def test_resize_downscale_preserves_aspect(tmp_path, monkeypatch):
    monkeypatch.setattr(_impl, "get_output_directory", lambda: str(tmp_path))
    video = make_resizable_video()
    result = ResampleFPS.execute(video, 0.0, 0.02, "test_resize")
    out = result[0]
    w, h = out.get_dimensions()
    assert w < 320 and h < 180
    assert w / h == pytest.approx(320 / 180, abs=0.05)


def test_resize_never_upscales(tmp_path, monkeypatch):
    monkeypatch.setattr(_impl, "get_output_directory", lambda: str(tmp_path))
    video = make_resizable_video()
    result = ResampleFPS.execute(video, 0.0, 100.0, "test_resize")
    out = result[0]
    assert out.get_dimensions() == (320, 180)


def test_resize_keeps_source_fps(tmp_path, monkeypatch):
    monkeypatch.setattr(_impl, "get_output_directory", lambda: str(tmp_path))
    video = make_resizable_video()
    result = ResampleFPS.execute(video, 0.0, 0.02, "test_resize")
    out = result[0]
    # fps=0 keeps the source 30 fps: 30 frames over 1 second.
    assert out.get_duration() == pytest.approx(1.0, abs=0.1)
    assert out.get_frame_count() == pytest.approx(30, abs=2)


def test_no_transform_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(_impl, "get_output_directory", lambda: str(tmp_path))
    video = make_resizable_video()
    with pytest.raises(ValueError):
        ResampleFPS.execute(video, 0.0, 0.0, "test_resize")
