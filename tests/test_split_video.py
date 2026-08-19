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

SplitVideoByFrames = _impl.SplitVideoByFrames


def make_video(frames, fps=30):
    return VideoFromComponents(
        VideoComponents(images=torch.rand(frames, 32, 32, 3), frame_rate=Fraction(fps))
    )


def decoded_frame_counts(segments):
    return [s.get_components().images.shape[0] for s in segments]


def test_split_into_two_segments():
    segments = SplitVideoByFrames.execute(make_video(100), 60)[0]
    assert len(segments) == 2
    assert decoded_frame_counts(segments) == [60, 40]


def test_split_single_segment_when_n_covers_all():
    segments = SplitVideoByFrames.execute(make_video(100), 100)[0]
    assert len(segments) == 1
    assert decoded_frame_counts(segments) == [100]


def test_split_keeps_remainder():
    segments = SplitVideoByFrames.execute(make_video(100), 30)[0]
    assert len(segments) == 4
    counts = decoded_frame_counts(segments)
    assert counts == [30, 30, 30, 10]
    assert sum(counts) == 100


def test_split_preserves_total_duration():
    video = make_video(100)
    segments = SplitVideoByFrames.execute(video, 30)[0]
    assert sum(s.get_duration() for s in segments) == pytest.approx(
        video.get_duration(), abs=0.05
    )
