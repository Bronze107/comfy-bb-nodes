# Split Video by Frames — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `SplitVideoByFrames` node that splits a video into lazy segments of N frames each, output as a list of videos.

**Architecture:** The node computes each segment's time window from the frame rate, calls `video.as_trimmed(start_time, duration, strict_duration=False)` per segment, and returns the list. Segments are lazy trim references over the source (no re-encode, no file writes), matching core `VideoSlice`. The last segment's duration is clamped to `total_duration - start_time` so a float-rounding boundary can never cause `VideoFromComponents.as_trimmed` to reject it.

**Tech Stack:** Python 3.13, ComfyUI v3 extension API (`io.ComfyNode`), `comfy_api.latest._input` `VideoInput` methods (`get_frame_rate`, `get_frame_count`, `get_duration`, `as_trimmed`), pytest.

**Verification baseline:** `23 passed` from `pytest custom_nodes/comfy-bb-nodes/tests` before this plan.

**Reference:** design spec at `docs/superpowers/specs/2026-08-19-split-video-by-frames-design.md`.

---

### Task 1: Write the failing test

**Files:**
- Create: `custom_nodes/comfy-bb-nodes/tests/test_split_video.py`

- [ ] **Step 1: Create the test file**

```python
import importlib.util
from fractions import Fraction
from pathlib import Path

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
```

Note: `decoded_frame_counts` decodes each segment via `get_components()` so the
assertions check the actual decoded frame count, not the metadata estimate.

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
cd "E:\ComfyUI_windows_portable\ComfyUI" && "E:\ComfyUI_windows_portable\python_embeded\python.exe" -m pytest custom_nodes/comfy-bb-nodes/tests/test_split_video.py -q
```
Expected: FAIL with `AttributeError: module '...video_nodes' has no attribute 'SplitVideoByFrames'`.

---

### Task 2: Implement `SplitVideoByFrames` in `video_nodes.py`

**Files:**
- Modify: `custom_nodes/comfy-bb-nodes/video_nodes.py`

- [ ] **Step 1: Append the node class at the end of `video_nodes.py`**

```python
class SplitVideoByFrames(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SplitVideoByFrames",
            display_name="Split Video by Frames",
            category="video",
            search_aliases=["split video by frames", "frame segments", "cut video into segments"],
            description="Split a video into segments of N frames each. The last segment may be shorter. Segments are lazy trim windows over the source, so nothing is re-encoded or written until saved.",
            inputs=[
                io.Video.Input("video", tooltip="The video to split."),
                io.Int.Input(
                    "frames_per_segment",
                    default=30,
                    min=1,
                    tooltip="Number of frames in each segment.",
                ),
            ],
            outputs=[
                io.Video.Output(display_name="segments", is_output_list=True),
            ],
        )

    @classmethod
    def execute(cls, video, frames_per_segment):
        frame_count = video.get_frame_count()
        if frame_count == 0:
            raise ValueError("Cannot split a video with no frames.")
        fps = video.get_frame_rate()
        total_duration = float(video.get_duration())
        segments = []
        for start in range(0, frame_count, frames_per_segment):
            start_time = float(start / fps)
            remaining = frame_count - start
            if remaining <= frames_per_segment:
                duration = total_duration - start_time
            else:
                duration = float(frames_per_segment / fps)
            segment = video.as_trimmed(start_time, duration, strict_duration=False)
            if segment is not None:
                segments.append(segment)
        return io.NodeOutput(segments)
```

The `duration = total_duration - start_time` clamp on the last segment guarantees
`start_time + duration == total_duration` exactly, so `as_trimmed` can never
reject the remainder segment due to float rounding.

- [ ] **Step 2: Run the test to verify it now passes**

Run:
```bash
cd "E:\ComfyUI_windows_portable\ComfyUI" && "E:\ComfyUI_windows_portable\python_embeded\python.exe" -m pytest custom_nodes/comfy-bb-nodes/tests/test_split_video.py -q
```
Expected: `4 passed`.

---

### Task 3: Register the node in `__init__.py`

**Files:**
- Modify: `custom_nodes/comfy-bb-nodes/__init__.py`

- [ ] **Step 1: Add `SplitVideoByFrames` to the import and to `get_node_list`**

Change the video import to:

```python
from video_nodes import VideoInfo, ConcatVideos, ResampleFPS, SplitVideoByFrames
```

and the `BBExtension.get_node_list` return list to:

```python
        return [
            LoadWildcardFile,
            RandomFromList,
            WildcardReplace,
            VideoInfo,
            ConcatVideos,
            ResampleFPS,
            SplitVideoByFrames,
        ]
```

- [ ] **Step 2: Verify the full test suite still passes**

Run:
```bash
cd "E:\ComfyUI_windows_portable\ComfyUI" && "E:\ComfyUI_windows_portable\python_embeded\python.exe" -m pytest custom_nodes/comfy-bb-nodes/tests -q
```
Expected: `27 passed` (23 existing + 4 new).

---

### Task 4: Verify ComfyUI loader imports the new node

**Files:**
- Read: `custom_nodes/comfy-bb-nodes/__init__.py`

- [ ] **Step 1: Replicate ComfyUI's `load_custom_node` directory-loading**

Run:
```bash
cd "E:\ComfyUI_windows_portable\ComfyUI" && "E:\ComfyUI_windows_portable\python_embeded\python.exe" -c "
import importlib.util, os, sys
mp = os.path.abspath('custom_nodes/comfy-bb-nodes')
name = mp.replace('.', '_x_')
spec = importlib.util.spec_from_file_location(name, os.path.join(mp, '__init__.py'))
mod = importlib.util.module_from_spec(spec)
sys.modules[name] = mod
spec.loader.exec_module(mod)
print('SplitVideoByFrames' in [c.__name__ for c in (mod.get_node_list and mod.BBExtension().get_node_list() or [])] if hasattr(mod, 'BBExtension') else 'NO EXT')
"
```
Expected: prints `True`.

---

### Task 5: Commit

**Files:**
- Commit: all changes in `custom_nodes/comfy-bb-nodes`

- [ ] **Step 1: Stage and commit**

```bash
cd "E:\ComfyUI_windows_portable\ComfyUI\custom_nodes\comfy-bb-nodes" && git add video_nodes.py __init__.py tests/test_split_video.py && git commit -m "Add SplitVideoByFrames node"
```

- [ ] **Step 2: Confirm commit**

Run:
```bash
cd "E:\ComfyUI_windows_portable\ComfyUI\custom_nodes\comfy-bb-nodes" && git log --oneline -3
```
Expected: newest commit is `Add SplitVideoByFrames node`.

---

## Self-review notes

- **Spec coverage:** Node schema/inputs/outputs (Task 2), remainder kept (Task 2 clamp), lazy `as_trimmed` list output (Task 2), registration (Task 3), tests for N=60/N=100/N=30 + duration (Task 1), ComfyUI loader check (Task 4), commit (Task 5). All spec items covered.
- **Placeholder scan:** No TBD/TODO; all code shown inline.
- **Type consistency:** `SplitVideoByFrames` used consistently in test, node class, and registration. `decoded_frame_counts` helper matches its use. `io.Video.Output(..., is_output_list=True)` matches the `LoadWildcardFile` precedent; `io.NodeOutput(segments_list)` matches list-output return convention.
- **Robustness deviation from spec sketch:** last-segment duration is clamped to `total_duration - start_time`; this only strengthens the spec's "keep remainder" requirement.
