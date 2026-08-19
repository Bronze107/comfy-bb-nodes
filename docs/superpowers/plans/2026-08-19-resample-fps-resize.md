# ResampleFPS Refactor: Add Megapixel-Based Scaling — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the `ResampleFPS` node to also scale a video to a target megapixel count (aspect-preserving, downscale-only), with `fps=0` meaning "keep source fps".

**Architecture:** The node keeps its materialize → probe → ffmpeg → re-encode flow. The new `megapixels` input drives a `scale` filter appended to the `fps` filter; the resulting filter chain is joined into one `-vf`. `node_id` stays `ResampleFPS` for workflow compatibility; `display_name` becomes "Resample Video".

**Tech Stack:** Python 3.13, ComfyUI v3 extension API (`io.ComfyNode`), ffmpeg filters (`fps`, `scale`), pytest.

**Verification baseline:** `27 passed` from `pytest custom_nodes/comfy-bb-nodes/tests` before this plan.

**Reference:** design spec at `docs/superpowers/specs/2026-08-19-resample-fps-resize-design.md`.

---

### Task 1: Update tests to the new signature and add resize tests

**Files:**
- Modify: `custom_nodes/comfy-bb-nodes/tests/test_resample_fps.py`

- [ ] **Step 1: Replace the whole file with:**

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
cd "E:\ComfyUI_windows_portable\ComfyUI" && "E:\ComfyUI_windows_portable\python_embeded\python.exe" -m pytest custom_nodes/comfy-bb-nodes/tests/test_resample_fps.py -q
```
Expected: FAIL with `TypeError` — the current `execute` takes 3 parameters, not 4.

---

### Task 2: Implement the node changes in `video_nodes.py`

**Files:**
- Modify: `custom_nodes/comfy-bb-nodes/video_nodes.py`

- [ ] **Step 1: Add `import math` to the imports**

The file currently starts with:

```python
import json
import os
import shutil
import subprocess
import uuid
```

Change it to:

```python
import json
import math
import os
import shutil
import subprocess
import uuid
```

- [ ] **Step 2: Replace `define_schema` and `execute` of the `ResampleFPS` class (currently lines 252-314) with:**

```python
class ResampleFPS(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ResampleFPS",
            display_name="Resample Video",
            category="video",
            search_aliases=["adjust fps", "change frame rate", "convert fps", "set fps", "resize", "scale resolution", "adjust resolution"],
            description="Re-encode a video to a new frame rate and/or scale it to a target megapixel count. fps 0 keeps the source frame rate; megapixels 0 keeps the source resolution. Scaling preserves aspect ratio and never upscales.",
            is_output_node=True,
            inputs=[
                io.Video.Input("video", tooltip="The video to resample or resize."),
                io.Float.Input(
                    "fps",
                    default=24.0,
                    min=0.0,
                    max=240.0,
                    step=0.01,
                    tooltip="Target frame rate in frames per second, or 0 to keep the source frame rate.",
                ),
                io.Float.Input(
                    "megapixels",
                    default=2.0,
                    min=0.0,
                    step=0.1,
                    tooltip="Target size in megapixels, or 0 to keep the source resolution. Aspect ratio is preserved and the video is never upscaled.",
                ),
                io.String.Input(
                    "filename_prefix",
                    default="video/ResampleFPS",
                    tooltip="Prefix for the output filename in the output directory.",
                ),
            ],
            outputs=[
                io.Video.Output(),
            ],
        )

    @classmethod
    def execute(cls, video, fps, megapixels, filename_prefix):
        temp_dir = get_temp_directory()
        path, is_temp = _materialize_video(video, temp_dir)
        temp_paths = [path] if is_temp else []
        try:
            ref_video, audio = _probe_video(path)
            if fps <= 0 and megapixels <= 0:
                raise ValueError("Set fps > 0 or megapixels > 0 to change the video.")
            vf = []
            if fps > 0:
                vf.append(f"fps={fps:g}")
            if megapixels > 0:
                width = ref_video.get("width")
                height = ref_video.get("height")
                if not width or not height:
                    raise RuntimeError("Could not determine the source video's dimensions.")
                factor = math.sqrt((megapixels * 1_000_000) / (width * height))
                if factor < 1.0:
                    new_w = round(width * factor) - (round(width * factor) % 2)
                    new_h = round(height * factor) - (round(height * factor) % 2)
                    vf.append(f"scale={new_w}:{new_h}:flags=lanczos")
            out_path, file, subfolder = _output_path(filename_prefix, ref_video)
            cmd = [
                _find_ffmpeg_tool("ffmpeg"),
                "-y",
                "-i", path,
            ]
            if vf:
                cmd.extend(["-vf", ",".join(vf)])
            cmd.extend(
                [
                    "-c:v", "libx264",
                    "-crf", "18",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                ]
            )
            if audio:
                cmd.extend(["-c:a", "aac"])
            cmd.append(out_path)
            _run_process(cmd)
            return io.NodeOutput(
                VideoFromFile(out_path),
                ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)]),
            )
        finally:
            for path in temp_paths:
                try:
                    os.remove(path)
                except OSError:
                    pass
```

Notes:
- The `-vf` flag is only passed when the filter chain is non-empty (covers the
  "megapixels set but no scale needed" case without a bogus `-vf ""`).
- `node_id` stays `ResampleFPS`; only `display_name` and schema/execute change.

- [ ] **Step 3: Run the tests to verify they pass**

Run:
```bash
cd "E:\ComfyUI_windows_portable\ComfyUI" && "E:\ComfyUI_windows_portable\python_embeded\python.exe" -m pytest custom_nodes/comfy-bb-nodes/tests/test_resample_fps.py -q
```
Expected: `5 passed`.

---

### Task 3: Run the full test suite

**Files:**
- Test: `custom_nodes/comfy-bb-nodes/tests`

- [ ] **Step 1: Run all tests**

Run:
```bash
cd "E:\ComfyUI_windows_portable\ComfyUI" && "E:\ComfyUI_windows_portable\python_embeded\python.exe" -m pytest custom_nodes/comfy-bb-nodes/tests -q
```
Expected: `32 passed` (27 existing + 5 in the updated test file).

---

### Task 4: Verify ComfyUI loader imports the updated node

**Files:**
- Read: `custom_nodes/comfy-bb-nodes/__init__.py`

- [ ] **Step 1: Replicate ComfyUI's `load_custom_node` directory-loading**

Run:
```bash
cd "E:\ComfyUI_windows_portable\ComfyUI" && "E:\ComfyUI_windows_portable\python_embeded\python.exe" -c "
import asyncio, importlib.util, os, sys
mp = os.path.abspath('custom_nodes/comfy-bb-nodes')
name = mp.replace('.', '_x_')
spec = importlib.util.spec_from_file_location(name, os.path.join(mp, '__init__.py'))
mod = importlib.util.module_from_spec(spec)
sys.modules[name] = mod
spec.loader.exec_module(mod)
ext = asyncio.run(mod.comfy_entrypoint())
names = [c.__name__ for c in asyncio.run(ext.get_node_list())]
print('has ResampleFPS:', 'ResampleFPS' in names)
schema = mod.ResampleFPS.GET_SCHEMA()
print('display_name:', schema.display_name)
print('inputs:', [i.name for i in schema.inputs])
"
```
Expected: `has ResampleFPS: True`, `display_name: Resample Video`, and inputs listed as `['video', 'fps', 'megapixels', 'filename_prefix']`.

---

### Task 5: Commit

**Files:**
- Commit: all changes in `custom_nodes/comfy-bb-nodes`

- [ ] **Step 1: Stage and commit**

```bash
cd "E:\ComfyUI_windows_portable\ComfyUI\custom_nodes\comfy-bb-nodes" && git add video_nodes.py tests/test_resample_fps.py && git commit -m "Add megapixel scaling to ResampleFPS"
```

- [ ] **Step 2: Confirm commit**

Run:
```bash
cd "E:\ComfyUI_windows_portable\ComfyUI\custom_nodes\comfy-bb-nodes" && git log --oneline -3
```
Expected: newest commit is `Add megapixel scaling to ResampleFPS`.

---

## Self-review notes

- **Spec coverage:** `megapixels` input (Task 2), `fps=0` keep (Task 2), downscale-only `factor < 1.0` gate (Task 2), display name + search aliases (Task 2), no-transform error (Task 2), tests for resize/aspect/no-upscale/fps-keep/error (Task 1), loader check (Task 4), commit (Task 5). All spec items covered.
- **Placeholder scan:** No TBD/TODO; all code shown inline.
- **Type consistency:** `execute(video, fps, megapixels, filename_prefix)` matches the schema input order everywhere; `ResampleFPS` name unchanged; `make_resizable_video` matches its uses; `out.get_dimensions()` returns `(w, h)` consistently.
- **Robustness deviation from spec sketch:** the error check uses input parameters (`fps <= 0 and megapixels <= 0`) instead of an empty filter chain, so "megapixels set but no scale needed" is a no-op rather than an error, and `-vf` is omitted when empty.
