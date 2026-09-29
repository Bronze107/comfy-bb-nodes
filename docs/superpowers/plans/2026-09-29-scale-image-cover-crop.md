# Scale Image to Total Pixels (Cover Crop) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `comfy-bb-nodes` image node that scales to a target megapixel count with step-aligned output size, using uniform scale + center-crop so content is never stretched.

**Architecture:** One new module `image_nodes.py` (mirrors `text_nodes.py`/`video_nodes.py` style), one node registered in `__init__.py`. Reuses `comfy.utils.common_upscale`; the crop is a plain tensor slice.

**Tech Stack:** Python 3.12 (portable embed), ComfyUI v3 extension API (`io.ComfyNode`), `comfy.utils.common_upscale`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-29-scale-image-cover-crop-design.md`

**Verification baseline:** `31 passed` from `E:\ComfyUI_windows_portable\ComfyUI` with
`"E:\ComfyUI_windows_portable\python_embeded\python.exe" -m pytest custom_nodes/comfy-bb-nodes/tests -q`.
All commands below run from the ComfyUI repo root (`E:\ComfyUI_windows_portable\ComfyUI`), not the pack dir.

---

### Task 1: Failing tests for the new node

**Files:**
- Create: `custom_nodes/comfy-bb-nodes/tests/test_image_scale_cover_crop.py`

- [ ] **Step 1: Write the failing test file**

```python
import importlib.util
from pathlib import Path

import pytest
import torch

_IMPL = Path(__file__).resolve().parent.parent / "image_nodes.py"
_spec = importlib.util.spec_from_file_location("comfy_bb_image_nodes", _IMPL)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)

ScaleImageToTotalPixelsCoverCrop = _impl.ScaleImageToTotalPixelsCoverCrop


def run(image, megapixels=1.0, steps=1, method="bilinear"):
    return ScaleImageToTotalPixelsCoverCrop.execute(image, method, megapixels, steps)[0]


def test_output_shape_step_aligned():
    image = torch.rand(1, 1080, 1920, 3)
    out = run(image, megapixels=1.0, steps=8)
    # scale_by = sqrt(1048576 / (1920*1080)) = 0.71109
    # width: round(1365.29 / 8) * 8 = 1368, height: round(767.98 / 8) * 8 = 768
    assert out.shape == (1, 768, 1368, 3)


def test_total_pixels_close_to_target():
    image = torch.rand(1, 1080, 1920, 3)
    out = run(image, megapixels=1.0, steps=8)
    target = 1.0 * 1024 * 1024
    assert abs(out.shape[1] * out.shape[2] - target) < 1368 * 8 + 768 * 8


def test_batch_frames_share_output_size():
    image = torch.rand(2, 1080, 1920, 3)
    out = run(image, megapixels=1.0, steps=8)
    assert out.shape == (2, 768, 1368, 3)


def test_identity_when_scale_is_one():
    image = torch.rand(1, 1024, 1024, 3)
    out = run(image, megapixels=1.0, steps=1, method="nearest-exact")
    assert out.shape == image.shape
    assert torch.equal(out, image)


def test_center_crop_trims_edges_symmetrically():
    # Pixel value encodes its original row position: 0 at top, 1 at bottom.
    image = torch.linspace(0.0, 1.0, 1080).view(1, 1080, 1, 1).expand(1, 1080, 1920, 3).contiguous()
    out = run(image, megapixels=1.0, steps=8, method="bilinear")
    assert out.shape == (1, 768, 1368, 3)
    # 1368x770 scaled result is cropped to 1368x768: one row trimmed from each
    # edge. First and last visible rows are therefore symmetric around 0.5.
    assert out[0, 0, 0, 0].item() + out[0, -1, 0, 0].item() == pytest.approx(1.0, abs=0.01)
    assert out[0, 383, 0, 0].item() == pytest.approx(0.5, abs=0.01)
    assert out[0, 384, 0, 0].item() == pytest.approx(0.5, abs=0.01)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `"E:\ComfyUI_windows_portable\python_embeded\python.exe" -m pytest custom_nodes/comfy-bb-nodes/tests/test_image_scale_cover_crop.py -v`
Expected: ERROR / collection failure — `image_nodes.py` does not exist yet
(`FileNotFoundError` while loading the module spec).

---

### Task 2: Implement the node

**Files:**
- Create: `custom_nodes/comfy-bb-nodes/image_nodes.py`
- Modify: `custom_nodes/comfy-bb-nodes/__init__.py`

- [ ] **Step 1: Create `image_nodes.py`**

```python
import math

import torch

import comfy.utils
from comfy_api.latest import io


class ScaleImageToTotalPixelsCoverCrop(io.ComfyNode):
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ScaleImageToTotalPixelsCoverCrop",
            display_name="Scale Image to Total Pixels (Cover Crop)",
            category="image/upscaling",
            description="Scale to a target megapixel count with step-aligned output size. "
            "Uniform scaling plus center-crop: no stretch, at most one step of edge content lost per axis.",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("upscale_method", options=cls.upscale_methods),
                io.Float.Input("megapixels", default=1.0, min=0.01, max=16.0, step=0.01),
                io.Int.Input("resolution_steps", default=1, min=1, max=256, advanced=True),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, image, upscale_method, megapixels, resolution_steps) -> io.NodeOutput:
        height, width = image.shape[-3:-1]
        total = megapixels * 1024 * 1024

        scale_by = math.sqrt(total / (width * height))
        out_w = max(round(width * scale_by / resolution_steps) * resolution_steps, resolution_steps)
        out_h = max(round(height * scale_by / resolution_steps) * resolution_steps, resolution_steps)

        cover = max(out_w / width, out_h / height)
        scaled_w = round(width * cover)
        scaled_h = round(height * cover)

        s = comfy.utils.common_upscale(image.movedim(-1, 1), scaled_w, scaled_h, upscale_method, "disabled")
        x0 = (scaled_w - out_w) // 2
        y0 = (scaled_h - out_h) // 2
        s = s[..., y0:y0 + out_h, x0:x0 + out_w]
        return io.NodeOutput(s.movedim(1, -1))
```

Notes:
- `image` is `[B, H, W, C]`; `shape[-3:-1]` reads H/W and the `...` slice crops
  H/W for every batch frame at once.
- `max(..., resolution_steps)` keeps the output at least one step when a huge
  `resolution_steps` would otherwise round a dimension to 0.
- `torch` is imported because the test loader and ComfyUI's image contract are
  tensor-based; keep the import even though this module only slices.

- [ ] **Step 2: Register the node in `__init__.py`**

Add to the imports after the `video_nodes` import:

```python
from image_nodes import ScaleImageToTotalPixelsCoverCrop
```

Add `ScaleImageToTotalPixelsCoverCrop` to the end of the list returned by
`get_node_list`, after `SplitVideoByFrames`:

```python
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LoadWildcardFile,
            RandomFromList,
            WildcardReplace,
            VideoInfo,
            ConcatVideos,
            ResampleFPS,
            SplitVideoByFrames,
            ScaleImageToTotalPixelsCoverCrop,
        ]
```

- [ ] **Step 3: Run the new tests to verify they pass**

Run: `"E:\ComfyUI_windows_portable\python_embeded\python.exe" -m pytest custom_nodes/comfy-bb-nodes/tests/test_image_scale_cover_crop.py -v`
Expected: 5 passed.

- [ ] **Step 4: Run the full suite to verify nothing regressed**

Run: `"E:\ComfyUI_windows_portable\python_embeded\python.exe" -m pytest custom_nodes/comfy-bb-nodes/tests -q`
Expected: 36 passed (31 existing + 5 new).

- [ ] **Step 5: Commit**

From the pack directory (`custom_nodes/comfy-bb-nodes`, which is its own git repo):

```bash
git add image_nodes.py __init__.py tests/test_image_scale_cover_crop.py
git commit -m "Add cover-crop image scale to total pixels node

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 3: Verify the node loads in ComfyUI

**Files:** none (verification only)

- [ ] **Step 1: Import the pack the way ComfyUI's loader does**

Run:
`"E:\ComfyUI_windows_portable\python_embeded\python.exe" -c "import sys; sys.path.insert(0, 'custom_nodes/comfy-bb-nodes'); import importlib.util; spec = importlib.util.spec_from_file_location('bb', 'custom_nodes/comfy-bb-nodes/__init__.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); import asyncio; ext = asyncio.run(m.comfy_entrypoint()); print([n.__name__ for n in asyncio.run(ext.get_node_list())])"`

Expected: the printed list ends with `ScaleImageToTotalPixelsCoverCrop` and no
import error.

- [ ] **Step 2: Report done**

No further commit needed; Task 2's commit is the complete change.

---

## Self-Review Notes

- Spec coverage: node definition, algorithm, edge clamp, batch handling, and
  all five planned test angles are covered by Tasks 1–2; loader verification is
  Task 3. Non-goals (mask, crop-offset output) are intentionally absent.
- Type consistency: `execute(image, upscale_method, megapixels, resolution_steps)`
  matches between the test helper `run()` and the implementation; node class
  name matches across test, module, and registration.
