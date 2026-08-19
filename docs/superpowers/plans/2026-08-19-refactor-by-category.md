# Refactor comfy-bb-nodes by Category — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the single `__init__.py` into `text_nodes.py` and `video_nodes.py` so each file has one clear responsibility, with `__init__.py` reduced to an entry point that registers all nodes.

**Architecture:** Move the wildcard/text code (4 helpers + 3 nodes) into `text_nodes.py`, the ffmpeg/video code (10 helpers + 3 nodes) into `video_nodes.py`, and keep only node imports + `BBExtension` + `comfy_entrypoint` in `__init__.py`. Node behavior, IDs, categories, and schemas are unchanged. Tests load the submodule files directly (relative imports do not work under the tests' standalone module loader).

**Tech Stack:** Python 3.13, ComfyUI v3 extension API (`io.ComfyNode`, `ComfyExtension`), `folder_paths`, pytest.

**Verification baseline:** `23 passed` from `pytest custom_nodes/comfy-bb-nodes/tests` before any change.

---

## Source-of-truth note

Tasks 1 and 2 move code **verbatim** (no behavior edits) from the current
`custom_nodes/comfy-bb-nodes/__init__.py` (state as of commit `c1739380c`'s custom-node tree,
i.e. before this plan's edits). Copy each listed symbol exactly as it appears there. The
only edits are the `import` blocks, which are trimmed to what each module uses.

---

### Task 1: Create `text_nodes.py`

**Files:**
- Create: `custom_nodes/comfy-bb-nodes/text_nodes.py`

- [ ] **Step 1: Create the file with the trimmed import block**

```python
import os
import random
import re

from comfy_api.latest import io
from folder_paths import (
    base_path,
    exists_annotated_filepath,
    get_annotated_filepath,
)
```

- [ ] **Step 2: Move these top-level symbols verbatim from `__init__.py` (top to bottom, no other symbols):**

1. `resolve_path`
2. `load_wildcard_lines`
3. `pick_random_line`
4. `wildcard_replace`
5. `class LoadWildcardFile`
6. `class RandomFromList`
7. `class WildcardReplace`

Each symbol's body must be byte-identical to its current definition. Do not modify
docstrings, schemas, defaults, or error messages.

- [ ] **Step 3: Verify the module loads standalone**

Run:
```bash
"E:\ComfyUI_windows_portable\python_embeded\python.exe" -c "import importlib.util; s=importlib.util.spec_from_file_location('t','custom_nodes/comfy-bb-nodes/text_nodes.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(m.LoadWildcardFile, m.RandomFromList, m.WildcardReplace)"
```
Expected: prints the three node classes, no exception.

---

### Task 2: Create `video_nodes.py`

**Files:**
- Create: `custom_nodes/comfy-bb-nodes/video_nodes.py`

- [ ] **Step 1: Create the file with the trimmed import block**

```python
import json
import os
import shutil
import subprocess
import uuid

from comfy_api.input_impl import VideoFromFile
from comfy_api.latest import io, Types, ui
from folder_paths import (
    get_output_directory,
    get_save_image_path,
    get_temp_directory,
)
```

- [ ] **Step 2: Move these top-level symbols verbatim from `__init__.py` (top to bottom, no other symbols):**

1. `class VideoInfo`
2. `_find_ffmpeg_tool`
3. `_run_process`
4. `_materialize_video`
5. `_probe_video`
6. `_stream_signature`
7. `_concat_list_line`
8. `_write_concat_list`
9. `_output_path`
10. `_concat_stream_copy`
11. `_concat_reencode`
12. `class ConcatVideos`
13. `class ResampleFPS`

Each symbol's body must be byte-identical to its current definition. Do not modify
docstrings, schemas, defaults, or error messages.

- [ ] **Step 3: Verify the module loads standalone**

Run:
```bash
"E:\ComfyUI_windows_portable\python_embeded\python.exe" -c "import importlib.util; s=importlib.util.spec_from_file_location('v','custom_nodes/comfy-bb-nodes/video_nodes.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(m.VideoInfo, m.ConcatVideos, m.ResampleFPS)"
```
Expected: prints the three node classes, no exception.

---

### Task 3: Rewrite `__init__.py` as a thin entry point

**Files:**
- Rewrite: `custom_nodes/comfy-bb-nodes/__init__.py`

- [ ] **Step 1: Replace the entire file content with:**

```python
import os
import sys

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

# The directory name "comfy-bb-nodes" is not a valid Python module name, so the
# package cannot use relative imports. Expose the sibling modules via sys.path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from text_nodes import LoadWildcardFile, RandomFromList, WildcardReplace
from video_nodes import VideoInfo, ConcatVideos, ResampleFPS


class BBExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LoadWildcardFile,
            RandomFromList,
            WildcardReplace,
            VideoInfo,
            ConcatVideos,
            ResampleFPS,
        ]


async def comfy_entrypoint() -> BBExtension:
    return BBExtension()
```

Notes:
- `WildcardExtension` is renamed to `BBExtension`; the entrypoint name
  `comfy_entrypoint` is unchanged (ComfyUI's `load_custom_node` only looks for that name).
- **Deviation from the original plan:** absolute imports via a `sys.path` insert
  replace relative imports. Relative imports break because pytest collects the
  `comfy-bb-nodes` directory (a hyphenated, non-identifier name) as a package and
  imports `__init__.py` with an empty `__package__`. No `pytest.ini` is added.

- [ ] **Step 2: Verify it loads under ComfyUI's loader pattern (module registered in sys.modules)**

Run:
```bash
"E:\ComfyUI_windows_portable\python_embeded\python.exe" -c "
import importlib.util, os, sys
mp = os.path.abspath('custom_nodes/comfy-bb-nodes')
name = mp.replace('.', '_x_')
spec = importlib.util.spec_from_file_location(name, os.path.join(mp, '__init__.py'))
mod = importlib.util.module_from_spec(spec)
sys.modules[name] = mod
spec.loader.exec_module(mod)
print('loaded', name)
print([n for n in ['LoadWildcardFile','RandomFromList','WildcardReplace','VideoInfo','ConcatVideos','ResampleFPS'] if hasattr(mod, n)])
"
```
Expected: prints `loaded <name>` and all 6 node names.

---

### Task 4: Update tests to load submodule files

**Files:**
- Modify: `custom_nodes/comfy-bb-nodes/tests/test_wildcards.py`
- Modify: `custom_nodes/comfy-bb-nodes/tests/test_video_info.py`
- Modify: `custom_nodes/comfy-bb-nodes/tests/test_concat_videos.py`
- Modify: `custom_nodes/comfy-bb-nodes/tests/test_resample_fps.py`

- [ ] **Step 1: In `test_wildcards.py`, change lines 8-11 to:**

```python
_IMPL = Path(__file__).resolve().parent.parent / "text_nodes.py"
_spec = importlib.util.spec_from_file_location("comfy_bb_text_nodes", _IMPL)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
```

The four `_impl.<helper>` assignments (lines 13-16) stay unchanged.

- [ ] **Step 2: In `test_video_info.py`, change lines 10-13 to:**

```python
_IMPL = Path(__file__).resolve().parent.parent / "video_nodes.py"
_spec = importlib.util.spec_from_file_location("comfy_bb_video_nodes", _IMPL)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
```

- [ ] **Step 3: In `test_concat_videos.py`, change lines 10-13 to the same block as Step 2.** Keep `ConcatVideos = _impl.ConcatVideos` and both `monkeypatch.setattr(_impl, "get_output_directory", ...)` lines unchanged — they now target the `video_nodes` module, which is the module that resolves `get_output_directory` in `_output_path`.

- [ ] **Step 4: In `test_resample_fps.py`, change lines 10-13 to the same block as Step 2.** Keep `ResampleFPS = _impl.ResampleFPS` and the `monkeypatch.setattr(_impl, "get_output_directory", ...)` line unchanged.

---

### Task 5: Run the full test suite

**Files:**
- Test: `custom_nodes/comfy-bb-nodes/tests`

- [ ] **Step 1: Run all tests**

Run:
```bash
cd "E:\ComfyUI_windows_portable\ComfyUI" && "E:\ComfyUI_windows_portable\python_embeded\python.exe" -m pytest custom_nodes/comfy-bb-nodes/tests -q
```
Expected: `23 passed` (matching baseline).

---

### Task 6: Verify the split didn't change module structure

**Files:**
- Read: `custom_nodes/comfy-bb-nodes/text_nodes.py`
- Read: `custom_nodes/comfy-bb-nodes/video_nodes.py`
- Read: `custom_nodes/comfy-bb-nodes/__init__.py`

- [ ] **Step 1: Confirm by inspection that no symbol appears in two files and that the union of moved symbols covers every symbol that was in the old `__init__.py` except the entry point + extension class.** The only symbols allowed to appear in more than one file are imports shared across modules.

Run:
```bash
cd "E:\ComfyUI_windows_portable\ComfyUI\custom_nodes\comfy-bb-nodes" && git diff --stat
```
Expected: shows `__init__.py`, `text_nodes.py`, `video_nodes.py`, and the four test files as changed; no other files.

---

### Task 7: Commit

**Files:**
- Commit: all changes in `custom_nodes/comfy-bb-nodes`

- [ ] **Step 1: Stage and commit**

```bash
cd "E:\ComfyUI_windows_portable\ComfyUI\custom_nodes\comfy-bb-nodes" && git add __init__.py text_nodes.py video_nodes.py tests docs/superpowers/plans/2026-08-19-refactor-by-category.md && git commit -m "Split nodes into text and video modules"
```

- [ ] **Step 2: Confirm commit**

Run:
```bash
cd "E:\ComfyUI_windows_portable\ComfyUI\custom_nodes\comfy-bb-nodes" && git log --oneline -3
```
Expected: newest commit is `Split nodes into text and video modules`.

---

## Self-review notes

- **Spec coverage:** Design spec requires `text_nodes.py` (Task 1), `video_nodes.py` (Task 2), thin `__init__.py` + `BBExtension` rename (Task 3), test import retargets + monkeypatch relocation (Task 4), passing suite (Task 5), unchanged structure (Task 6), commit (Task 7). All spec items covered.
- **Placeholder scan:** No TBD/TODO; moved-code tasks reference the verbatim source file explicitly.
- **Type consistency:** `BBExtension` used consistently; `comfy_entrypoint() -> BBExtension` matches the class in Task 3. Test module names `comfy_bb_text_nodes` / `comfy_bb_video_nodes` are unique per file and do not collide across tests.
