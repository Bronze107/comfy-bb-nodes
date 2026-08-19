# Refactor comfy-bb-nodes by Category

## Goal

Split the single `__init__.py` (which currently holds all 6 nodes and helpers) into
per-category modules so each file has one clear responsibility. No node behavior,
schema, node ID, or category changes.

## Target structure

```
comfy-bb-nodes/
├── __init__.py      # imports the 6 node classes + BBExtension + comfy_entrypoint
├── text_nodes.py    # wildcard helpers + LoadWildcardFile / RandomFromList / WildcardReplace
├── video_nodes.py   # ffmpeg helpers + VideoInfo / ConcatVideos / ResampleFPS
├── res/             # unchanged
├── workflows/       # unchanged
└── tests/           # import targets updated (minimal change)
```

## File contents

**`text_nodes.py`** — moved verbatim from `__init__.py`:
- Helpers: `resolve_path`, `load_wildcard_lines`, `pick_random_line`, `wildcard_replace`
- Nodes: `LoadWildcardFile`, `RandomFromList`, `WildcardReplace`
- Imports: `re`, `random`, `os`, `folder_paths` path helpers (`base_path`,
  `exists_annotated_filepath`, `get_annotated_filepath`)

**`video_nodes.py`** — moved verbatim from `__init__.py`:
- Helpers: `_find_ffmpeg_tool`, `_run_process`, `_materialize_video`, `_probe_video`,
  `_stream_signature`, `_concat_list_line`, `_write_concat_list`, `_output_path`,
  `_concat_stream_copy`, `_concat_reencode`
- Nodes: `VideoInfo`, `ConcatVideos`, `ResampleFPS`
- Imports: `json`, `os`, `shutil`, `subprocess`, `uuid`, `VideoFromFile`,
  `io`, `ui`, `Types`, `folder_paths` output/temp helpers

**`__init__.py`** — becomes the entry point only:
- Imports the 6 node classes from the two submodules via `sys.path` insertion +
  absolute imports (see constraint below)
- `class BBExtension(ComfyExtension)` with `get_node_list` returning all 6 nodes
  (renamed from `WildcardExtension`, which no longer reflects the content)
- `comfy_entrypoint()` returns `BBExtension()`
- Imports: `os`, `sys`, `typing_extensions.override`, `ComfyExtension`, `io`

## Import mechanism constraint (verified empirically)

ComfyUI's `load_custom_node` (nodes.py:2258) loads a directory custom node with
`importlib.util.spec_from_file_location` on `__init__.py`, registering the module
in `sys.modules` before `exec_module`. Relative imports would work there, but the
directory name `comfy-bb-nodes` is not a valid Python module name (hyphens), so
pytest collects the directory as `<Package comfy-bb-nodes>` and imports
`__init__.py` as a top-level module with an empty `__package__` — which breaks
relative imports during test runs.

The working approach: `__init__.py` inserts its own directory at the front of
`sys.path` and imports the sibling modules with absolute imports. This works both
under ComfyUI's loader and under pytest (any collection mode).

The submodule files (`text_nodes.py`, `video_nodes.py`) have no relative imports,
so tests load them directly.

Verified under `python_embeded/python.exe` (3.13.11): plain
`pytest custom_nodes/comfy-bb-nodes/tests` passes, and a replica of ComfyUI's
`load_custom_node` directory-loading imports all 6 nodes without error.

## Test changes

| File | Change |
|---|---|
| `tests/test_wildcards.py` | `_IMPL` → `text_nodes.py` |
| `tests/test_video_info.py` | `_IMPL` → `video_nodes.py` |
| `tests/test_concat_videos.py` | `_IMPL` → `video_nodes.py`; monkeypatch `video_nodes.get_output_directory` |
| `tests/test_resample_fps.py` | `_IMPL` → `video_nodes.py`; monkeypatch `video_nodes.get_output_directory` |

The monkeypatch target moves because `get_output_directory` is resolved in the
module namespace of the video code, which is now `video_nodes`.

## Compatibility guarantees

- Node IDs, display names, categories, schemas, and `execute` behavior unchanged.
- `comfy_entrypoint` remains the entrypoint; ComfyUI only needs that name.
- `res/` and `workflows/` untouched.

## Non-goals

- Do not change any node behavior or schema.
- Do not add abstractions, utils modules, or shared helpers beyond the split.
- Do not reorganize `res/` or `workflows/`.

## Verification

- `pytest custom_nodes/comfy-bb-nodes/tests` passes (baseline: 23 passed).
- Replicate `load_custom_node` directory-loading for `__init__.py` to confirm no
  import errors after the split.
