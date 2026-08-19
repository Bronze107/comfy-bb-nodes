# ResampleFPS Refactor: Add Megapixel-Based Scaling — Design

## Goal

Refactor the existing `ResampleFPS` node in `comfy-bb-nodes` so it can also
scale a video by a target megapixel count (aspect-preserving, downscale-only),
while keeping the existing fps resampling and workflow compatibility.

## Requirements (confirmed with user)

- Keep `node_id = "ResampleFPS"` so existing saved workflows keep loading.
- Add a `megapixels` input: scale the video to that many megapixels,
  preserving the aspect ratio.
- Downscale only: if the target exceeds the source resolution, keep the source
  unchanged (never upscale).
- `fps = 0` means "keep the original frame rate" (only adjust resolution).
- Change `display_name` to "Resample Video" (node_id unchanged), add search
  aliases for resize.

## Node definition

Modified in `custom_nodes/comfy-bb-nodes/video_nodes.py`.

- **node_id**: `ResampleFPS` (unchanged)
- **display_name**: "Resample Video"
- **category**: `video` (unchanged)
- **search_aliases**: `resample fps`, `adjust fps`, `change frame rate`,
  `convert fps`, `set fps`, `resize`, `scale resolution`, `adjust resolution`
- **inputs** (order `video, fps, megapixels, filename_prefix`):
  - `video` (`io.Video`)
  - `fps` (`io.Float`, default `24.0`, min `0`, max `240`) — `0` keeps the source fps
  - `megapixels` (`io.Float`, default `2.0`, min `0`) — `0` keeps the source resolution
  - `filename_prefix` (`io.String`, default `video/ResampleFPS`)
- **outputs**:
  - `video` (`io.Video.Output`) — unchanged

## Implementation

The node keeps its current flow: materialize the input to a temp file path,
probe it for the reference video stream (width/height/audio), build an ffmpeg
filter chain, re-encode to the output directory, and return a `VideoFromFile`
with a preview.

New execute signature and filter building:

```python
@classmethod
def execute(cls, video, fps, megapixels, filename_prefix):
    temp_dir = get_temp_directory()
    path, is_temp = _materialize_video(video, temp_dir)
    temp_paths = [path] if is_temp else []
    try:
        ref_video, audio = _probe_video(path)
        vf = []
        if fps > 0:
            vf.append(f"fps={fps:g}")
        if megapixels > 0:
            width = ref_video.get("width")
            height = ref_video.get("height")
            if not width or not height:
                raise RuntimeError("Could not determine the source video's dimensions.")
            factor = math.sqrt((megapixels * 1_000_000) / (width * height))
            if factor < 1.0:  # downscale only
                new_w = round(width * factor) - (round(width * factor) % 2)
                new_h = round(height * factor) - (round(height * factor) % 2)
                vf.append(f"scale={new_w}:{new_h}:flags=lanczos")
        if not vf:
            raise ValueError("Set fps > 0 or megapixels > 0 to change the video.")
        out_path, file, subfolder = _output_path(filename_prefix, ref_video)
        cmd = [
            _find_ffmpeg_tool("ffmpeg"),
            "-y",
            "-i", path,
            "-vf", ",".join(vf),
            "-c:v", "libx264",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ]
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
- `math` is added to the `video_nodes.py` imports.
- `scale` uses `flags=lanczos` for quality; computed `w`/`h` are rounded to even
  numbers so `yuv420p` stays valid.
- The aspect ratio is preserved because `w` and `h` derive from the same scale
  factor; the even-rounding deviation is under half a percent.
- Downscale-only: `factor < 1.0` gates the scale filter.

## Edge cases

- `fps = 0` and `megapixels = 0` → `ValueError` ("Set fps > 0 or megapixels > 0").
- `megapixels` larger than the source → factor `>= 1.0`, no scale filter, source
  resolution preserved.
- Source dimensions missing → `RuntimeError`.

## Defaults note

`megapixels` defaults to `2.0`, so a freshly placed node scales to ~2 MP by
default. This is an intentional behavior change for the new capability; setting
`0` keeps the source resolution. `fps` keeps its `24.0` default, preserving
existing fresh-node behavior.

## Testing

`custom_nodes/comfy-bb-nodes/tests/test_resample_fps.py` is updated for the new
signature: `ResampleFPS.execute(video, 24.0, 0.0, "test_resample")` keeps the
existing fps-only assertion.

New tests in `test_resample_fps.py`:
- `fps=0, megapixels>0` → output frame count/duration matches the source, output
  dimensions are smaller, and `new_w / new_h` ratio matches the source ratio
  (within tolerance).
- `megapixels` far above the source → output dimensions unchanged (no upscale).
- `fps=0, megapixels=0` → `ValueError`.

## Non-goals

- No new node; `ResampleFPS` is extended in place.
- No crop/pad modes, no width/height inputs, no upscale.
- No changes to `ConcatVideos` or other nodes.

## Verification

- `pytest custom_nodes/comfy-bb-nodes/tests` passes (27 existing + new).
- ComfyUI loader replica imports the updated node without error.
