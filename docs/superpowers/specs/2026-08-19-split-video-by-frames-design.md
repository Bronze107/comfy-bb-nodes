# Split Video by Frames — Design

## Goal

Add a `SplitVideoByFrames` node to `comfy-bb-nodes` that splits a video into
segments of a fixed number of frames each, outputting the segments as a lazy
list of videos. This complements the core `GetVideoComponents` node (which
extracts individual frame images) by producing video clips.

## Requirements (confirmed with user)

- Split a video into multiple video segments, each holding `N` frames.
- Output form: a list of `io.Video` segments that are lazy trim windows over the
  source (no files written, no re-encode), consistent with core `VideoSlice`.
- When the total frame count is not a multiple of `N`, keep the shorter final
  segment rather than dropping it.
- Audio is preserved per segment.

## Node definition

Added to `custom_nodes/comfy-bb-nodes/video_nodes.py`, registered in
`__init__.py`.

- **node_id**: `SplitVideoByFrames`
- **display_name**: "Split Video by Frames"
- **category**: `video`
- **search_aliases**: `split video by frames`, `frame segments`, `cut video into segments`
- **inputs**:
  - `video` (`io.Video`) — the video to split.
  - `frames_per_segment` (`io.Int`, default `30`, min `1`) — frames in each segment.
- **outputs**:
  - `segments` (`io.Video.Output`, `is_output_list=True`) — the split segments.

## Implementation

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
        segments = []
        for start in range(0, frame_count, frames_per_segment):
            start_time = start / fps
            duration = min(frames_per_segment, frame_count - start) / fps
            segment = video.as_trimmed(float(start_time), float(duration), strict_duration=False)
            if segment is not None:
                segments.append(segment)
        return io.NodeOutput(segments)
```

Notes:
- `video.get_frame_rate()` returns a `Fraction`; dividing frame indices by it
  yields exact rational times, converted to `float` at the `as_trimmed` boundary.
- `as_trimmed(..., strict_duration=False)` never returns `None` for valid
  windows, but the code keeps the guard for safety.
- Frame-count reporting on a trimmed `VideoFromFile` uses metadata estimates
  (may be `±1` on unusual frame rates); actual decoded output is frame-accurate.
  This matches core `VideoSlice` behavior.

## Edge cases

- `frames_per_segment >= total frames` → a single segment holding the whole video.
- Last segment shorter than `frames_per_segment` → kept as-is.
- Zero-frame video → `ValueError`.
- `frames_per_segment < 1` → prevented by schema `min=1`.

## Testing

New `custom_nodes/comfy-bb-nodes/tests/test_split_video.py`, using
`VideoFromComponents` (which implements `as_trimmed` by wrapping its stream in a
`VideoFromFile` with a trim window).

- Split a 30 fps, 100-frame video with `frames_per_segment=60` → 2 segments with
  frame counts `[60, 40]`.
- Split the same video with `frames_per_segment=100` → 1 segment with 100 frames.
- Split with `frames_per_segment=30` → 4 segments (last has 10), each reported
  frame count equals the segment's frames, and total reported duration matches
  the source.

## Non-goals

- No file writing, re-encode, or audio options. Segments are lazy references.
- No frame-range output or segment indexing output.
- No changes to existing nodes or schemas.

## Verification

- `pytest custom_nodes/comfy-bb-nodes/tests` passes (existing 23 + new tests).
- ComfyUI loader replica imports the new node without error.
