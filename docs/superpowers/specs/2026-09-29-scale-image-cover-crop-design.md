# Scale Image to Total Pixels (Cover Crop) — Design

## Goal

Add a new image node to `comfy-bb-nodes` that scales an image to a target
megapixel count with step-aligned output dimensions, using uniform scaling plus
center-crop so the image content is never stretched by the step rounding.

## Background

Core ComfyUI's `ImageScaleToTotalPixels` rounds width and height independently
to `resolution_steps` multiples, then resizes directly to that W×H. Because the
two roundings differ, `W/w != H/h` and the direct resize stretches the content
by up to ~1 step worth of aspect error. This node replaces that stretch with a
crop: scale uniformly to cover the target, then crop the excess from the
center.

## Behavior (confirmed with user)

- Cover + crop: target `W`/`H` are rounded to `resolution_steps` multiples the
  same way as the core node; the image is then scaled by the larger factor
  needed to cover `W×H` (uniform, zero stretch) and center-cropped down to
  exactly `W×H`. One axis loses at most about one step of edge content.

## Node definition

New file `custom_nodes/comfy-bb-nodes/image_nodes.py`, registered in
`custom_nodes/comfy-bb-nodes/__init__.py`.

- **node_id**: `ScaleImageToTotalPixelsCoverCrop`
- **display_name**: "Scale Image to Total Pixels (Cover Crop)"
- **category**: `image/upscaling`
- **inputs**:
  - `image` (`io.Image`) — batched input handled uniformly
  - `upscale_method` (`io.Combo`): `nearest-exact`, `bilinear`, `area`,
    `bicubic`, `lanczos`
  - `megapixels` (`io.Float`, default `1.0`, min `0.01`, max `16.0`, step `0.01`)
  - `resolution_steps` (`io.Int`, default `1`, min `1`, max `256`, advanced)
- **outputs**:
  - `image` (`io.Image.Output`) — exactly `W×H`, step-aligned

## Implementation

```python
scale_by = math.sqrt((megapixels * 1024 * 1024) / (w * h))
W = round(w * scale_by / steps) * steps
H = round(h * scale_by / steps) * steps
cover = max(W / w, H / h)
sw, sh = round(w * cover), round(h * cover)
s = comfy.utils.common_upscale(image.movedim(-1, 1), sw, sh, upscale_method, "disabled")
x0 = (sw - W) // 2
y0 = (sh - H) // 2
s = s[:, :, y0:y0 + H, x0:x0 + W].movedim(1, -1)
```

Notes:
- `comfy.utils.common_upscale` handles both upscale and downscale; no new
  dependency, same helper the core node uses.
- Crop is a direct tensor slice on the batched tensor; all batch frames crop
  identically.
- `resolution_steps = 1` reduces to proportional scaling with a sub-pixel crop.

## Edge cases

- Rounding to nearest can make `W` slightly smaller than `w * scale_by`; the
  `max()` cover factor still guarantees `sw >= W` and `sh >= H`, so the crop
  slices are always valid and non-negative.
- `steps` larger than the scaled size: `W` or `H` rounds to 0 only if
  `w * scale_by < steps / 2`; clamp `W`/`H` to at least `steps` to keep output
  sane.

## Testing

New `custom_nodes/comfy-bb-nodes/tests/test_image_scale_cover_crop.py`:

- Output shape is exactly `W×H` with `W`, `H` multiples of `resolution_steps`.
- `W*H` is within one step row/column of the target megapixel count.
- No-stretch property: content scaled by a single uniform factor — verify with
  a synthetic pattern (e.g., a diagonal or checkerboard) that pixel spacing in
  `x` and `y` matches within tolerance.
- Center-crop symmetry: feeding a shifted-pattern image and comparing borders,
  or verifying `x0`/`y0` selection on a known gradient.
- Batch input: every frame in the batch has the same output size.

## Non-goals

- No mask support, no crop-offset output, no crop-direction option.
- No changes to existing nodes.

## Verification

- `pytest custom_nodes/comfy-bb-nodes/tests` passes (existing + new).
- ComfyUI loads the new node without error.
