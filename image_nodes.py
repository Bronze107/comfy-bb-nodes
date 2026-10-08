import math

import torch

import comfy.utils
import node_helpers
from comfy_api.latest import io


def resize_mask(mask, height, width):
    # a mask can arrive at a different resolution than the image; put it on the image grid
    if mask.shape[-2:] == (height, width):
        return mask
    return torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(height, width), mode="bilinear").reshape(mask.shape[0], height, width)


def masked_region_box(mask, padding, height, width):
    # WebUI "only masked" box: mask bounding box grown by padding, aligned to the 8px VAE stride
    m = mask.amax(dim=0) if mask.dim() == 3 else mask
    rows, cols = torch.where(m > 0.5)
    if rows.numel() == 0:
        raise ValueError("Mask is empty, nothing to inpaint")
    x0 = max(int(cols.min()) - padding, 0)
    y0 = max(int(rows.min()) - padding, 0)
    x1 = min(int(cols.max()) + 1 + padding, width)
    y1 = min(int(rows.max()) + 1 + padding, height)
    x0, x1 = (x0 // 8) * 8, min(math.ceil(x1 / 8) * 8, width)
    y0, y1 = (y0 // 8) * 8, min(math.ceil(y1 / 8) * 8, height)
    return x0, y0, x1, y1


class MaskedInpaintCrop(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MaskedInpaintCrop",
            display_name="Masked Inpaint Crop",
            category="image/inpaint",
            description="WebUI-style only-masked inpainting, crop half: cut the mask region plus surrounding "
            "padding out of the image. Run your inpaint sampling chain on the outputs, then paste the result "
            "back with Masked Inpaint Paste using the same mask and padding.",
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("mask"),
                io.Int.Input("padding", default=32, min=0, max=2048,
                             tooltip="Context pixels added around the mask bounding box on every side."),
                io.Int.Input("resolution", default=0, min=0, max=4096, step=8,
                             tooltip="Scale the crop so its longest side is this many pixels, e.g. 1024 for SDXL. 0 keeps the crop at its original size."),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="mask"),
            ],
        )

    @classmethod
    def execute(cls, image, mask, padding, resolution) -> io.NodeOutput:
        mask = resize_mask(mask, image.shape[1], image.shape[2])
        x0, y0, x1, y1 = masked_region_box(mask, padding, image.shape[1], image.shape[2])
        crop = image[:, y0:y1, x0:x1, :]
        crop_mask = mask[:, y0:y1, x0:x1]
        height, width = crop.shape[1:3]
        if resolution > 0 and max(height, width) != resolution:
            scale = resolution / max(height, width)
            width = max(round(width * scale / 8) * 8, 64)
            height = max(round(height * scale / 8) * 8, 64)
            crop = comfy.utils.common_upscale(crop.movedim(-1, 1), width, height, "lanczos", "disabled").movedim(1, -1)
            crop_mask = resize_mask(crop_mask, height, width)
        return io.NodeOutput(crop, crop_mask)


class MaskedInpaintPaste(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MaskedInpaintPaste",
            display_name="Masked Inpaint Paste",
            category="image/inpaint",
            description="WebUI-style only-masked inpainting, paste half: resize the inpainted crop back and "
            "composite it into the original image through the mask. Use the same mask and padding as Masked Inpaint Crop.",
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("mask"),
                io.Image.Input("inpainted"),
                io.Int.Input("padding", default=32, min=0, max=2048,
                             tooltip="Must match the padding used in Masked Inpaint Crop."),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, image, mask, inpainted, padding) -> io.NodeOutput:
        mask = resize_mask(mask, image.shape[1], image.shape[2])
        image, inpainted = node_helpers.image_alpha_fix(image, inpainted)
        x0, y0, x1, y1 = masked_region_box(mask, padding, image.shape[1], image.shape[2])
        region = comfy.utils.common_upscale(inpainted.movedim(-1, 1), x1 - x0, y1 - y0, "lanczos", "disabled").movedim(1, -1)
        m = mask[:, y0:y1, x0:x1].unsqueeze(-1)
        out = image.clone()
        out[:, y0:y1, x0:x1, :] = image[:, y0:y1, x0:x1, :] * (1.0 - m) + region * m
        return io.NodeOutput(out)


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
