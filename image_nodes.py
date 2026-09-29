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
