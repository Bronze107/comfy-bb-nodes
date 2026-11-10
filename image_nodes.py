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


def masked_region_box(mask, padding, height, width, alignment=8):
    # WebUI "only masked" box: mask bounding box grown by padding, aligned to the VAE stride
    m = mask.amax(dim=0) if mask.dim() == 3 else mask
    rows, cols = torch.where(m > 0.5)
    if rows.numel() == 0:
        raise ValueError("Mask is empty, nothing to inpaint")
    x0 = max(int(cols.min()) - padding, 0)
    y0 = max(int(rows.min()) - padding, 0)
    x1 = min(int(cols.max()) + 1 + padding, width)
    y1 = min(int(rows.max()) + 1 + padding, height)
    x0, x1 = (x0 // alignment) * alignment, min(math.ceil(x1 / alignment) * alignment, width)
    y0, y1 = (y0 // alignment) * alignment, min(math.ceil(y1 / alignment) * alignment, height)
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
                io.Float.Input("megapixels", default=1.0, min=0, max=16.0, step=0.01,
                               tooltip="Scale the crop to this many megapixels, keeping aspect ratio. 0 keeps the crop at its original size."),
                io.Int.Input("alignment", default=8, min=8, max=256, step=8,
                             tooltip="Round the crop box and scaled size to this pixel step: 8 for SD/SDXL/Flux/Qwen-Image VAEs, 32 for Qwen-Image 2.1. Must match Masked Inpaint Paste."),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="mask"),
            ],
        )

    @classmethod
    def execute(cls, image, mask, padding, megapixels, alignment=8) -> io.NodeOutput:
        mask = resize_mask(mask, image.shape[1], image.shape[2])
        x0, y0, x1, y1 = masked_region_box(mask, padding, image.shape[1], image.shape[2], alignment)
        crop = image[:, y0:y1, x0:x1, :]
        crop_mask = mask[:, y0:y1, x0:x1]
        height, width = crop.shape[1:3]
        if megapixels > 0:
            scale = math.sqrt(megapixels * 1024 * 1024 / (width * height))
            width = max(round(width * scale / alignment) * alignment, alignment)
            height = max(round(height * scale / alignment) * alignment, alignment)
            if (height, width) != crop.shape[1:3]:
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
                io.Int.Input("alignment", default=8, min=8, max=256, step=8,
                             tooltip="Must match the alignment used in Masked Inpaint Crop."),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, image, mask, inpainted, padding, alignment=8) -> io.NodeOutput:
        mask = resize_mask(mask, image.shape[1], image.shape[2])
        image, inpainted = node_helpers.image_alpha_fix(image, inpainted)
        x0, y0, x1, y1 = masked_region_box(mask, padding, image.shape[1], image.shape[2], alignment)
        region = comfy.utils.common_upscale(inpainted.movedim(-1, 1), x1 - x0, y1 - y0, "lanczos", "disabled").movedim(1, -1)
        m = mask[:, y0:y1, x0:x1].unsqueeze(-1)
        out = image.clone()
        out[:, y0:y1, x0:x1, :] = image[:, y0:y1, x0:x1, :] * (1.0 - m) + region * m
        return io.NodeOutput(out)


class RicohGR3HDF(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RicohGR3HDF",
            display_name="Ricoh GR III HDF",
            category="image/filters",
            description="Simulate the Ricoh GR III HDF highlight diffusion filter: bright areas bloom with "
            "halation glow while midtones stay sharp, global contrast drops like a black mist filter, plus "
            "the mild wide-open vignette of the 28mm f/2.8 and fine APS-C grain. The contrast and saturation "
            "knobs give the muted Negative Film palette HDF shooters often pair with it.",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True,
                             tooltip="Seed for the grain."),
                io.Float.Input("glow_strength", default=0.5, min=0.0, max=1.0, step=0.01,
                               tooltip="Highlight bloom amount; strongest on point lights, neon and speculars."),
                io.Float.Input("glow_threshold", default=0.65, min=0.3, max=0.95, step=0.01,
                               tooltip="Luminance where diffusion starts."),
                io.Int.Input("glow_radius", default=16, min=0, max=256,
                             tooltip="Bloom spread in pixels; 0 disables the diffusion."),
                io.Float.Input("mist_strength", default=0.15, min=0.0, max=1.0, step=0.01,
                               tooltip="Global black-mist veiling: lowers contrast and lifts blacks."),
                io.Float.Input("contrast", default=0.95, min=0.5, max=1.5, step=0.01,
                               tooltip="The diffusion filter lowers overall contrast; below 1 softens the tone curve."),
                io.Float.Input("saturation", default=0.95, min=0.5, max=1.2, step=0.01,
                               tooltip="Below 1 gives the faded Negative Film look."),
                io.Float.Input("vignette", default=0.10, min=0.0, max=0.5, step=0.01,
                               tooltip="Mild corner falloff of the 28mm f/2.8 wide open."),
                io.Float.Input("grain_strength", default=0.008, min=0.0, max=0.1, step=0.001,
                               tooltip="Fine APS-C sensor grain."),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, image, seed, glow_strength, glow_threshold, glow_radius, mist_strength, contrast,
                saturation, vignette, grain_strength) -> io.NodeOutput:
        return io.NodeOutput(ricoh_gr3_hdf(image, seed, glow_strength, glow_threshold, glow_radius,
                                           mist_strength, contrast, saturation, vignette, grain_strength))


class CCTVBodycam(io.ComfyNode):
    cast_factors = {"green": [-0.06, 0.08, -0.10], "magenta": [0.07, -0.08, 0.07], "amber": [0.08, 0.02, -0.20]}

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CCTVBodycam",
            display_name="CCTV / Bodycam Footage",
            category="image/filters",
            description="Simulate cheap surveillance footage: H.264 8x8 macroblock flattening, wide-angle "
            "barrel distortion, heavy sensor noise, faint scanlines, muted colors with a sensor/cabling "
            "color cast, cheap-lens vignette, and an optional IR night mode with green monochrome rendering "
            "and the center hotspot of the IR illuminator.",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True,
                             tooltip="Seed for the sensor noise."),
                io.Float.Input("noise_strength", default=0.04, min=0.0, max=0.2, step=0.005,
                               tooltip="Heavy sensor noise, strongest on cheap low-light cameras."),
                io.Float.Input("blockiness", default=0.5, min=0.0, max=1.0, step=0.01,
                               tooltip="Blend toward 8x8 macroblock flattening like low-bitrate H.264."),
                io.Float.Input("fisheye", default=0.15, min=0.0, max=0.5, step=0.01,
                               tooltip="Wide-angle barrel distortion: straight lines bow outward, edges compress."),
                io.Float.Input("scanlines", default=0.15, min=0.0, max=1.0, step=0.01,
                               tooltip="Faint line pattern from analog / interlaced capture."),
                io.Float.Input("saturation", default=0.85, min=0.0, max=1.5, step=0.01,
                               tooltip="Cheap sensors render muted colors."),
                io.Combo.Input("color_cast", options=["off", "green", "magenta", "amber"],
                               tooltip="Sensor/cabling color shift: green (common fault), magenta (failed IR-cut "
                               "bleed), amber (poor white balance under street lighting)."),
                io.Float.Input("cast_strength", default=0.5, min=0.0, max=1.0, step=0.01,
                               tooltip="How strong the color cast is."),
                io.Boolean.Input("night_ir", default=False,
                                 tooltip="IR night mode: green monochrome plus the overexposed hotspot "
                                 "in front of the IR illuminator."),
                io.Float.Input("vignette", default=0.15, min=0.0, max=0.5, step=0.01,
                               tooltip="Corner falloff of a cheap wide lens."),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, image, seed, noise_strength, blockiness, fisheye, scanlines, saturation, color_cast,
                cast_strength, night_ir, vignette) -> io.NodeOutput:
        return io.NodeOutput(cctv_bodycam(image, seed, noise_strength, blockiness, fisheye, scanlines,
                                          saturation, color_cast, cast_strength, night_ir, vignette))


def cctv_bodycam(image, seed, noise_strength, blockiness, fisheye, scanlines, saturation, color_cast,
                 cast_strength, night_ir, vignette):
    height, width = image.shape[1:3]
    rgb, alpha = image[..., :3], image[..., 3:]
    if fisheye > 0:
        # wide-angle barrel distortion: sample inward as the radius grows
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, height, device=rgb.device),
                                torch.linspace(-1, 1, width, device=rgb.device), indexing="ij")
        shrink = (1.0 - fisheye * (xx.square() + yy.square())).clamp(min=0.0)
        grid = torch.stack([xx * shrink, yy * shrink], dim=-1).unsqueeze(0).expand(rgb.shape[0], -1, -1, -1)
        rgb = torch.nn.functional.grid_sample(rgb.movedim(-1, 1), grid, mode="bilinear",
                                              padding_mode="border", align_corners=False).movedim(1, -1)
    if blockiness > 0:
        blocks = torch.nn.functional.avg_pool2d(rgb.movedim(-1, 1), 8)
        blocks = torch.nn.functional.interpolate(blocks, size=(height, width), mode="nearest").movedim(1, -1)
        rgb = rgb + (blocks - rgb) * blockiness
    if scanlines > 0:
        rows = (torch.arange(height, device=rgb.device) % 2).to(rgb.dtype)
        rgb = rgb * (1.0 - scanlines * 0.4 * rows).view(1, height, 1, 1)
    if night_ir:
        # IR cut filter off: monochrome sensor output tinted by the IR response,
        # with the illuminator's overexposed hotspot in the center
        lum = (rgb * torch.tensor([0.2126, 0.7152, 0.0722], device=rgb.device)).sum(dim=-1, keepdim=True)
        hotspot = 1.0 + 0.35 * (1.0 - radial_falloff(height, width, rgb.device)).unsqueeze(-1)
        rgb = lum * torch.tensor([0.35, 1.0, 0.45], device=rgb.device) * hotspot
    if saturation != 1.0:
        lum = (rgb * torch.tensor([0.2126, 0.7152, 0.0722], device=rgb.device)).sum(dim=-1, keepdim=True)
        rgb = lum + (rgb - lum) * saturation
    if color_cast != "off":
        rgb = rgb * (1.0 + cast_strength * torch.tensor(CCTVBodycam.cast_factors[color_cast], device=rgb.device))
    rgb = rgb * (1.0 - vignette * radial_falloff(height, width, rgb.device).unsqueeze(-1))
    if noise_strength > 0:
        generator = torch.manual_seed(seed)
        noise = torch.randn(rgb.shape[:-1] + (1,), generator=generator, device="cpu").to(rgb)
        rgb = rgb + noise * noise_strength
    return torch.cat([rgb.clamp(0.0, 1.0), alpha], dim=-1)


class IPhone5Camera(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="IPhone5Camera",
            display_name="iPhone 5 Camera",
            category="image/filters",
            description="Simulate the 2012 iPhone 5 iSight look from its documented characteristics: 8 MP "
            "resolution cap, luminance noise in flat areas, ~8% vignetting with pink-center/green-corner "
            "color shading, soft small-lens rendering, warm white balance, chroma-subsampled JPEG output, "
            "and the purple flare Apple called normal in support doc TS4436.",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True,
                             tooltip="Seed for the sensor noise."),
                io.Float.Input("noise_strength", default=0.035, min=0.0, max=0.2, step=0.005,
                               tooltip="Gray luminance noise, visible in plain areas like blue sky even at base ISO."),
                io.Float.Input("softness", default=0.25, min=0.0, max=1.0, step=0.01,
                               tooltip="Blend toward a slight lens blur; the small f/2.4 module is never razor sharp."),
                io.Float.Input("vignette", default=0.08, min=0.0, max=0.5, step=0.01,
                               tooltip="Corner darkening; DxOMark measured ~8% luminance shading."),
                io.Float.Input("color_shading", default=0.5, min=0.0, max=1.0, step=0.01,
                               tooltip="Pinkish center, greenish corners (DxOMark color shading)."),
                io.Float.Input("purple_haze", default=0.35, min=0.0, max=1.0, step=0.01,
                               tooltip="Purple veil over bright areas, the iPhone 5's documented flare behavior."),
                io.Float.Input("warmth", default=0.04, min=0.0, max=0.3, step=0.01,
                               tooltip="Warm white-balance shift, like tungsten-lit shots."),
                io.Boolean.Input("resolution_cap", default=True,
                                 tooltip="Downscale images larger than 3264x2448 (8 MP) like the original sensor."),
                io.Boolean.Input("chroma_subsampling", default=True,
                                 tooltip="Average chroma 2x2 like 4:2:0 JPEG output; luma detail is kept."),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, image, seed, noise_strength, softness, vignette, color_shading, purple_haze, warmth,
                resolution_cap=True, chroma_subsampling=True) -> io.NodeOutput:
        return io.NodeOutput(iphone5_camera(image, seed, noise_strength, softness, vignette, color_shading,
                                            purple_haze, warmth, resolution_cap, chroma_subsampling))


def radial_falloff(height, width, device):
    # normalized squared corner distance in [0, 1]: 0 at center, 1 at corners
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, height, device=device),
                            torch.linspace(-1, 1, width, device=device), indexing="ij")
    return (xx.square() + yy.square()) * 0.5


def bloom(rgb, radius):
    # soft optical spread: downscale by the radius so a pixel of light bleeds
    # over roughly that many pixels, then upscale bilinearly
    factor = max(int(radius), 1)
    height, width = rgb.shape[1:3]
    x = rgb.movedim(-1, 1)
    small = torch.nn.functional.interpolate(x, size=(math.ceil(height / factor), math.ceil(width / factor)),
                                            mode="bilinear", antialias=True)
    return torch.nn.functional.interpolate(small, size=(height, width), mode="bilinear").movedim(1, -1)


def ricoh_gr3_hdf(image, seed, glow_strength, glow_threshold, glow_radius, mist_strength, contrast,
                  saturation, vignette, grain_strength):
    height, width = image.shape[1:3]
    rgb, alpha = image[..., :3], image[..., 3:]
    if glow_radius > 0 and glow_strength > 0:
        # highlight diffusion: light above the threshold blooms outward, screen-blended
        lum = (rgb * torch.tensor([0.2126, 0.7152, 0.0722], device=rgb.device)).sum(dim=-1, keepdim=True)
        mask = ((lum - glow_threshold) / (1.0 - glow_threshold)).clamp(0.0, 1.0).square()
        glow = bloom(rgb * mask, glow_radius) * glow_strength
        rgb = 1.0 - (1.0 - rgb) * (1.0 - glow)
    if mist_strength > 0:
        # black-mist veiling: scattered light lifts the whole tonal range
        veil = bloom(rgb, max(glow_radius * 2, 32)) * mist_strength
        rgb = 1.0 - (1.0 - rgb) * (1.0 - veil)
    if contrast != 1.0:
        rgb = (rgb - 0.5) * contrast + 0.5
    if saturation != 1.0:
        lum = (rgb * torch.tensor([0.2126, 0.7152, 0.0722], device=rgb.device)).sum(dim=-1, keepdim=True)
        rgb = lum + (rgb - lum) * saturation
    rgb = rgb * (1.0 - vignette * radial_falloff(height, width, rgb.device).unsqueeze(-1))
    if grain_strength > 0:
        generator = torch.manual_seed(seed)
        noise = torch.randn(rgb.shape[:-1] + (1,), generator=generator, device="cpu").to(rgb)
        rgb = rgb + noise * grain_strength
    return torch.cat([rgb.clamp(0.0, 1.0), alpha], dim=-1)


def iphone5_camera(image, seed, noise_strength, softness, vignette, color_shading, purple_haze, warmth,
                   resolution_cap, chroma_subsampling):
    height, width = image.shape[1:3]
    if resolution_cap and height * width > 3264 * 2448:
        scale = math.sqrt(3264 * 2448 / (height * width))
        width = max(round(width * scale), 1)
        height = max(round(height * scale), 1)
        image = comfy.utils.common_upscale(image.movedim(-1, 1), width, height, "area", "disabled").movedim(1, -1)
    rgb, alpha = image[..., :3], image[..., 3:]
    # purple flare from out-of-frame bright light sources (Apple TS4436), screen-blended so highlights roll off
    lum = (rgb * torch.tensor([0.2126, 0.7152, 0.0722], device=rgb.device)).sum(dim=-1, keepdim=True)
    haze = lum * purple_haze * torch.tensor([0.45, 0.10, 0.55], device=rgb.device)
    rgb = 1.0 - (1.0 - rgb) * (1.0 - haze)
    # vignette (DxOMark ~8%) and color shading (pink center, green corners) share one radial falloff
    corner2 = radial_falloff(rgb.shape[1], rgb.shape[2], rgb.device).unsqueeze(-1)
    rgb = rgb * (1.0 - vignette * corner2)
    rgb = rgb * (1.0 + color_shading * corner2 * torch.tensor([-0.04, 0.05, -0.02], device=rgb.device))
    rgb = rgb * torch.tensor([1.0 + warmth, 1.0, 1.0 - warmth * 0.6], device=rgb.device)
    if softness > 0:
        blur = torch.nn.functional.avg_pool2d(rgb.movedim(-1, 1), 3, stride=1, padding=1, count_include_pad=False).movedim(1, -1)
        rgb = rgb + (blur - rgb) * softness
    if noise_strength > 0:
        generator = torch.manual_seed(seed)
        noise = torch.randn(rgb.shape[:-1] + (1,), generator=generator, device="cpu").to(rgb)
        rgb = rgb + noise * noise_strength
    if chroma_subsampling:
        r, g, b = rgb.unbind(-1)
        y = 0.299 * r + 0.587 * g + 0.114 * b
        chroma = torch.stack([(b - y) * 0.564 + 0.5, (r - y) * 0.713 + 0.5], dim=1)
        chroma = torch.nn.functional.interpolate(torch.nn.functional.avg_pool2d(chroma, 2), size=rgb.shape[1:3], mode="nearest")
        r = (chroma[:, 1] - 0.5) / 0.713 + y
        b = (chroma[:, 0] - 0.5) / 0.564 + y
        g = (y - 0.299 * r - 0.114 * b) / 0.587
        rgb = torch.stack([r, g, b], dim=-1)
    return torch.cat([rgb.clamp(0.0, 1.0), alpha], dim=-1)


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
                io.Boolean.Input("only_downscale", default=False,
                                 tooltip="Cap the scale at 1: images below the target megapixels keep their size "
                                 "(still step-aligned and center-cropped) instead of being upscaled."),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, image, upscale_method, megapixels, resolution_steps, only_downscale=False) -> io.NodeOutput:
        height, width = image.shape[-3:-1]
        total = megapixels * 1024 * 1024

        scale_by = math.sqrt(total / (width * height))
        if only_downscale:
            scale_by = min(scale_by, 1.0)
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
