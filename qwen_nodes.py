import math

import torch

import comfy.utils
import comfy.model_management
import node_helpers
from comfy_api.latest import io


class TextEncodeQwenImage21Latent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeQwenImage21Latent",
            display_name="Text Encode Qwen Image 2.1 (Latent Refs)",
            category="model/conditioning/qwen image",
            description="Same as Text Encode Qwen Image 2.1, but each reference image can take a pre-encoded "
            "latent instead of being VAE-encoded here, so an unchanged reference skips re-encoding on every run.",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("negative_prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("resolution", default=1024, min=0, max=4096, step=32,
                             tooltip="Reference images are resized to about resolution x resolution pixels, at multiples of 32, preserving aspect ratio. 0 keeps each reference at its own size, rounded to a multiple of 32."),
                io.Autogrow.Input(
                    "images",
                    template=io.Autogrow.TemplateNames(
                        io.Image.Input("image"),
                        names=[f"image_{i}" for i in range(1, 17)],
                        min=0,
                    ),
                    tooltip="Reference images, seen by the text encoder and spliced into the sequence as VAE latents.",
                ),
                io.Autogrow.Input(
                    "latents",
                    template=io.Autogrow.TemplateNames(
                        io.Latent.Input("latent"),
                        names=[f"latent_{i}" for i in range(1, 17)],
                        min=0,
                    ),
                    tooltip="Optional pre-encoded latent for image_i (from VAEEncode on the same image); when connected it replaces the VAE encode for that slot.",
                ),
                io.Vae.Input("vae", optional=True,
                             tooltip="Fallback VAE for reference images without a matching latent_i. Leave empty when every reference has a latent."),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent",
                                 tooltip="Empty latent on the first reference image's size, to match with sampling as any other size shifts the edit."),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, negative_prompt, resolution=1024, images=None, latents=None, vae=None) -> io.NodeOutput:
        ref_latents = []
        images_vl = []
        images = images or {}
        latents = latents or {}
        latent_w = latent_h = resolution or 1024
        for name in sorted(images, key=lambda n: int(n.rsplit("_", 1)[-1])):
            image = images[name]
            if image is None:
                continue
            # same resize for the text encoder and the VAE, so every vision slot covers 2x2 latents; one image per input
            samples = image[:1].movedim(-1, 1)
            if resolution > 0:
                ratio = samples.shape[3] / samples.shape[2]
                width = round(math.sqrt(resolution * resolution * ratio) / 32) * 32
                height = round(math.sqrt(resolution * resolution / ratio) / 32) * 32
            else:
                width, height = round(samples.shape[3] / 32) * 32, round(samples.shape[2] / 32) * 32
            width, height = max(32, width), max(32, height)
            if (width, height) == (samples.shape[3], samples.shape[2]):
                s = image[:1]
            else:
                s = comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled").movedim(1, -1)
            if not images_vl:
                latent_w, latent_h = width, height
            rgb = s[:, :, :, :3]
            if s.shape[-1] > 3:
                rgb = rgb * s[:, :, :, 3:] + (1.0 - s[:, :, :, 3:])  # the vision tower sees alpha over white, the vae keeps all four
            images_vl.append(rgb)
            ref = latents.get("latent_{}".format(name.rsplit("_", 1)[-1]))
            if ref is not None:
                ref_latents.append(ref["samples"][:1])
            elif vae is not None:
                ref_latents.append(vae.encode(s))

        keep_vision = len(ref_latents) == 0
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt, images=images_vl, keep_vision=keep_vision, prevent_empty_text=True))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt, images=images_vl, keep_vision=keep_vision, prevent_empty_text=True))
        if len(ref_latents) > 0:
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": ref_latents}, append=True)
            negative = node_helpers.conditioning_set_values(negative, {"reference_latents": ref_latents}, append=True)
        latent = torch.zeros([1, 64, latent_h // 16, latent_w // 16], device=comfy.model_management.intermediate_device())
        return io.NodeOutput(positive, negative, {"samples": latent})
