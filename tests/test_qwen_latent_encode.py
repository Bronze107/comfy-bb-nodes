import importlib.util
from pathlib import Path

import pytest
import torch

_IMPL = Path(__file__).resolve().parent.parent / "qwen_nodes.py"
_spec = importlib.util.spec_from_file_location("comfy_bb_qwen_nodes", _IMPL)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)

TextEncodeQwenImage21Latent = _impl.TextEncodeQwenImage21Latent


class FakeClip:
    def __init__(self):
        self.calls = []

    def tokenize(self, prompt, images=(), keep_vision=False, **kwargs):
        self.calls.append({"prompt": prompt, "n_images": len(images), "keep_vision": keep_vision})
        return prompt

    def encode_from_tokens_scheduled(self, tokens):
        return [[tokens, {}]]


class ExplodingVae:
    def encode(self, image):
        raise AssertionError("vae.encode should not be called when a latent is supplied")


class CountingVae:
    def __init__(self):
        self.calls = []

    def encode(self, image):
        self.calls.append(image.shape)
        return torch.full((1, 64, image.shape[2] // 16, image.shape[3] // 16), 7.0)


def run(images=None, latents=None, vae=None, resolution=1024):
    clip = FakeClip()
    out = TextEncodeQwenImage21Latent.execute(
        clip, "p", "n", resolution=resolution, images=images, latents=latents, vae=vae
    )
    return clip, out


def ref_latents(cond):
    return cond[0][1]["reference_latents"]


def test_external_latent_skips_vae():
    image = torch.rand(1, 512, 768, 3)
    latent = {"samples": torch.randn(1, 64, 32, 48)}
    clip, (positive, negative, empty) = run(
        images={"image_1": image}, latents={"latent_1": latent}, vae=ExplodingVae()
    )
    assert torch.equal(ref_latents(positive)[0], latent["samples"])
    assert torch.equal(ref_latents(negative)[0], latent["samples"])
    # empty target latent keeps the first image's resized size, 64 channels at /16
    # 512x768 at resolution 1024 -> 832x1248 -> /16
    assert empty["samples"].shape == (1, 64, 52, 78)
    assert all(c["keep_vision"] is False for c in clip.calls)


def test_latent_paired_by_index_vae_fills_gaps():
    image1 = torch.rand(1, 512, 512, 3)
    image3 = torch.rand(1, 512, 512, 3)
    latent3 = {"samples": torch.randn(1, 64, 32, 32)}
    vae = CountingVae()
    _, (positive, _, _) = run(
        images={"image_1": image1, "image_3": image3},
        latents={"latent_3": latent3},
        vae=vae,
    )
    refs = ref_latents(positive)
    assert len(refs) == 2
    assert torch.equal(refs[1], latent3["samples"])
    assert torch.all(refs[0] == 7.0)  # image_1 fell back to the vae
    assert len(vae.calls) == 1


def test_no_latents_no_vae_keeps_vision_tokens():
    image = torch.rand(1, 512, 512, 3)
    clip, (positive, _, _) = run(images={"image_1": image})
    assert "reference_latents" not in positive[0][1]
    assert all(c["keep_vision"] is True for c in clip.calls)


def test_vae_only_matches_core_behavior():
    image = torch.rand(1, 512, 512, 3)
    vae = CountingVae()
    _, (positive, _, _) = run(images={"image_1": image}, vae=vae)
    assert torch.all(ref_latents(positive)[0] == 7.0)


def test_batch_latent_collapsed_to_one():
    image = torch.rand(1, 512, 512, 3)
    latent = {"samples": torch.randn(2, 64, 32, 32)}
    _, (positive, _, _) = run(images={"image_1": image}, latents={"latent_1": latent})
    assert ref_latents(positive)[0].shape[0] == 1


def test_alpha_image_composited_for_vision_tower():
    image = torch.rand(1, 512, 512, 4)
    latent = {"samples": torch.randn(1, 64, 32, 32)}
    clip, _ = run(images={"image_1": image}, latents={"latent_1": latent}, vae=ExplodingVae())
    assert clip.calls[0]["n_images"] == 1
