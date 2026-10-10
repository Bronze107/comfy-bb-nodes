import importlib.util
from pathlib import Path

import pytest

_IMPL = Path(__file__).resolve().parent.parent / "model_nodes.py"
_spec = importlib.util.spec_from_file_location("comfy_bb_model_nodes", _IMPL)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)

parse_lora_spec = _impl.parse_lora_spec
MultiLoraLoader = _impl.MultiLoraLoader


def test_parse_single():
    assert parse_lora_spec("a.safetensors:0.8") == [("a.safetensors", 0.8)]


def test_parse_multiple_with_spaces():
    assert parse_lora_spec("a.safetensors:0.8 | b.safetensors: 0.5 ") == [
        ("a.safetensors", 0.8),
        ("b.safetensors", 0.5),
    ]


def test_parse_default_strength():
    assert parse_lora_spec("a.safetensors") == [("a.safetensors", 1.0)]


def test_parse_windows_path():
    # rpartition on ':' keeps drive letters and folder colons in the name
    assert parse_lora_spec(r"C:\loras\my:thing.safetensors:0.7") == [(r"C:\loras\my:thing.safetensors", 0.7)]


def test_parse_skips_empty_entries():
    assert parse_lora_spec(" | a.safetensors:0.8 | ") == [("a.safetensors", 0.8)]


def test_parse_negative_strength():
    assert parse_lora_spec("a.safetensors:-0.5") == [("a.safetensors", -0.5)]


def test_parse_bad_strength_raises():
    with pytest.raises(ValueError):
        parse_lora_spec("a.safetensors:abc")


class FakePatcher:
    def __init__(self, tag):
        self.tag = tag

    def __repr__(self):
        return f"FakePatcher({self.tag})"


def make_fake_loader(calls):
    def fake_load_lora_for_models(model, clip, lora, strength_model, strength_clip, lora_metadata=None):
        calls.append((lora, strength_model, strength_clip))
        return FakePatcher(f"{model}+{lora}"), FakePatcher(f"{clip}+{lora}")

    return fake_load_lora_for_models


def test_execute_loads_each_lora(monkeypatch):
    calls = []
    monkeypatch.setattr(_impl.folder_paths, "get_full_path_or_raise", lambda folder, name: f"loras/{name}")
    monkeypatch.setattr(_impl.comfy.utils, "load_torch_file", lambda path, safe_load, return_metadata: (path, {"m": path}))
    monkeypatch.setattr(_impl.comfy.sd, "load_lora_for_models", make_fake_loader(calls))

    model = MultiLoraLoader.execute("M", "a.safetensors:0.8 | b.safetensors")
    assert [c[0] for c in calls] == ["loras/a.safetensors", "loras/b.safetensors"]
    assert [c[1] for c in calls] == [0.8, 1.0]
    # model-only loading: clip is None and gets strength 0
    assert [c[2] for c in calls] == [0, 0]
    assert repr(model[0]) == "FakePatcher(FakePatcher(M+loras/a.safetensors)+loras/b.safetensors)"


def test_execute_skips_zero_strength(monkeypatch):
    calls = []
    monkeypatch.setattr(_impl.folder_paths, "get_full_path_or_raise", lambda folder, name: pytest.fail("should not resolve"))
    monkeypatch.setattr(_impl.comfy.sd, "load_lora_for_models", make_fake_loader(calls))

    model = MultiLoraLoader.execute("M", "a.safetensors:0")
    assert calls == []
    assert model[0] == "M"
