import importlib.util
from pathlib import Path

import pytest
import torch

_IMPL = Path(__file__).resolve().parent.parent / "qwen_nodes.py"
_spec = importlib.util.spec_from_file_location("comfy_bb_qwen_nodes", _IMPL)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)

QwenImage21Sigmas = _impl.QwenImage21Sigmas


def run(width=1024, height=1024, steps=40, shift_terminal=0.02):
    return QwenImage21Sigmas.execute(width, height, steps, shift_terminal)[0]


# reference values computed against the diffusers formula:
# linspace(1, 1/N, N) -> exponential mu shift (0.5 @ 256 tokens, 0.9 @ 8192) -> stretch to 0.02 -> append 0
REF_1024 = {
    0: 1.000000, 1: 0.986964, 20: 0.656666, 38: 0.067881, 39: 0.020000, 40: 0.000000,
}
REF_768_1024 = {0: 1.000000, 1: 0.986316, 38: 0.065691, 39: 0.020000, 40: 0.000000}


def test_matches_diffusers_reference_1024():
    sigmas = run()
    assert sigmas.shape == (41,)
    for i, expected in REF_1024.items():
        assert sigmas[i].item() == pytest.approx(expected, abs=1e-6)


def test_matches_diffusers_reference_768_1024():
    sigmas = run(width=768, height=1024)
    assert sigmas.shape == (41,)
    for i, expected in REF_768_1024.items():
        assert sigmas[i].item() == pytest.approx(expected, abs=1e-6)


def test_larger_resolution_shifts_schedule_up():
    low = run(width=768, height=1024, shift_terminal=0.0)
    high = run(width=2048, height=2048, shift_terminal=0.0)
    # both start fully noised, but the big image spends more sigma range in high-noise steps
    assert low[0] == high[0] == 1.0
    assert torch.all(high[1:-1] > low[1:-1])


def test_shift_terminal_zero_disables_stretch():
    stretched = run(shift_terminal=0.02)
    plain = run(shift_terminal=0.0)
    assert plain[-1].item() == 0.0
    # without the stretch the smallest sigma keeps the raw shifted value
    mu = 4096 * (0.9 - 0.5) / (8192 - 256) + (0.5 - (0.9 - 0.5) / (8192 - 256) * 256)
    e_mu = pow(2.718281828459045, mu)
    raw_last = e_mu / (e_mu + (1.0 / 0.025 - 1.0))
    assert plain[-2].item() == pytest.approx(raw_last, abs=1e-6)
    assert plain[-2].item() == pytest.approx(0.048801, abs=1e-4)
    # stretching preserves the top of the schedule and presses the tail down onto 0.02
    assert stretched[0] == plain[0] == 1.0
    assert stretched[-2].item() == pytest.approx(0.02, abs=1e-6)
    assert stretched[-3].item() < plain[-3].item()


def test_step_count_matches_steps():
    sigmas = run(steps=8)
    assert sigmas.shape == (9,)
    assert sigmas[-1].item() == 0.0
    # 8-step grid is coarser: each step covers more sigma, so the same index sits lower
    assert run(steps=8)[2] < run(steps=40)[2]
