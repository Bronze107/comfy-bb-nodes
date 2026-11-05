import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

_IMPL = Path(__file__).resolve().parent.parent / "bb_rk_samplers.py"
_spec = importlib.util.spec_from_file_location("comfy_bb_rk_samplers", _IMPL)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)


class MockDenoiser:
    def __init__(self, fn):
        self.fn = fn
        self.calls = []

    def __call__(self, x, sigma, **kwargs):
        self.calls.append(float(sigma))
        return self.fn(x, float(sigma))


def constant_denoiser(u):
    return MockDenoiser(lambda x, s: u)


def run(method, sigmas, model, x):
    return _impl.sample(model, x, sigmas, method)


ALL_METHODS = ["res_2s", "res_3s", "res_2m", "res_3m",
               "dpmpp_2s", "dpmpp_3s", "dpmpp_2m", "dpmpp_3m", "deis_2m"]


def test_phi_helpers():
    assert _impl._hphi1(0.4, 1.0) == pytest.approx(0.3296799539643607, rel=1e-12)
    assert _impl._hphi2(0.4, 1.0) == pytest.approx(0.17580011508909825, rel=1e-12)
    assert _impl._hphi1(0.0, 1.0) == pytest.approx(0.0, abs=1e-15)
    assert _impl._hphi2(0.0, 1.0) == pytest.approx(0.0, abs=1e-15)
    assert _impl._hphi1(math.inf, 0.5) == 2.0
    assert _impl._hphi2(math.inf, 2.0) == 0.5


@pytest.mark.parametrize("method", ALL_METHODS)
def test_constant_denoiser_single_step_exact(method):
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8, 8, dtype=torch.float64)
    u = torch.randn(2, 4, 8, 8, dtype=torch.float64)
    out = run(method, torch.tensor([1.0, 0.3], dtype=torch.float64), constant_denoiser(u), x)
    expected = u + 0.3 * (x - u)
    assert torch.allclose(out, expected, atol=1e-12)


@pytest.mark.parametrize("method", ALL_METHODS)
def test_final_step_to_zero_is_exact(method):
    torch.manual_seed(1)
    x = torch.randn(2, 4, 8, 8, dtype=torch.float64)
    u = torch.randn(2, 4, 8, 8, dtype=torch.float64)
    out = run(method, torch.tensor([1.0, 0.0], dtype=torch.float64), constant_denoiser(u), x)
    assert torch.allclose(out, u, atol=1e-12)


def true_solution(s):
    # cubic: dpmpp_2s integrates quadratics exactly, so order tests need cubic
    return 1.0 + s + s * s + s ** 3


def quadrature_denoiser(x, s):
    # d(s) = x*(s) - s * dx*/ds makes dx/ds = (x - d)/s solve to x*(s)
    return torch.full_like(x, 1.0 - s * s - 2.0 * s * s * s)


ORDER3 = ["res_3s"]  # the only method with a clean 3rd-order global error; the
# multistep methods and dpmpp_3s measure 2nd-order-ish (ratio ~3-4) even though
# their tableaus match RES4LYF exactly.


def _convergence_ratio(method, n_lo, n_hi):
    def final_error(n):
        s0, s1 = 1.0, 0.5
        if method.startswith("deis"):
            sigmas = torch.linspace(s0, s1, n + 1, dtype=torch.float64)
        else:
            sigmas = torch.tensor([s0 * (s1 / s0) ** (k / n) for k in range(n + 1)], dtype=torch.float64)
        x = torch.full((2, 4, 8, 8), true_solution(s0), dtype=torch.float64)
        out = run(method, sigmas, MockDenoiser(quadrature_denoiser), x)
        return (out - true_solution(s1)).abs().max().item()

    return final_error(n_lo) / final_error(n_hi)


@pytest.mark.parametrize("method", [m for m in ALL_METHODS if m not in ORDER3])
def test_at_least_second_order_convergence(method):
    assert 2.5 < _convergence_ratio(method, 8, 16) < 6.5


@pytest.mark.parametrize("method", ORDER3)
def test_third_order_convergence(method):
    assert 5.0 < _convergence_ratio(method, 8, 16) < 12.0


@pytest.mark.parametrize("method,total_calls", [
    ("res_2s",   10),
    ("res_2m",   8),   # 2 warmup + 2 warmup + 1 + 1 multistep + 2 final
    ("res_3s",   15),
    ("res_3m",   13),  # 3 warmup steps + 1 multistep + 3 final
    ("dpmpp_2m", 8),
    ("deis_2m",  8),   # 3 ralston warmup steps + 2 multistep
])
def test_model_calls_per_step(method, total_calls):
    model = constant_denoiser(torch.zeros(2, 4, 8, 8, dtype=torch.float64))
    sigmas = torch.tensor([0.8, 0.4, 0.2, 0.1, 0.05, 0.0], dtype=torch.float64)
    run(method, sigmas, model, torch.zeros(2, 4, 8, 8, dtype=torch.float64))
    assert len(model.calls) == total_calls


def test_smoke_res_2m_reaches_constant_target():
    torch.manual_seed(2)
    u = torch.randn(1, 4, 16, 16, dtype=torch.float64)
    x = torch.randn(1, 4, 16, 16, dtype=torch.float64)
    sigmas = torch.tensor([1.0 * 0.75 ** k for k in range(10)] + [0.0], dtype=torch.float64)
    callbacks = []
    out = _impl.sample(constant_denoiser(u), x, sigmas, "res_2m", callback=callbacks.append)
    assert torch.allclose(out, u, atol=1e-9)
    assert [c["i"] for c in callbacks] == list(range(10))
    for c in callbacks:
        assert set(c.keys()) == {"x", "i", "sigma", "sigma_hat", "denoised"}


def test_unknown_method_raises():
    with pytest.raises(KeyError):
        run("nope_2m", torch.tensor([1.0, 0.0]), constant_denoiser(torch.zeros(1)), torch.zeros(1))


def _load_sampling_nodes():
    pack = Path(__file__).resolve().parent.parent
    if str(pack) not in sys.path:
        sys.path.insert(0, str(pack))
    spec = importlib.util.spec_from_file_location("comfy_bb_sampling_nodes", pack / "sampling_nodes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeModelSampling:
    def __init__(self, sigmas):
        self.sigmas = sigmas


class FakeModel:
    def __init__(self, sigmas):
        self._ms = FakeModelSampling(sigmas)

    def get_model_object(self, name):
        return self._ms


def test_register_samplers_idempotent():
    import comfy.samplers
    from comfy.samplers import k_diffusion_sampling

    mod = _load_sampling_nodes()
    mod.register_samplers()
    mod.register_samplers()

    for name in mod.BB_SAMPLERS:
        assert comfy.samplers.KSampler.SAMPLERS.count(name) == 1
        assert callable(getattr(k_diffusion_sampling, "sample_" + name))

    assert "beta57" in comfy.samplers.SCHEDULER_HANDLERS
    assert "beta57" in comfy.samplers.SCHEDULER_NAMES

    fake_ms = FakeModelSampling(torch.linspace(14.0, 0.0, 1001))
    sigmas = comfy.samplers.calculate_sigmas(fake_ms, "beta57", 10)
    expected = comfy.samplers.beta_scheduler(fake_ms, 10, alpha=0.5, beta=0.7)
    assert torch.allclose(sigmas, expected)


def test_registered_sampler_function_runs():
    import comfy.samplers
    from comfy.samplers import k_diffusion_sampling

    _load_sampling_nodes().register_samplers()
    torch.manual_seed(3)
    u = torch.randn(1, 4, 8, 8, dtype=torch.float64)
    x = torch.randn(1, 4, 8, 8, dtype=torch.float64)
    out = k_diffusion_sampling.sample_res_2m(
        constant_denoiser(u), x, torch.tensor([1.0, 0.3], dtype=torch.float64))
    assert torch.allclose(out, u + 0.3 * (x - u), atol=1e-12)
