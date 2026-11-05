# RES4LYF RK samplers + beta57 scheduler — design

## Goal

Use RES4LYF's core samplers and scheduler without installing the full
extension. Port the math into comfy-bb-nodes and register it the same way
RES4LYF does: sampler functions injected into `KSampler.SAMPLERS` +
`comfy.k_diffusion.sampling`, scheduler into `SCHEDULER_HANDLERS`.

Scope: ODE only — no SDE/eta, no guides, no implicit methods, no c2/c3
overrides. Only deps: `math` + `torch`.

## What gets registered

- Samplers: `res_2m`, `res_3m`, `res_2s`, `res_3s`, `dpmpp_2m`, `deis_2m`
  (inserted after `uni_pc_bh2` in `KSampler.SAMPLERS`, idempotent).
- Scheduler: `beta57` = `comfy.samplers.beta_scheduler(alpha=0.5, beta=0.7)`.

Use with the stock KSampler node (e.g. res_2m + beta57).

## Structure

- `bb_rk_samplers.py` — pure math, imports only `math` (tensor ops need no
  torch import). `sample(model, x, sigmas, method, ...)` drives all methods
  from a `_METHODS` table `(family, order, multistep)`.
- `sampling_nodes.py` — sampler function wrappers + `register_samplers()`.
- `__init__.py` calls `register_samplers()` once at pack load.

## Math (ported from RES4LYF beta/)

Contract: the KSAMPLER `model(x, sigma)` callable returns denoised x0
(comfy/samplers.py "#Returns denoised"). Anchored epsilons, all relative to
the step-base x: exponential family `k = d − x`, linear family (deis)
`k = (x − d) / sigma`. `h = −log(sigma_next/sigma)`, `h = inf` on the final
step.

- Phi products use `expm1` closed forms (`_hphi1`/`_hphi2`) instead of
  RES4LYF's mpmath: cancellation-free at small h, exact at `h = inf` (where
  RES4LYF's mpmath path produces NaN), no mpmath dependency.
- One-step tableaus are the RES4LYF `res_2s`/`res_3s`/`dpmpp_2s`/`dpmpp_3s`
  cases: res stages advance through the tableau, dpmpp stages evaluate at the
  step base with default nodes c2=0.5, c3=1.0.
- Multistep (`*_2m`/`*_3m`) = the corresponding `*s` tableau with negative
  c-nodes `c2 = −h_prev1/h`, `c3 = −h_prev2/h` where `h_prev2` spans two
  intervals (`−log(sigma_i/sigma_{i−2})`). Active when `step >= order` and
  `h < 1.0`; otherwise the one-step method runs (warmup / fallback). One model
  call per multistep step; history is a ring of fresh denoiseds kept across
  fallback steps.
- `deis_2m`: DEIS "rhoab" second-order coefficients computed per step from
  the sigma schedule (linear sigma space, h = sigma_next − sigma, no h gate);
  warmup steps run `ralston_2s` in linear space, matching RES4LYF's default
  initial sampler.

Coefficient-level fidelity was verified by running RES4LYF's own
`get_rk_methods_beta` on shared schedules and comparing `h*A`/`h*B` rows and
the deis coefficients — all match.

## Intentional deviations

- Final step uses exact h→∞ tableau limits; RES4LYF feeds h=inf to mpmath and
  tolerates NaN.
- `h >= 1.0` fallback always runs the one-step method (RES4LYF switches to
  ddim when sigma < 0.1 — micro-optimization, omitted).
- dpmpp_3s/dpmpp_3m and the multistep methods measure 2nd-order global
  accuracy on analytic tests (their b-rows match RES4LYF exactly); only
  res_3s is cleanly 3rd order. This mirrors upstream, it is not a port bug.

## Tests

`tests/test_bb_rk_samplers.py`: constant-denoiser exactness per method,
final-step-to-zero exactness, convergence-order ratios (cubic true solution —
dpmpp_2s integrates quadratics exactly), total model-call counts per method,
callback contract, registration idempotency, beta57 schedule equality with
`beta_scheduler(alpha=0.5, beta=0.7)`.
