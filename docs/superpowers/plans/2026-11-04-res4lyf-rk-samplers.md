# Plan: RES4LYF RK samplers + beta57 scheduler

Design: `docs/superpowers/specs/2026-11-04-res4lyf-rk-samplers-design.md`

1. `bb_rk_samplers.py` — phi helpers (`_hphi1`/`_hphi2` with exact h→∞
   limits), b-row builders for res/dpmpp (order 2/3), stage-3 product
   `_a32`, DEIS rhoab coefficients, `ralston_2s` linear warmup, and the
   shared `sample()` loop (multistep negative c-nodes, ring history,
   warmup/fallback, per-step callback).
2. `tests/test_bb_rk_samplers.py` — exactness, convergence, NFE, callback,
   registration tests. Coefficient-level comparison against RES4LYF's own
   `get_rk_methods_beta` done during development via a throwaway script
   (all tableaus match); not kept as a test because it needs the RES4LYF
   checkout on disk.
3. `sampling_nodes.py` — `register_samplers()` injecting 6 sampler names
   after `uni_pc_bh2` in `KSampler.SAMPLERS` + `sample_<name>` attrs on
   `comfy.k_diffusion.sampling`, and `beta57` into `SCHEDULER_HANDLERS`.
4. `__init__.py` — import + single `register_samplers()` call.

Verification: `python_embeded\python.exe -m pytest custom_nodes/comfy-bb-nodes/tests -q`
from the ComfyUI root (146 passed), then a real generation with KSampler
set to res_2m + beta57.
