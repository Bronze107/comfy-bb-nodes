# Registers the BB samplers and the beta57 scheduler into ComfyUI's global
# KSampler dropdowns, the same way RES4LYF injects its own samplers.

import functools

import comfy.samplers
from comfy.samplers import KSampler, SCHEDULER_HANDLERS, SCHEDULER_NAMES, SchedulerHandler

import bb_rk_samplers


BB_SAMPLERS = ["res_2m", "res_3m", "res_2s", "res_3s", "dpmpp_2m", "deis_2m"]


def _sample_function(method):
    def sample(model, x, sigmas, extra_args=None, callback=None, disable=None):
        return bb_rk_samplers.sample(model, x, sigmas, method, extra_args=extra_args, callback=callback, disable=disable)
    return sample


def register_samplers():
    from comfy.samplers import k_diffusion_sampling

    for name in BB_SAMPLERS:
        if name in KSampler.SAMPLERS:
            continue
        # add after uni_pc_bh2 so custom samplers sit at the end of the list
        KSampler.SAMPLERS.insert(KSampler.SAMPLERS.index("uni_pc_bh2") + 1, name)
        setattr(k_diffusion_sampling, "sample_" + name, _sample_function(name))

    if "beta57" not in SCHEDULER_HANDLERS:
        SCHEDULER_HANDLERS["beta57"] = SchedulerHandler(
            handler=functools.partial(comfy.samplers.beta_scheduler, alpha=0.5, beta=0.7),
            use_ms=True,
        )
        SCHEDULER_NAMES.append("beta57")
