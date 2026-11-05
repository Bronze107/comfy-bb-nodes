# Exponential Runge-Kutta samplers ported from RES4LYF (ClownsharkBatwing/RES4LYF), ODE only.
# `model` follows the comfy KSAMPLER contract: model(x, sigma, **extra_args) returns denoised.

import math


C2 = 0.5
C3 = 1.0  # RES4LYF default stage nodes for one-step / warmup steps


def _hphi1(h, c):
    # h * phi_1(-h * c) = (1 - exp(-h * c)) / c, with the exact h -> inf limit
    if c == 0:
        return 0.0
    if math.isinf(h):
        return 1.0 / c
    return -math.expm1(-h * c) / c


def _hphi2(h, c):
    # h * phi_2(-h * c) = (h * c - 1 + exp(-h * c)) / (h * c * c), with the exact h -> inf limit
    if c == 0 or h == 0:
        return 0.0
    if math.isinf(h):
        return 1.0 / c
    u = h * c
    return (u + math.expm1(-u)) / (c * u)


def _gamma(c2, c3):
    return (3 * c3 ** 3 - 2 * c3) / (c2 * (2 - 3 * c2))


def _b_row(family, order, h, c2, c3=None):
    # final-row products h * b[j], shared by the one-step and multistep forms
    p1 = _hphi1(h, 1.0)
    p2 = _hphi2(h, 1.0)
    if order == 2:
        hb2 = p2 / c2 if family == "res" else p1 / (2 * c2)
        return p1 - hb2, hb2
    g = _gamma(c2, c3)
    if family == "res":
        hb3 = p2 / (g * c2 + c3)
        hb2 = g * hb3
    else:
        hb3 = p2 / c3
        hb2 = 0.0
    return p1 - hb2 - hb3, hb2, hb3


def _a32(family, h, c2, c3):
    # stage-3 product on e2: h * a[2][1]
    p23 = _hphi2(h, c3)
    if family == "res":
        return _gamma(c2, c3) * c2 * _hphi2(h, c2) + (c3 * c3 / c2) * p23
    return (c3 * c3 / c2) * p23


def _deis_coeffs(t_cur, t_next, t_prev):
    # DEIS "rhoab" second-order coefficients, adapted from RES4LYF beta/deis_coefficients.py
    coeff_cur = ((t_next - t_prev) ** 2 - (t_cur - t_prev) ** 2) / (2 * (t_cur - t_prev))
    coeff_prev = (t_next - t_cur) ** 2 / (2 * (t_prev - t_cur))
    return coeff_cur, coeff_prev


def _onestep_exp(model, x, sigma, h, denoised, family, order, extra_args):
    e1 = denoised - x
    # dpmpp stages evaluate at the step base x; res stages advance through the tableau
    x2 = x if family == "dpmpp" else x + (C2 * _hphi1(h, C2)) * e1
    d2 = model(x2, sigma * math.exp(-h * C2), **extra_args)
    e2 = d2 - x
    if order == 2:
        hb1, hb2 = _b_row(family, 2, h, C2)
        return x + hb1 * e1 + hb2 * e2
    a32 = _a32(family, h, C2, C3)
    x3 = x + (C3 * _hphi1(h, C3) - a32) * e1 + a32 * e2
    d3 = model(x3, sigma * math.exp(-h * C3), **extra_args)
    e3 = d3 - x
    hb1, hb2, hb3 = _b_row(family, 3, h, C2, C3)
    return x + hb1 * e1 + hb2 * e2 + hb3 * e3


def _ralston_2s(model, x, sigma, h, denoised, extra_args):
    # linear-space warmup step for deis, matching RES4LYF's multistep initial sampler
    s = float(sigma)
    k1 = (x - denoised) / s
    x2 = x + (2.0 / 3.0) * h * k1
    d2 = model(x2, sigma + h * (2.0 / 3.0), **extra_args)
    k2 = (x - d2) / s
    return x + h * (0.25 * k1 + 0.75 * k2)


_METHODS = {
    # name: (family, order, multistep)
    "res_2s":    ("res",   2, False),
    "res_3s":    ("res",   3, False),
    "res_2m":    ("res",   2, True),
    "res_3m":    ("res",   3, True),
    "dpmpp_2s":  ("dpmpp", 2, False),
    "dpmpp_3s":  ("dpmpp", 3, False),
    "dpmpp_2m":  ("dpmpp", 2, True),
    "dpmpp_3m":  ("dpmpp", 3, True),
    "deis_2m":   ("deis",  2, True),
}


def sample(model, x, sigmas, method, extra_args=None, callback=None, disable=None):
    family, order, multistep = _METHODS[method]
    extra_args = {} if extra_args is None else extra_args

    hist = []  # previous fresh denoiseds, most recent last
    h_prev1 = h_prev2 = 0.0

    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        s = float(sigma)
        s_next = float(sigma_next)

        denoised = model(x, sigma, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_hat': sigma, 'denoised': denoised})

        if family == "deis":
            h = s_next - s
            if i >= order + 1:
                c1, c0 = _deis_coeffs(s, s_next, float(sigmas[i - 1]))
                x = x + c1 * (x - denoised) / s + c0 * (x - hist[-1]) / s
            else:
                x = _ralston_2s(model, x, sigma, h, denoised, extra_args)
        else:
            h = -math.log(s_next / s) if s_next > 0.0 else math.inf
            if multistep and i >= order and h < 1.0:
                e1 = denoised - x
                c2 = -h_prev1 / h
                if order == 2:
                    hb1, hb2 = _b_row(family, 2, h, c2)
                    x = x + hb1 * e1 + hb2 * (hist[-1] - x)
                else:
                    c3 = -h_prev2 / h
                    hb1, hb2, hb3 = _b_row(family, 3, h, c2, c3)
                    x = x + hb1 * e1 + hb2 * (hist[-1] - x) + hb3 * (hist[-2] - x)
            else:
                x = _onestep_exp(model, x, sigma, h, denoised, family, order, extra_args)

        hist.append(denoised)
        hist = hist[-(order - 1):]
        # RES4LYF h_prev2 spans two intervals: -log(sigmas[step] / sigmas[step - 2])
        h_prev1, h_prev2 = h, h_prev1 + h

    return x
