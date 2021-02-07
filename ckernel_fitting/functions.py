import torch
import numpy as np
import math


def get_function_to_fit(config):
    # form lin_space
    x = np.linspace(config.min, config.max, config.no_samples)
    # select function
    function = {
        "Gaussian": _gaussian,
        "Constant": _constant,
        "Linear": _linear,
        "Sawtooth": _sawtooth,
        "Sinus": _sinus,
        "SinusChirp": _sinus_chirp,
        "Random": _random,
    }[config.function]
    # apply
    sampled_function = function(config, x)
    return sampled_function


def _gaussian(config, x):
    # params
    mean = 0
    sigma = 0.2
    # apply function
    f = (
        1
        / (sigma * math.sqrt(2.0 * math.pi))
        * np.exp(-1 / 2.0 * ((x - mean) / sigma) ** 2)
    )
    f = 1 / float(max(f)) * f
    # return
    return f


def _constant(config, x):
    # apply function
    f = np.ones_like(x)
    # f[:int(len(f)/2)] = -1.0
    # return
    return f


def _sawtooth(config, x):
    # apply function
    f = np.ones_like(x)
    f[::2] = 0.0
    # return
    return f


def _linear(config, x):
    # apply function
    f = np.copy(x)
    # f[:int(len(f)/2)] = -1.0
    # return
    return f


def _sinus(config, x):
    # apply function
    f = np.sin(x)
    # f[:int(len(f)/2)] = -1.0
    # return
    return f


def _sinus_chirp(config, x):
    # apply function
    f = np.sin(x ** 2)
    # f[:int(len(f)/2)] = -1.0
    # return
    return f


def _random(config, x):
    # apply function
    f = np.random.rand(*x.shape)
    # f[:int(len(f)/2)] = -1.0
    # return
    return f
