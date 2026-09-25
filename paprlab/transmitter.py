"""Transmitter side PAPR reduction studied in my thesis and papers.

Distortion based (thesis chapter 3):
    repeated clipping and filtering (RCF) and rooting companding (RCT).
Distortionless, with side information (my SLM and PTS papers):
    selected mapping (SLM) and partial transmit sequences (PTS).
"""
import itertools

import numpy as np

from .ofdm import in_band, to_time
from .papr import papr_db


# ---------------------------------------------------------------- clipping

def clip(x, ratio, level=None):
    """Hard amplitude limiter that keeps the phase (thesis eq. 3-9).

    ratio is the thesis clipping ratio CR: the clipping power over the mean
    power of each symbol, so the amplitude limit is sqrt(CR * E|x|^2). Pass
    level instead to clip at a fixed amplitude.
    """
    mag = np.abs(x)
    if level is None:
        level = np.sqrt(ratio * np.mean(mag ** 2, axis=-1, keepdims=True))
    scale = np.where(mag > level, level / np.maximum(mag, 1e-300), 1.0)
    return x * scale


def band_limit(x, n_subcarriers):
    """Frequency domain filter: zero the out of band bins, keep the rest."""
    F = np.fft.fft(x, axis=-1)
    F[..., ~in_band(n_subcarriers, x.shape[-1])] = 0.0
    return np.fft.ifft(F, axis=-1)


def repeated_clipping_filtering(x, n_subcarriers, ratio, passes, filtering=True):
    """RCF as in my thesis code: each pass clips at sqrt(CR * mean power of the
    current symbol) and then removes out of band regrowth.

    Returns the final signal and the PAPR of every symbol after each pass,
    shape (passes, symbols).
    """
    history = []
    for _ in range(passes):
        x = clip(x, ratio)
        if filtering:
            x = band_limit(x, n_subcarriers)
        history.append(papr_db(x))
    return x, np.array(history)


# ---------------------------------------------------------------- companding

def root_compand(x, exponent):
    """Rooting companding (RCT, thesis eq. 3-11): |x|^R with the phase kept.
    R = 0.5 is square root companding; the thesis explored 0.1 <= R <= 0.9."""
    return np.abs(x) ** exponent * np.exp(1j * np.angle(x))


def root_expand(y, exponent):
    """Receiver side inverse of root_compand."""
    return np.abs(y) ** (1.0 / exponent) * np.exp(1j * np.angle(y))


# ---------------------------------------------------------------- SLM

def slm_phase_sequences(rng, candidates, n_subcarriers):
    """U phase sequences from {1, j, -1, -j}; the first leaves the data unchanged."""
    phases = np.exp(0.5j * np.pi * rng.integers(0, 4, size=(candidates, n_subcarriers)))
    phases[0] = 1.0
    return phases


def _chunk(candidates, size, budget=4_000_000):
    """Symbols per batch so a batch of candidate signals stays near budget samples."""
    return max(1, int(budget // (candidates * size)))


def selected_mapping(X, phases, oversampling=1):
    """Keep, for each symbol, the rotated copy with the lowest PAPR.

    Returns (time signals, chosen candidate index, PAPR in dB). The index is the
    side information: log2(U) bits per OFDM symbol.
    """
    X = np.atleast_2d(X)
    chunk = _chunk(phases.shape[0], int(round(X.shape[-1] * oversampling)))
    signals, chosen, values = [], [], []
    for start in range(0, X.shape[0], chunk):
        block = X[start:start + chunk]
        cands = to_time(block[:, None, :] * phases[None, :, :], oversampling)
        p = papr_db(cands)
        idx = np.argmin(p, axis=1)
        rows = np.arange(block.shape[0])
        signals.append(cands[rows, idx])
        chosen.append(idx)
        values.append(p[rows, idx])
    return np.concatenate(signals), np.concatenate(chosen), np.concatenate(values)


# ---------------------------------------------------------------- PTS

def pts_phase_combinations(subblocks, phase_set):
    """Every weighting vector with the first factor fixed to 1 (W^(V-1) of them)."""
    rest = np.array(list(itertools.product(phase_set, repeat=subblocks - 1)), dtype=complex)
    if rest.size == 0:
        rest = np.zeros((1, 0), dtype=complex)
    return np.concatenate([np.ones((rest.shape[0], 1), dtype=complex), rest], axis=1)


def partial_transmit_sequences(X, subblocks, phase_set=(1, -1), oversampling=1):
    """PTS with adjacent partitioning and an exhaustive phase search.

    The N subcarriers are split into V adjacent blocks, each block gets its own
    IFFT, and the transmitter keeps the weighted sum with the lowest PAPR.
    Returns (time signals, chosen combination index, PAPR in dB); the side
    information is (V - 1) log2(W) bits per OFDM symbol.
    """
    X = np.atleast_2d(X)
    n = X.shape[-1]
    if n % subblocks:
        raise ValueError("subcarriers must divide evenly into subblocks")
    width = n // subblocks
    weights = pts_phase_combinations(subblocks, phase_set)
    chunk = _chunk(weights.shape[0] + subblocks, int(round(n * oversampling)))
    signals, chosen, values = [], [], []
    for start in range(0, X.shape[0], chunk):
        block = X[start:start + chunk]
        parts = np.zeros((block.shape[0], subblocks, n), dtype=complex)
        for v in range(subblocks):
            parts[:, v, v * width:(v + 1) * width] = block[:, v * width:(v + 1) * width]
        partial = to_time(parts, oversampling)                  # (symbols, V, size)
        cands = np.einsum("cv,svt->sct", weights, partial)       # (symbols, C, size)
        p = papr_db(cands)
        idx = np.argmin(p, axis=1)
        rows = np.arange(block.shape[0])
        signals.append(cands[rows, idx])
        chosen.append(idx)
        values.append(p[rows, idx])
    return np.concatenate(signals), np.concatenate(chosen), np.concatenate(values)
