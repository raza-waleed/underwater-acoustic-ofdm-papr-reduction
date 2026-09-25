"""PAPR statistics: per symbol PAPR, the CCDF, its classic closed forms, and what
a lower PAPR is worth to a battery powered transmitter."""
import numpy as np


def papr_db(x):
    """PAPR of each symbol (last axis) in dB: 10 log10(max|x|^2 / mean|x|^2)."""
    p = np.abs(x) ** 2
    return 10.0 * np.log10(p.max(axis=-1) / p.mean(axis=-1))


def ccdf(papr_values_db, grid_db):
    """Empirical Pr[PAPR > z] for each z in grid_db."""
    v = np.sort(np.ravel(papr_values_db))
    return 1.0 - np.searchsorted(v, grid_db, side="right") / v.size


def papr_at(papr_values_db, probability=1e-3):
    """PAPR level exceeded with the given probability (the usual CCDF readout)."""
    return float(np.quantile(np.ravel(papr_values_db), 1.0 - probability))


def ccdf_theory(grid_db, n_subcarriers, oversampling_factor=1.0):
    """Nyquist rate approximation 1 - (1 - exp(-z))^(a N) for Gaussian samples.

    a = 1 treats the N Nyquist samples as independent; a = 2.8 is the empirical
    correction van Nee and de Wild proposed for an oversampled signal.
    """
    z = 10.0 ** (np.asarray(grid_db) / 10.0)
    return 1.0 - (1.0 - np.exp(-z)) ** (oversampling_factor * n_subcarriers)


def slm_ccdf_theory(grid_db, n_subcarriers, candidates):
    """Selected mapping keeps the best of U independent candidates."""
    return ccdf_theory(grid_db, n_subcarriers) ** candidates


def class_b_efficiency(papr_db_value):
    """Average efficiency of an ideal class B amplifier backed off so the peak
    just reaches saturation: (pi / 4) / sqrt(PAPR)."""
    return (np.pi / 4.0) * 10.0 ** (-np.asarray(papr_db_value) / 20.0)


def class_a_efficiency(papr_db_value):
    """Ideal class A amplifier under the same rule: 0.5 / PAPR."""
    return 0.5 * 10.0 ** (-np.asarray(papr_db_value) / 10.0)
