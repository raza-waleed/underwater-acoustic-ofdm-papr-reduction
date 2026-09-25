"""OFDM signal model used throughout the thesis: Gray QAM mapping and an IFFT
with oversampling by zero padding in the middle of the spectrum (thesis Fig. 3.6).

Conventions
-----------
Symbols are arranged as arrays of shape (symbols, N). Subcarrier k < N/2 sits at
baseband frequency +k * df and subcarrier k >= N/2 at (k - N) * df, the same order
MATLAB's fft uses, so the thesis layout [X(1:N/2); zeros; X(N/2+1:N)] maps across
directly. Time signals are scaled to unit average power when E|X|^2 = 1, so a
PAPR in dB is simply 10 log10 of the peak sample power.
"""
import numpy as np


def bits_per_symbol(order):
    k = int(round(np.log2(order)))
    m = int(round(np.sqrt(order)))
    if 2 ** k != order or m * m != order or order < 4:
        raise ValueError("order must be a square QAM size: 4 (QPSK), 16, 64, ...")
    return k


def _gray_to_binary(g):
    b = g.copy()
    shift = g >> 1
    while np.any(shift):
        b ^= shift
        shift >>= 1
    return b


def _bits_to_int(bits):
    weights = 1 << np.arange(bits.shape[-1] - 1, -1, -1)
    return (bits * weights).sum(axis=-1)


def _int_to_bits(values, width):
    shifts = np.arange(width - 1, -1, -1)
    return (values[..., None] >> shifts) & 1


def _norm(order):
    return np.sqrt(2.0 * (order - 1) / 3.0)


def modulate(bits, order):
    """Gray-coded square QAM with unit average power. bits: (..., n * log2(order))."""
    k = bits_per_symbol(order)
    m = int(round(np.sqrt(order)))
    half = k // 2
    b = np.asarray(bits, dtype=np.int64).reshape(*np.shape(bits)[:-1], -1, k)
    i_level = _gray_to_binary(_bits_to_int(b[..., :half]))
    q_level = _gray_to_binary(_bits_to_int(b[..., half:]))
    return ((2 * i_level - (m - 1)) + 1j * (2 * q_level - (m - 1))) / _norm(order)


def demodulate(symbols, order):
    """Hard decision back to bits (nearest constellation point, per axis)."""
    k = bits_per_symbol(order)
    m = int(round(np.sqrt(order)))
    half = k // 2
    s = np.asarray(symbols) * _norm(order)
    i_level = np.clip(np.rint((s.real + (m - 1)) / 2), 0, m - 1).astype(np.int64)
    q_level = np.clip(np.rint((s.imag + (m - 1)) / 2), 0, m - 1).astype(np.int64)
    i_bits = _int_to_bits(i_level ^ (i_level >> 1), half)
    q_bits = _int_to_bits(q_level ^ (q_level >> 1), half)
    out = np.concatenate([i_bits, q_bits], axis=-1)
    return out.reshape(*out.shape[:-2], -1)


def slice_symbols(symbols, order):
    """Nearest constellation point for each received value."""
    return modulate(demodulate(symbols, order), order)


def random_symbols(rng, count, n_subcarriers, order):
    """Random data symbols and the bits that produced them."""
    bits = rng.integers(0, 2, size=(count, n_subcarriers * bits_per_symbol(order)))
    return modulate(bits, order), bits


def fft_size(n_subcarriers, oversampling):
    size = int(round(n_subcarriers * oversampling))
    if size < n_subcarriers:
        raise ValueError("oversampling must be at least 1")
    return size


def in_band(n_subcarriers, size):
    """Boolean mask of the FFT bins that carry subcarriers."""
    mask = np.zeros(size, dtype=bool)
    mask[: n_subcarriers // 2] = True
    mask[size - (n_subcarriers - n_subcarriers // 2):] = True
    return mask


def to_time(X, oversampling=1):
    """Oversampled IFFT. X: (symbols, N) -> x: (symbols, round(N * oversampling))."""
    X = np.atleast_2d(X)
    n = X.shape[-1]
    size = fft_size(n, oversampling)
    F = np.zeros(X.shape[:-1] + (size,), dtype=complex)
    F[..., : n // 2] = X[..., : n // 2]
    F[..., size - (n - n // 2):] = X[..., n // 2:]
    return np.fft.ifft(F, axis=-1) * (size / np.sqrt(n))


def to_freq(x, n_subcarriers):
    """FFT back to the N in-band subcarriers (inverse of to_time)."""
    size = x.shape[-1]
    F = np.fft.fft(x, axis=-1) * (np.sqrt(n_subcarriers) / size)
    return np.concatenate([F[..., : n_subcarriers // 2], F[..., size - (n_subcarriers - n_subcarriers // 2):]], axis=-1)


def out_of_band_power(x, n_subcarriers):
    """Fraction of a signal's power that falls outside the occupied band."""
    F = np.fft.fft(x, axis=-1)
    p = np.abs(F) ** 2
    mask = in_band(n_subcarriers, x.shape[-1])
    return p[..., ~mask].sum() / p.sum()


def subcarrier_offsets(n_subcarriers, spacing_hz):
    """Baseband frequency of each subcarrier, in the order to_time/to_freq use."""
    k = np.arange(n_subcarriers)
    return np.where(k < n_subcarriers // 2, k, k - n_subcarriers) * spacing_hz


def add_cyclic_prefix(x, length):
    return np.concatenate([x[..., x.shape[-1] - length:], x], axis=-1) if length else x
