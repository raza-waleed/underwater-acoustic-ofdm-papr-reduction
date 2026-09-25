"""Receiver side nonlinear distortion removal: frequentative decision feedback
(FFB), thesis section 4.2 and Algorithm 1.

Received subcarrier k (thesis eq. 4-18, with the Bussgang gain kept explicit):

    Y_k = H_k (alpha X_k + D_k) + W_k

D_k is the amplifier's distortion, a deterministic function of the whole OFDM
symbol. The receiver equalises with the known channel, decides, rebuilds the
transmitted time signal from its decisions, runs it through its model of the
transmitter (the learned network or the exact equations), and subtracts the
distortion that model predicts before deciding again.
"""
import numpy as np

from .ofdm import slice_symbols, to_freq, to_time


def ffb_detect(Y, H, order, model, alpha, oversampling, iterations=3):
    """Return the symbol decisions after 0, 1, ..., iterations feedback passes.

    Pass 0 is the plain zero forcing receiver without distortion removal.
    """
    n = Y.shape[-1]
    Z = Y / H
    D = np.zeros_like(Z)
    decisions = []
    for it in range(iterations + 1):
        X_hat = slice_symbols((Z - D) / alpha, order)
        decisions.append(X_hat)
        if it == iterations:
            break
        x_hat = to_time(X_hat, oversampling)
        D = to_freq(model(x_hat) - alpha * x_hat, n)
    return decisions


def transmit(X, oversampling, transmitter):
    """Time signal after the nonlinear transmitter and its in-band spectrum."""
    y = transmitter(to_time(X, oversampling))
    return y, to_freq(y, X.shape[-1])


def add_noise(rng, Y_clean, H, snr_db):
    """Channel plus white noise at an average received SNR per subcarrier,
    measured on the transmitted in-band power (like MATLAB's awgn 'measured')."""
    es = np.mean(np.abs(Y_clean) ** 2)
    n0 = es / 10.0 ** (snr_db / 10.0)
    noise = np.sqrt(n0 / 2.0) * (rng.standard_normal(Y_clean.shape) + 1j * rng.standard_normal(Y_clean.shape))
    return H * Y_clean + noise
