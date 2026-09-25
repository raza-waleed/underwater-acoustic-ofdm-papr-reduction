"""The nonlinear transmitter of thesis chapter 4: an optional clipper followed by
a solid state power amplifier (SSPA, Rapp model) with no phase distortion."""
import numpy as np

from .transmitter import clip


def rapp(x, saturation, smoothness=2.0):
    """Rapp SSPA: |y| = |x| / (1 + (|x| / A_sat)^(2p))^(1 / 2p), phase unchanged.
    p = 2 matches the amplifier curve in my thesis (Fig. 4.6)."""
    a = np.abs(x)
    return x * (1.0 + (a / saturation) ** (2.0 * smoothness)) ** (-1.0 / (2.0 * smoothness))


def level_db(db, rms=1.0):
    """Amplitude that sits db decibels above an RMS amplitude."""
    return rms * 10.0 ** (db / 20.0)


def bussgang_gain(x, y):
    """alpha = E[y x*] / E|x|^2, the linear part of a memoryless nonlinearity
    driven by a Gaussian like input (Bussgang). y = alpha x + d, d uncorrelated with x."""
    return float(np.real(np.vdot(x, y) / np.vdot(x, x)))


class NonlinearTransmitter:
    """Clipper (fixed level, dB above the RMS input) followed by the Rapp SSPA.

    ibo_db is the amplifier's saturation level above the RMS input amplitude
    (input back-off); the thesis simulated 5, 7 and 10 dB.
    """

    def __init__(self, ibo_db=7.0, clip_db=None, smoothness=2.0, rms=1.0):
        self.ibo_db = ibo_db
        self.clip_db = clip_db
        self.smoothness = smoothness
        self.saturation = level_db(ibo_db, rms)
        self.clip_level = None if clip_db is None else level_db(clip_db, rms)

    def __call__(self, x):
        if self.clip_level is not None:
            x = clip(x, None, level=self.clip_level)
        return rapp(x, self.saturation, self.smoothness)

    def amplitude(self, a):
        """Output amplitude for an input amplitude (the AM/AM curve)."""
        return np.abs(self(np.asarray(a, dtype=complex)))
