"""PAPR reduction and nonlinear distortion removal for underwater acoustic OFDM.

Python implementation of the methods from my M.Eng. thesis (Harbin Engineering
University, 2021): repeated clipping and filtering, rooting companding, selected
mapping, partial transmit sequences, and a receiver that learns the power
amplifier with a small neural network and removes its distortion by frequentative
decision feedback (FFB).
"""

__version__ = "1.0.0"
