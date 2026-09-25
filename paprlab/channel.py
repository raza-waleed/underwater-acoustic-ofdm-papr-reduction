"""Underwater acoustic channel from BELLHOP multipath arrivals.

The arrivals parser follows the one I wrote and validated for my BELLHOP report
(github.com/raza-waleed/bellhop-acoustic-modeling-report); BELLHOP itself is
Michael B. Porter's Acoustics Toolbox and is not part of this repository.
"""
import numpy as np


def read_arrivals(path):
    """Parse a BELLHOP ASCII arrivals (.arr) file.

    Layout: frequency and the NSD, NRD, NR counts; the source depths, receiver
    depths and ranges; then per source depth a summary block of arrival counts,
    followed for each (receiver depth, range) pair by a count and that many lines
    of amplitude, phase (deg), delay (s), source angle, receiver angle, surface
    bounces and bottom bounces.
    """
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]
    freq, nsd, nrd, nr = lines[0].split()
    nsd, nrd, nr = int(nsd), int(nrd), int(nr)
    idx = 1
    sd = [float(lines[idx + i]) for i in range(nsd)]
    idx += nsd
    rd = [float(lines[idx + i]) for i in range(nrd)]
    idx += nrd
    ranges = [float(lines[idx + i]) for i in range(nr)]
    idx += nr
    arrivals = []
    for isd in range(nsd):
        idx += nrd * nr
        for ird in range(nrd):
            for irr in range(nr):
                count = int(lines[idx])
                idx += 1
                for _ in range(count):
                    v = lines[idx].split()
                    idx += 1
                    arrivals.append({
                        "sd": sd[isd], "rd": rd[ird], "r": ranges[irr],
                        "amp": float(v[0]), "phase_deg": float(v[1]), "delay_s": float(v[2]),
                        "src_angle_deg": float(v[3]), "rcv_angle_deg": float(v[4]),
                        "n_top": int(v[5]), "n_bot": int(v[6]),
                    })
    arrivals.sort(key=lambda a: a["delay_s"])
    return {"freq": float(freq), "sd": sd, "rd": rd, "r": ranges, "arrivals": arrivals}


def frequency_response(arrivals, freqs_hz):
    """H(f) = sum_i A_i exp(-j phi_i) exp(-j 2 pi f (tau_i - tau_0)).

    BELLHOP works with an exp(-i w t) time convention, so its arrival phases
    enter conjugated when written in the usual exp(+j w t) engineering form.
    Delays are taken relative to the first arrival (synchronised receiver).
    """
    f = np.asarray(freqs_hz, dtype=float)
    t0 = min(a["delay_s"] for a in arrivals)
    H = np.zeros(f.shape, dtype=complex)
    for a in arrivals:
        H += a["amp"] * np.exp(-1j * np.radians(a["phase_deg"])) * np.exp(-2j * np.pi * f * (a["delay_s"] - t0))
    return H


def normalized(H):
    """Scale a response to unit mean power so SNR is the average received SNR."""
    return H / np.sqrt(np.mean(np.abs(H) ** 2))


def energy_after(arrivals, window_s):
    """Fraction of the arrival energy that lands later than window_s after the first."""
    t0 = min(a["delay_s"] for a in arrivals)
    total = sum(a["amp"] ** 2 for a in arrivals)
    late = sum(a["amp"] ** 2 for a in arrivals if a["delay_s"] - t0 > window_s)
    return late / total


def rms_delay_spread(arrivals):
    t0 = min(a["delay_s"] for a in arrivals)
    w = np.array([a["amp"] ** 2 for a in arrivals])
    t = np.array([a["delay_s"] - t0 for a in arrivals])
    mean = np.sum(w * t) / w.sum()
    return float(np.sqrt(np.sum(w * t ** 2) / w.sum() - mean ** 2))
