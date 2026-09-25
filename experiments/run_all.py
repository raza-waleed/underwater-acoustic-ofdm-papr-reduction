"""Run every experiment in the report and write results/results.json.

    python experiments/run_all.py            full run (a few minutes)
    python experiments/run_all.py --quick    smaller sample sizes, for a smoke test

Every number quoted in the report, the site page and the lab comes from the
JSON this script writes; experiments/make_figures.py draws the figures from it
and build.py fills the report template from it.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import erfc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paprlab import __version__, amplifier as amp, channel as chn, neural, ofdm, papr, receiver as rx  # noqa: E402
from paprlab import transmitter as tx  # noqa: E402

GRID = np.round(np.arange(2.0, 14.0001, 0.05), 2)
# Chapter 4 link: 512 subcarriers in 6.25 kHz, 8192 point FFT at 100 kHz
UWA = {"subcarriers": 512, "bandwidth_hz": 6250.0, "fft_points": 8192, "fs_hz": 100000.0,
       "centre_hz": 11000.0, "sound_speed": 1500.0}


def r4(v):
    return float(np.round(v, 4))


def curve(values):
    """CCDF on the shared grid, rounded, trailing zeros trimmed for size."""
    c = papr.ccdf(values, GRID)
    return [float(f"{p:.5g}") for p in c]


def papr_random(rng, n, oversampling, count, order=4):
    chunk = max(1, int(4_000_000 // ofdm.fft_size(n, oversampling)))
    out = []
    for start in range(0, count, chunk):
        X, _ = ofdm.random_symbols(rng, min(chunk, count - start), n, order)
        out.append(papr.papr_db(ofdm.to_time(X, oversampling)))
    return np.concatenate(out)


def q_ber(margins, snr_db, es):
    """QPSK bit error rate in AWGN computed from each bit's noiseless decision
    margin: the mean of Q(margin / sigma). margins is (centres, weights), a fine
    histogram of the margins (bins of about 1e-4), so no Monte Carlo noise enters."""
    centres, weights = margins
    sigma = np.sqrt(es / 10.0 ** (snr_db / 10.0) / 2.0)
    return float(np.sum(weights * 0.5 * erfc(centres / (sigma * np.sqrt(2.0)))))


def margin_histogram(margins, bins=40000):
    counts, edges = np.histogram(margins, bins=bins)
    keep = counts > 0
    return 0.5 * (edges[:-1] + edges[1:])[keep], counts[keep] / margins.size


def snr_for_ber(raw_margins, es, target=1e-4):
    """SNR (dB per subcarrier) at which the BER reaches the target, or None if a floor stops it."""
    floor = float(np.mean(raw_margins <= 0))
    margins = margin_histogram(raw_margins)
    if floor >= target or q_ber(margins, 80.0, es) > target:
        return None
    lo, hi = -5.0, 80.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if q_ber(margins, mid, es) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def qpsk_margins(X_sent, X_rx):
    """Distance of each received component to the QPSK decision boundary, signed by the sent bit."""
    return np.concatenate([(np.sign(X_sent.real) * X_rx.real).ravel(), (np.sign(X_sent.imag) * X_rx.imag).ravel()])


def sdr_db(X, Xp):
    """In-band signal to distortion ratio after the best linear scaling."""
    a = np.vdot(X, Xp) / np.vdot(X, X)
    return float(10 * np.log10(np.mean(np.abs(a * X) ** 2) / np.mean(np.abs(Xp - a * X) ** 2)))


def interp_snr(snrs, bers, target=1e-4):
    """Log-linear interpolation of the SNR where a measured BER curve crosses target."""
    b = np.asarray(bers, dtype=float)
    for i in range(len(b) - 1):
        if b[i] >= target > b[i + 1]:
            if b[i + 1] > 0:
                j0, j1 = i, i + 1
            elif i > 0 and b[i - 1] > b[i] > 0:
                j0, j1 = i - 1, i          # no errors at the next point: extend the last slope
            else:
                return float(snrs[i + 1])
            f = (np.log10(b[j0]) - np.log10(target)) / (np.log10(b[j0]) - np.log10(b[j1]))
            return float(snrs[j0] + f * (snrs[j1] - snrs[j0]))
    return None


# =========================================================================== experiments

def exp_basics(rng, n_sym):
    res = {"sizes": [64, 128, 256, 512, 1024], "L1": {}, "L4": {}, "curves": {}}
    for n in res["sizes"]:
        for L in (1, 4):
            v = papr_random(rng, n, L, n_sym if n <= 256 else n_sym // 2)
            res[f"L{L}"][str(n)] = r4(papr.papr_at(v))
            if n in (64, 256, 1024):
                res["curves"][f"N{n}_L{L}"] = curve(v)
        res.setdefault("theory_L1", {})[str(n)] = r4(float(np.interp(-3.0, np.log10(papr.ccdf_theory(GRID, n)[::-1] + 1e-300), GRID[::-1])))
    for n in (64, 256, 1024):
        res["curves"][f"theory_N{n}_L1"] = [float(f"{p:.5g}") for p in papr.ccdf_theory(GRID, n)]
        res["curves"][f"theory_N{n}_L4"] = [float(f"{p:.5g}") for p in papr.ccdf_theory(GRID, n, 2.8)]
    return res


def exp_rcf_thesis(rng, n_sym):
    n, L, cr = 128, 2, 3.0
    X, _ = ofdm.random_symbols(rng, n_sym, n, 4)
    x = ofdm.to_time(X, L)
    p0 = papr.papr_db(x)
    y, hist = tx.repeated_clipping_filtering(x, n, cr, 4)
    yc, hist_c = tx.repeated_clipping_filtering(x, n, cr, 1, filtering=False)
    # spectra (averaged periodogram, dB relative to the in-band level)
    def spectrum(sig):
        P = np.mean(np.abs(np.fft.fft(sig, axis=1)) ** 2, axis=0)
        P = np.fft.fftshift(P)
        return (10 * np.log10(P / np.mean(P[np.fft.fftshift(ofdm.in_band(n, sig.shape[1]))]))).round(2).tolist()
    Xr = ofdm.to_freq(y, n)
    return {
        "config": {"subcarriers": n, "oversampling": L, "cr": cr, "cr_db": r4(10 * np.log10(cr)), "symbols": n_sym},
        "original": r4(papr.papr_at(p0)),
        "passes": [r4(papr.papr_at(h)) for h in hist],
        "passes_1e2": [r4(papr.papr_at(h, 1e-2)) for h in hist],
        "clip_only": r4(papr.papr_at(hist_c[0])),
        "oob_clip_only_db": r4(10 * np.log10(ofdm.out_of_band_power(yc, n))),
        "oob_rcf_db": None if ofdm.out_of_band_power(y, n) == 0 else r4(10 * np.log10(ofdm.out_of_band_power(y, n))),
        "sdr_db": r4(sdr_db(X, Xr)),
        "curves": {"original": curve(p0), **{f"pass{i + 1}": curve(h) for i, h in enumerate(hist)}, "clip_only": curve(hist_c[0])},
        "spectrum": {"original": spectrum(x[:2000]), "clip_only": spectrum(yc[:2000]), "rcf": spectrum(y[:2000])},
    }


def exp_rcf_sweep(rng, n_sym):
    n, rows = 128, []
    X, _ = ofdm.random_symbols(rng, n_sym, n, 4)
    reference = snr_for_ber(qpsk_margins(X, X), 1.0)
    for I in (1, 2, 3, 4):
        x = ofdm.to_time(X, I)
        for cr in (4.0, 3.0, 2.0, 1.75, 1.5):
            y, hist = tx.repeated_clipping_filtering(x, n, cr, 4, filtering=I > 1)
            Xr = ofdm.to_freq(y, n)
            es = float(np.mean(np.abs(Xr) ** 2))
            snr = snr_for_ber(qpsk_margins(X, Xr), es)
            rows.append({
                "oversampling": I, "cr": cr, "cr_db": r4(10 * np.log10(cr)),
                "papr_1e3": r4(papr.papr_at(hist[-1])), "sdr_db": r4(sdr_db(X, Xr)),
                "snr_1e4": None if snr is None else r4(snr),
                "penalty_db": None if snr is None else r4(snr - reference),
            })
    return {"symbols": n_sym, "reference_snr_1e4": r4(reference), "rows": rows}


def exp_rct(rng, n_sym, ber_sym):
    n = 128
    # 1. the thesis setting: Nyquist rate, PAPR measured with the 32 sample cyclic prefix
    X, _ = ofdm.random_symbols(rng, n_sym, n, 4)
    x1 = ofdm.add_cyclic_prefix(ofdm.to_time(X, 1), 32)
    thesis_setting = {"original": r4(papr.papr_at(papr.papr_db(x1))),
                      "companded": r4(papr.papr_at(papr.papr_db(tx.root_compand(x1, 0.5))))}
    # 2. accurate peaks at L = 4 for a range of exponents
    x4 = ofdm.to_time(X, 4)
    p0 = papr.papr_db(x4)
    by_r, curves = {}, {"original": curve(p0)}
    for R in (0.3, 0.5, 0.7, 0.9):
        pr = papr.papr_db(tx.root_compand(x4, R))
        by_r[str(R)] = r4(papr.papr_at(pr))
        curves[f"R{R}"] = curve(pr)
    # 3. BER in AWGN with the receiver expanding the samples (L = 1, as in the thesis)
    snrs = list(np.arange(0.0, 32.1, 1.0))
    ber = {}
    Xb, bits = ofdm.random_symbols(rng, ber_sym, n, 4)
    xb = ofdm.to_time(Xb, 1)
    for R in (None, 0.5, 0.7, 0.9):
        if R is None:
            s, c = xb, 1.0
        else:
            s = tx.root_compand(xb, R)
            c = np.sqrt(np.mean(np.abs(s) ** 2))
            s = s / c
        row = []
        for snr in snrs:
            n0 = 10.0 ** (-snr / 10.0)
            r = s + np.sqrt(n0 / 2) * (rng.standard_normal(s.shape) + 1j * rng.standard_normal(s.shape))
            z = r if R is None else tx.root_expand(c * r, R)
            row.append(float(np.mean(ofdm.demodulate(ofdm.to_freq(z, n), 4) != bits)))
        ber["none" if R is None else str(R)] = row
    snr_1e4 = {k: (None if interp_snr(snrs, v) is None else r4(interp_snr(snrs, v))) for k, v in ber.items()}
    return {"thesis_setting": thesis_setting, "L4": {"original": r4(papr.papr_at(p0)), **by_r}, "curves": curves,
            "ber": {"snr_db": [float(s) for s in snrs], "curves": ber, "snr_1e4": snr_1e4, "symbols": ber_sym}}


def exp_slm(rng, n_sym):
    n = 256
    X, _ = ofdm.random_symbols(rng, n_sym, n, 4)
    out = {"subcarriers": n, "symbols": n_sym, "L4": {}, "L1": {}, "theory_L1": {}, "curves": {}}
    for U in (1, 2, 4, 8, 16):
        phases = tx.slm_phase_sequences(rng, U, n)
        for L in (1, 4):
            _, _, p = tx.selected_mapping(X, phases, L)
            out[f"L{L}"][str(U)] = r4(papr.papr_at(p))
            if L == 4:
                out["curves"][f"U{U}"] = curve(p)
        th = papr.slm_ccdf_theory(GRID, n, U)
        out["theory_L1"][str(U)] = r4(float(np.interp(-3.0, np.log10(th[::-1] + 1e-300), GRID[::-1])))
        out["curves"][f"theory_U{U}"] = [float(f"{p:.5g}") for p in th]
    return out


def exp_pts(rng, n_sym):
    n = 256
    X, _ = ofdm.random_symbols(rng, n_sym, n, 4)
    out = {"subcarriers": n, "symbols": n_sym, "rows": [], "curves": {}}
    for V, W in ((2, 2), (4, 2), (8, 2), (4, 4)):
        phase_set = (1, -1) if W == 2 else (1, -1, 1j, -1j)
        _, _, p = tx.partial_transmit_sequences(X, V, phase_set, 4)
        out["rows"].append({"subblocks": V, "phases": W, "papr_1e3": r4(papr.papr_at(p)),
                            "combinations": W ** (V - 1), "side_bits": int((V - 1) * np.log2(W)), "ifft_per_symbol": V})
        out["curves"][f"V{V}_W{W}"] = curve(p)
    return out


def exp_compare(rng, n_sym):
    """All methods on one footing: N = 128, L = 4, QPSK."""
    n, L = 128, 4
    X, _ = ofdm.random_symbols(rng, n_sym, n, 4)
    x = ofdm.to_time(X, L)
    rows = []

    def add(name, sig, side_bits, iffts, note):
        pv = papr.papr_db(sig)
        Xr = ofdm.to_freq(sig, n)
        oob = ofdm.out_of_band_power(sig, n)
        rows.append({"method": name, "papr_1e3": r4(papr.papr_at(pv)), "sdr_db": None if note in ("distortionless", "reference") else r4(sdr_db(X, Xr)),
                     "oob_db": None if oob < 1e-12 else r4(10 * np.log10(oob)), "side_bits": side_bits, "ifft_per_symbol": iffts,
                     "class_b": r4(100 * papr.class_b_efficiency(papr.papr_at(pv))), "class_a": r4(100 * papr.class_a_efficiency(papr.papr_at(pv))),
                     "kind": note})

    add("Original OFDM", x, 0, 1, "reference")
    add("Clipping only (CR 3)", tx.clip(x, 3.0), 0, 1, "distortion")
    add("RCF (CR 3, 4 passes)", tx.repeated_clipping_filtering(x, n, 3.0, 4)[0], 0, 9, "distortion")  # 1 IFFT + an FFT/IFFT pair per pass
    add("Rooting companding (R 0.5)", tx.root_compand(x, 0.5), 0, 1, "distortion")
    phases = tx.slm_phase_sequences(rng, 16, n)
    add("SLM (U 16)", tx.selected_mapping(X, phases, L)[0], 4, 16, "distortionless")
    add("PTS (V 8, W 2)", tx.partial_transmit_sequences(X, 8, (1, -1), L)[0], 7, 8, "distortionless")
    base = rows[0]["class_b"]
    for r in rows:
        r["battery_factor"] = r4(r["class_b"] / base)
    return {"subcarriers": n, "oversampling": L, "symbols": n_sym, "rows": rows}


def exp_channel():
    d = chn.read_arrivals(str(ROOT / "channel" / "thesis_2km.arr"))
    arr = d["arrivals"]
    t0 = arr[0]["delay_s"]
    peak = max(a["amp"] for a in arr)
    spacing = UWA["bandwidth_hz"] / UWA["subcarriers"]
    f = UWA["centre_hz"] + ofdm.subcarrier_offsets(UWA["subcarriers"], spacing)
    H = chn.normalized(chn.frequency_response(arr, f))
    order = np.argsort(f)
    length = float(np.hypot(d["r"][0], d["rd"][0] - d["sd"][0]))
    z = np.linspace(d["sd"][0], d["rd"][0], 2001)
    c = np.interp(z, [0, 10, 20, 30, 40, 50, 100], [1482.0, 1484.6, 1489.6, 1492.8, 1496.8, 1498.4, 1498.4])
    straight = length * np.mean(1.0 / c)
    return {
        "geometry": {"depth_m": 100.0, "source_m": d["sd"][0], "receiver_m": d["rd"][0], "range_m": d["r"][0], "freq_hz": d["freq"],
                     "ssp": [[0, 1482.0], [10, 1484.6], [20, 1489.6], [30, 1492.8], [40, 1496.8], [50, 1498.4], [100, 1498.4]],
                     "bottom": {"speed": 1600.0, "density": 1.8, "attenuation_db_per_wavelength": 0.8}},
        "arrivals": [{"delay_ms": round(1e3 * (a["delay_s"] - t0), 6), "amp": float(f"{a['amp'] / peak:.7g}"), "phase_deg": round(a["phase_deg"] % 360, 4),
                      "top": a["n_top"], "bottom": a["n_bot"], "angle": r4(a["src_angle_deg"])} for a in arr],
        "first_arrival_s": r4(t0), "straight_line_s": r4(straight), "first_arrival_error_pct": r4(100 * (t0 - straight) / straight),
        "rms_delay_ms": r4(1e3 * chn.rms_delay_spread(arr)), "max_delay_ms": r4(1e3 * (arr[-1]["delay_s"] - t0)),
        "energy_after_ms": {str(w): r4(100 * chn.energy_after(arr, w / 1e3)) for w in (25, 50, 100)},
        "response_db": {"freq_hz": [float(v) for v in f[order][::4]], "mag_db": [r4(20 * np.log10(abs(h))) for h in H[order][::4]]},
        "fade_stats": {"min_db": r4(20 * np.log10(np.abs(H).min())), "max_db": r4(20 * np.log10(np.abs(H).max())),
                       "below_10db_pct": r4(100 * np.mean(np.abs(H) < 10 ** (-10 / 20)))},
        "H": H,
    }


def train_network(rng, ibo_db, seed):
    """Training data as in thesis section 4.3.2: 25,800 amplitude samples of real
    OFDM symbols through the transmitter, measured at 35 dB SNR, in vectors of 6."""
    T = amp.NonlinearTransmitter(ibo_db=ibo_db)
    X, _ = ofdm.random_symbols(rng, 13, UWA["subcarriers"], 4)
    x = ofdm.to_time(X, 4).ravel()[:25800]
    y = T(x)
    sigma = 10 ** (-35 / 20) * np.sqrt(np.mean(np.abs(y) ** 2) / 2)
    y_meas = y + sigma * (rng.standard_normal(y.size) + 1j * rng.standard_normal(y.size))
    net = neural.AmplifierNetwork(seed=seed).fit(np.abs(x).reshape(-1, 6), np.abs(y_meas).reshape(-1, 6), seed=seed)
    model = neural.NeuralAmplifierModel(net)
    a_grid = np.linspace(0, 4.0, 81)
    # memorylessness check: how much does output j move when input i != j moves?
    base = np.full((1, 6), 1.0)
    jac = np.zeros((6, 6))
    for i in range(6):
        up = base.copy(); up[0, i] += 1e-3
        jac[:, i] = (net.predict_vectors(up) - net.predict_vectors(base))[0] / 1e-3
    return T, net, model, {
        "ibo_db": ibo_db, **{k: v for k, v in net.history.items() if not k.endswith("_curve")},
        "train_curve": [float(f"{v:.4g}") for v in net.history["train_curve"]],
        "validation_curve": [float(f"{v:.4g}") for v in net.history["validation_curve"]],
        "alpha_exact": r4(amp.bussgang_gain(x, y)), "alpha_network": r4(amp.bussgang_gain(x, model(x))),
        "amam": {"input": [r4(v) for v in a_grid], "true": [r4(v) for v in T.amplitude(a_grid)],
                 "network": [r4(v) for v in net.predict_vectors(np.repeat(a_grid[:, None], 6, axis=1)).mean(axis=1)]},
        "train_amplitude_999": r4(np.quantile(np.abs(x), 0.999)),
        "max_error_below_3rms": r4(np.max(np.abs(T.amplitude(a_grid[a_grid <= 3]) - net.predict_amplitude(a_grid[a_grid <= 3])))),
        "cross_sensitivity": r4(np.max(np.abs(jac - np.diag(np.diag(jac)))) / np.max(np.abs(np.diag(jac)))),
        "weights": net.to_dict(),
    }


def ber_run(rng, order, transmitter, models, H, snrs, n_sym, iterations=3):
    """BER for a list of receivers at each SNR. models: {name: (model or None, alpha, passes)}."""
    X, bits = ofdm.random_symbols(rng, n_sym, UWA["subcarriers"], order)
    _, Ytx = rx.transmit(X, 4, transmitter)
    out = {name: [] for name in models}
    for snr in snrs:
        Y = rx.add_noise(rng, Ytx, H, snr)
        for name, (model, alpha, passes) in models.items():
            dec = rx.ffb_detect(Y, H, order, model if model is not None else transmitter, alpha, 4, iterations=passes)
            out[name].append(float(np.mean(ofdm.demodulate(dec[-1], order) != bits)))
    return out


def exp_receiver(rng, H_channel, quick):
    snrs = list(np.arange(10.0, 40.1, 2.5))
    n_sym = 60 if quick else 400
    nets = {}
    for ibo, seed in ((3.0, 11), (5.0, 12), (7.0, 13)):
        nets[str(int(ibo))] = train_network(rng, ibo, seed)
    flat = np.ones(UWA["subcarriers"], dtype=complex)
    linear = amp.NonlinearTransmitter(ibo_db=200.0)  # effectively linear
    res = {"snr_db": [float(s) for s in snrs], "symbols": n_sym, "order": 64, "networks": {}, "curves": {}}
    for key, (T, net, model, info) in nets.items():
        res["networks"][key] = {k: v for k, v in info.items()}
    # FFB bit error rate: AWGN and BELLHOP, 64-QAM, 5 dB back-off
    T5, _, m5, i5 = nets["5"]
    for label, H in (("awgn", flat), ("bellhop", H_channel)):
        lin = ber_run(rng, 64, linear, {"linear": (linear, 1.0, 0)}, H, snrs, n_sym)
        res["curves"][f"{label}_linear"] = lin["linear"]
        c = ber_run(rng, 64, T5, {"none": (T5, i5["alpha_exact"], 0), "ffb_nn_1": (m5, i5["alpha_network"], 1),
                                  "ffb_nn_3": (m5, i5["alpha_network"], 3), "ffb_exact_3": (T5, i5["alpha_exact"], 3)}, H, snrs, n_sym)
        for k, v in c.items():
            res["curves"][f"{label}_ibo5_{k}"] = v
    # back-off sweep over the BELLHOP channel (and AWGN, for the lab)
    for key in ("3", "7"):
        T, _, m, info = nets[key]
        for label, H in (("bellhop", H_channel), ("awgn", flat)):
            c = ber_run(rng, 64, T, {"none": (T, info["alpha_exact"], 0), "ffb_nn_3": (m, info["alpha_network"], 3)}, H, snrs, n_sym)
            for k, v in c.items():
                res["curves"][f"{label}_ibo{key}_{k}"] = v
    # a clipper at the saturation level ahead of the amplifier (the thesis "with clipping" case)
    Tc = amp.NonlinearTransmitter(ibo_db=5.0, clip_db=5.0)
    Xc, _ = ofdm.random_symbols(rng, 13, UWA["subcarriers"], 4)
    xc = ofdm.to_time(Xc, 4).ravel()[:25800]
    yc = Tc(xc)
    yc_meas = yc + 10 ** (-35 / 20) * np.sqrt(np.mean(np.abs(yc) ** 2) / 2) * (rng.standard_normal(yc.size) + 1j * rng.standard_normal(yc.size))
    net_c = neural.AmplifierNetwork(seed=14).fit(np.abs(xc).reshape(-1, 6), np.abs(yc_meas).reshape(-1, 6), seed=14)
    m_c = neural.NeuralAmplifierModel(net_c)
    c = ber_run(rng, 64, Tc, {"none": (Tc, amp.bussgang_gain(xc, yc), 0), "ffb_nn_3": (m_c, amp.bussgang_gain(xc, m_c(xc)), 3)}, H_channel, snrs, n_sym)
    for k, v in c.items():
        res["curves"][f"bellhop_ibo5clip5_{k}"] = v
    res["clip_network"] = {k: v for k, v in net_c.history.items() if not k.endswith("_curve")}
    res["efficiency_at_backoff"] = {str(b): r4(100 * papr.class_b_efficiency(b)) for b in (3, 5, 7, 10)}
    res["summary"] = {}
    for name, v in res["curves"].items():
        res["summary"][name] = {"at_30db": v[snrs.index(30.0)], "at_40db": v[snrs.index(40.0)], "snr_1e3": interp_snr(snrs, v, 1e-3)}
    # constellation snapshot (AWGN, 35 dB, 64-QAM, 5 dB back-off)
    X, _ = ofdm.random_symbols(rng, 4, UWA["subcarriers"], 64)
    _, Ytx = rx.transmit(X, 4, T5)
    Y = rx.add_noise(rng, Ytx, flat, 35.0)
    Z = Y / i5["alpha_network"]
    x_hat = ofdm.to_time(rx.ffb_detect(Y, flat, 64, m5, i5["alpha_network"], 4, iterations=3)[-1], 4)
    clean = Z - ofdm.to_freq(m5(x_hat) - i5["alpha_network"] * x_hat, UWA["subcarriers"]) / i5["alpha_network"]
    res["constellation"] = {"before": [[r4(v.real), r4(v.imag)] for v in Z.ravel()[:1400]],
                            "after": [[r4(v.real), r4(v.imag)] for v in clean.ravel()[:1400]]}
    # what FFB buys the amplifier: back-off needed without and with FFB for BER 1e-3 in AWGN at 30 dB
    return res


def lab_channel(arrivals):
    """The channel exactly as the browser rebuilds it from the exported arrivals."""
    arr = [{"delay_s": a["delay_ms"] / 1e3, "amp": a["amp"], "phase_deg": a["phase_deg"]} for a in arrivals]
    spacing = UWA["bandwidth_hz"] / UWA["subcarriers"]
    f = UWA["centre_hz"] + ofdm.subcarrier_offsets(UWA["subcarriers"], spacing)
    return chn.normalized(chn.frequency_response(arr, f))


def js_reference(rng, net5, arrivals):
    """Deterministic cases the browser lab must reproduce exactly (verified on load)."""
    n = 64
    X, bits = ofdm.random_symbols(rng, 2, n, 4)
    phases_idx = rng.integers(0, 4, size=(4, n)); phases_idx[0] = 0
    phases = np.exp(0.5j * np.pi * phases_idx)
    x4 = ofdm.to_time(X, 4)
    rcf_hist = tx.repeated_clipping_filtering(ofdm.to_time(X, 2), n, 3.0, 4)[1]
    _, slm_idx, slm_p = tx.selected_mapping(X, phases, 4)
    _, pts_idx, pts_p = tx.partial_transmit_sequences(X, 4, (1, -1), 4)
    X64, bits64 = ofdm.random_symbols(rng, 2, n, 64)
    T5 = amp.NonlinearTransmitter(ibo_db=5.0)
    model = neural.NeuralAmplifierModel(net5)
    y, Ytx = rx.transmit(X64, 4, T5)
    alpha = amp.bussgang_gain(ofdm.to_time(X64, 4), y)
    dec = rx.ffb_detect(Ytx, np.ones(n, dtype=complex), 64, model, alpha, 4, iterations=2)
    x_hat = ofdm.to_time(dec[1], 4)
    soft = (Ytx - ofdm.to_freq(model(x_hat) - alpha * x_hat, n)) / alpha
    amps = np.round(np.linspace(0.0, 3.5, 12), 4)   # exported exactly as used
    H = lab_channel(arrivals)
    return {
        "qpsk_bits": bits.tolist(), "qam64_bits": bits64.tolist(), "slm_phase_index": phases_idx.tolist(),
        "expect": {
            "papr_L4": [r4(v) for v in papr.papr_db(x4)],
            "rcf_cr3_L2_passes": [[r4(v) for v in h] for h in rcf_hist],
            "rct_r05_L4": [r4(v) for v in papr.papr_db(tx.root_compand(x4, 0.5))],
            "slm_u4_L4": {"index": slm_idx.tolist(), "papr": [r4(v) for v in slm_p]},
            "pts_v4_w2_L4": {"index": pts_idx.tolist(), "papr": [r4(v) for v in pts_p]},
            "network_ibo5": {"input": [r4(v) for v in amps], "output": [float(f"{v:.8f}") for v in net5.predict_amplitude(amps)]},
            "ffb_ibo5_alpha": float(f"{alpha:.10f}"),
            "ffb_ibo5_errors_per_pass": [int(np.sum(ofdm.demodulate(d, 64) != bits64)) for d in dec],
            "ffb_ibo5_soft_pass2": [[float(f"{v.real:.9f}"), float(f"{v.imag:.9f}")] for v in soft[0, :6]],
            "channel_H": [[k, float(f"{H[k].real:.9f}"), float(f"{H[k].imag:.9f}")] for k in (0, 1, 100, 255, 256, 400, 511)],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    q = args.quick
    t_start = time.time()
    rng = np.random.default_rng(2021)
    results = {"version": __version__, "quick": q, "uwa": UWA, "grid_db": GRID.tolist()}

    steps = [
        ("basics", lambda: exp_basics(np.random.default_rng(1), 4000 if q else 20000)),
        ("rcf_thesis", lambda: exp_rcf_thesis(np.random.default_rng(2), 4000 if q else 20000)),
        ("rcf_sweep", lambda: exp_rcf_sweep(np.random.default_rng(3), 2000 if q else 10000)),
        ("rct", lambda: exp_rct(np.random.default_rng(4), 4000 if q else 20000, 500 if q else 4000)),
        ("slm", lambda: exp_slm(np.random.default_rng(5), 2000 if q else 10000)),
        ("pts", lambda: exp_pts(np.random.default_rng(6), 1000 if q else 10000)),
        ("compare", lambda: exp_compare(np.random.default_rng(7), 2000 if q else 10000)),
    ]
    for name, fn in steps:
        t = time.time()
        results[name] = fn()
        print(f"{name:10s} done in {time.time() - t:5.1f} s", flush=True)

    t = time.time()
    ch = exp_channel()
    H = ch.pop("H")
    results["channel"] = ch
    results["receiver"] = exp_receiver(np.random.default_rng(8), H, q)
    print(f"receiver   done in {time.time() - t:5.1f} s", flush=True)

    net5 = neural.AmplifierNetwork()
    w = results["receiver"]["networks"]["5"]["weights"]
    net5.w1, net5.b1 = np.array(w["w1"]), np.array(w["b1"])
    net5.w2, net5.b2 = np.array(w["w2"]), np.array(w["b2"])
    net5.in_range, net5.out_range = tuple(w["in_range"]), tuple(w["out_range"])
    results["js_reference"] = js_reference(np.random.default_rng(9), net5, results["channel"]["arrivals"])
    results["runtime_s"] = round(time.time() - t_start, 1)

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, separators=(",", ":")), encoding="utf-8")
    print(f"wrote results/results.json in {results['runtime_s']} s")


if __name__ == "__main__":
    main()
