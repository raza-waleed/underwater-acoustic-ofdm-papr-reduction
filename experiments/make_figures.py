"""Draw every report figure from results/results.json (run run_all.py first)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

BG, PANEL, GRIDC, TEXT, MUTED = "#0a0e14", "#0d131e", "#2a3648", "#c7d0dc", "#8695a8"
CYAN, ORANGE, MAGENTA, GREEN, VIOLET, RED = "#22d3ee", "#f59e0b", "#e879f9", "#34d399", "#a78bfa", "#f87171"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG, "text.color": TEXT,
    "axes.edgecolor": GRIDC, "axes.labelcolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
    "font.size": 10.5, "axes.titlesize": 11, "legend.fontsize": 9, "legend.facecolor": PANEL,
    "legend.edgecolor": GRIDC, "grid.color": GRIDC, "grid.alpha": 0.6, "axes.grid": True,
    "lines.linewidth": 1.8,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=130)
    plt.close(fig)
    print("saved", name)


def ccdf_axes(ax, xmax=12.5, ymin=1e-4, xmin=2.0):
    ax.set_yscale("log")
    ax.set_ylim(ymin, 1.05)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("PAPR threshold z (dB)")
    ax.set_ylabel("CCDF  Pr[PAPR > z]")
    ax.axhline(1e-3, color=MUTED, lw=0.8, ls=":")


def plot_ccdf(ax, grid, values, **kw):
    v = np.array(values, dtype=float)
    v[v <= 0] = np.nan
    ax.plot(grid, v, **kw)


def main():
    r = json.loads((ROOT / "results" / "results.json").read_text(encoding="utf-8"))
    g = np.array(r["grid_db"])

    # 1 ------------------------------------------------------------ PAPR grows with N, and needs oversampling
    b = r["basics"]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for n, col in ((64, CYAN), (256, ORANGE), (1024, MAGENTA)):
        plot_ccdf(ax, g, b["curves"][f"N{n}_L4"], color=col, label=f"N = {n}, oversampled x4")
        plot_ccdf(ax, g, b["curves"][f"N{n}_L1"], color=col, ls="--", lw=1.2, label=f"N = {n}, Nyquist rate")
        plot_ccdf(ax, g, b["curves"][f"theory_N{n}_L1"], color=col, ls=":", lw=1.0)
    ccdf_axes(ax, 13, 1e-4, 5)
    ax.set_title(r"OFDM PAPR: dotted is the closed form $1-(1-e^{-z})^N$; Nyquist rate sampling misses peaks")
    ax.legend(ncol=2, loc="lower left")
    save(fig, "papr_ccdf_basics.png")

    # 2 ------------------------------------------------------------ RCF in the thesis setting
    rc = r["rcf_thesis"]
    fig, ax = plt.subplots(figsize=(10.4, 4.4))
    plot_ccdf(ax, g, rc["curves"]["original"], color=CYAN, label=f"Original ({rc['original']:.2f} dB)")
    for i, col in enumerate((ORANGE, GREEN, VIOLET, MAGENTA)):
        plot_ccdf(ax, g, rc["curves"][f"pass{i + 1}"], color=col, label=f"RCF, {i + 1} pass{'es' if i else ''} ({rc['passes'][i]:.2f} dB)")
    plot_ccdf(ax, g, rc["curves"]["clip_only"], color=MUTED, ls="--", lw=1.2, label=f"Clipping only, no filter ({rc['clip_only']:.2f} dB)")
    ccdf_axes(ax, 12, 1e-4, 4)
    ax.set_title("Repeated clipping and filtering: 128 subcarriers, I = 2, CR = 3 (4.77 dB)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    save(fig, "rcf_ccdf.png")

    # 3 ------------------------------------------------------------ why the filter matters
    sp = rc["spectrum"]
    size = len(sp["original"])
    f = (np.arange(size) - size // 2) / (size / 2)
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    ax.plot(f, sp["clip_only"], color=ORANGE, lw=1.3, label=f"Clipping only: out-of-band power {rc['oob_clip_only_db']:.1f} dB")
    ax.plot(f, sp["rcf"], color=GREEN, lw=1.3, label="RCF: regrowth filtered out every pass")
    ax.plot(f, sp["original"], color=CYAN, lw=1.0, label="Original OFDM")
    ax.set_ylim(-60, 5)
    ax.set_xlabel("frequency / (fs / 2), with I = 2 oversampling")
    ax.set_ylabel("power spectrum (dB)")
    ax.axvspan(-0.5, 0.5, color=CYAN, alpha=0.06)
    ax.set_title("Clipping spreads energy outside the band; the filter in RCF removes it")
    ax.legend(loc="lower center")
    save(fig, "rcf_spectrum.png")

    # 4 ------------------------------------------------------------ RCF parameter study
    rows = r["rcf_sweep"]["rows"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for I, col in ((1, CYAN), (2, ORANGE), (3, GREEN), (4, MAGENTA)):
        rr = [x for x in rows if x["oversampling"] == I]
        crs = [x["cr_db"] for x in rr]
        a1.plot(crs, [x["papr_1e3"] for x in rr], "-o", color=col, ms=4, label=f"I = {I}")
        ok = [x for x in rr if x["snr_1e4"] is not None]
        a2.plot([x["cr_db"] for x in ok], [x["snr_1e4"] for x in ok], "-o", color=col, ms=4, label=f"I = {I}")
    a1.set_xlabel("clipping ratio CR (dB)")
    a1.set_ylabel("PAPR at CCDF 1e-3 (dB), 4 passes")
    a1.set_title("PAPR after four RCF passes")
    a1.legend()
    a2.axhline(r["rcf_sweep"]["reference_snr_1e4"], color=MUTED, ls=":", lw=1)
    a2.text(4.4, r["rcf_sweep"]["reference_snr_1e4"] - 0.9, "no clipping", color=MUTED, fontsize=9)
    a2.set_xlabel("clipping ratio CR (dB)")
    a2.set_ylabel("SNR per subcarrier for BER 1e-4 (dB)")
    a2.set_title("The price: SNR needed in AWGN, QPSK")
    a2.set_ylim(10, 30)
    save(fig, "rcf_tradeoff.png")

    # 5 ------------------------------------------------------------ rooting companding
    rt = r["rct"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.2))
    plot_ccdf(a1, g, rt["curves"]["original"], color=CYAN, label=f"Original ({rt['L4']['original']:.2f} dB)")
    for R, col in (("0.9", ORANGE), ("0.7", GREEN), ("0.5", MAGENTA), ("0.3", VIOLET)):
        plot_ccdf(a1, g, rt["curves"][f"R{R}"], color=col, label=f"R = {R} ({rt['L4'][R]:.2f} dB)")
    ccdf_axes(a1, 12, 1e-4, 2)
    a1.set_title(r"Rooting companding $|x|^R$, oversampled x4")
    a1.legend(loc="lower left", fontsize=8.5)
    s = rt["ber"]["snr_db"]
    for key, col, lab in (("none", CYAN, "No companding"), ("0.9", ORANGE, "R = 0.9"), ("0.7", GREEN, "R = 0.7"), ("0.5", MAGENTA, "R = 0.5")):
        v = np.array(rt["ber"]["curves"][key], dtype=float)
        v[v <= 0] = np.nan
        a2.semilogy(s, v, "-o", color=col, ms=3, label=lab)
    a2.set_ylim(1e-5, 0.5)
    a2.set_xlim(0, 20)
    a2.set_xlabel("SNR (dB)")
    a2.set_ylabel("bit error rate")
    a2.set_title(r"BER in AWGN, receiver expands $|y|^{1/R}$")
    a2.legend()
    save(fig, "rct.png")

    # 6 ------------------------------------------------------------ SLM and PTS
    sl, pt = r["slm"], r["pts"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for U, col in (("1", CYAN), ("2", ORANGE), ("4", GREEN), ("8", VIOLET), ("16", MAGENTA)):
        plot_ccdf(a1, g, sl["curves"][f"U{U}"], color=col, label=f"U = {U} ({sl['L4'][U]:.2f} dB)")
        plot_ccdf(a1, g, sl["curves"][f"theory_U{U}"], color=col, ls=":", lw=1.0)
    ccdf_axes(a1, 12.5, 1e-4, 5)
    a1.set_title("Selected mapping, N = 256 (dotted: theory at Nyquist rate)")
    a1.legend(loc="lower left", fontsize=8.5)
    plot_ccdf(a2, g, sl["curves"]["U1"], color=CYAN, label=f"Original ({sl['L4']['1']:.2f} dB)")
    for row, col in zip(pt["rows"], (ORANGE, GREEN, MAGENTA, VIOLET)):
        V, W = row["subblocks"], row["phases"]
        bits = f"{row['side_bits']} bit" + ("" if row["side_bits"] == 1 else "s")
        plot_ccdf(a2, g, pt["curves"][f"V{V}_W{W}"], color=col, label=f"V = {V}, W = {W} ({row['papr_1e3']:.2f} dB, {bits})")
    ccdf_axes(a2, 12.5, 1e-4, 5)
    a2.set_title("Partial transmit sequences, N = 256")
    a2.legend(loc="lower left", fontsize=8.5)
    save(fig, "slm_pts.png")

    # 7 ------------------------------------------------------------ one footing comparison
    cmp_rows = r["compare"]["rows"]
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    names = [x["method"] for x in cmp_rows]
    vals = [x["papr_1e3"] for x in cmp_rows]
    cols = [CYAN if x["kind"] == "reference" else (ORANGE if x["kind"] == "distortion" else GREEN) for x in cmp_rows]
    bars = ax.barh(names[::-1], vals[::-1], color=cols[::-1])
    for bar, row in zip(bars, cmp_rows[::-1]):
        ax.text(bar.get_width() + 0.12, bar.get_y() + bar.get_height() / 2,
                f"{row['papr_1e3']:.2f} dB   class B {row['class_b']:.0f}%", va="center", fontsize=9, color=TEXT)
    ax.set_xlim(0, 14.5)
    ax.set_xlabel("PAPR at CCDF 1e-3 (dB); N = 128, oversampled x4, QPSK")
    ax.set_title("Every method on one footing (orange: adds distortion, green: needs side information)")
    ax.grid(axis="y", alpha=0)
    save(fig, "method_comparison.png")

    # 8 ------------------------------------------------------------ amplifier and network
    nets = r["receiver"]["networks"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.2))
    a_in = np.array(nets["5"]["amam"]["input"])
    a1.plot(a_in, a_in, color=MUTED, lw=1, ls="--", label="Linear")
    for key, col in (("3", MAGENTA), ("5", ORANGE), ("7", GREEN)):
        a1.plot(a_in, nets[key]["amam"]["true"], color=col, label=f"Rapp SSPA, back-off {key} dB")
        a1.plot(a_in, nets[key]["amam"]["network"], color=col, ls=":", lw=2.2)
    edge = max(nets[k]["train_amplitude_999"] for k in nets)
    a1.axvspan(edge, 4.0, color=MUTED, alpha=0.08)
    a1.text(edge + 0.05, 0.25, "fewer than 1 in 1000\ntraining samples", color=MUTED, fontsize=8.5)
    a1.set_xlabel("input amplitude (x RMS)")
    a1.set_ylabel("output amplitude")
    a1.set_title("Amplifier (solid) and what the network learned (dotted)")
    a1.set_xlim(0, 4)
    a1.set_ylim(0, 4)
    a1.legend(loc="upper left")
    h = nets["5"]
    a2.semilogy(h["train_curve"], color=CYAN, label="training")
    a2.semilogy(h["validation_curve"], color=ORANGE, label="validation")
    a2.axvline(h["best_epoch"], color=MUTED, ls=":", lw=1)
    a2.set_xlabel("Levenberg-Marquardt epoch")
    a2.set_ylabel("mean squared error")
    a2.set_title(f"6-12-6 network, back-off 5 dB: best epoch {h['best_epoch']}, {h['seconds']:.1f} s")
    a2.legend()
    save(fig, "amplifier_network.png")

    # 9 ------------------------------------------------------------ BELLHOP channel
    c = r["channel"]
    fig = plt.figure(figsize=(10.8, 4.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1.6, 1.6])
    a0, a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    ssp = np.array(c["geometry"]["ssp"])
    a0.plot(ssp[:, 1], ssp[:, 0], color=CYAN)
    a0.plot([ssp[:, 1].min() - 1], [c["geometry"]["source_m"]], "o", color=ORANGE)
    a0.plot([ssp[:, 1].min() - 1], [c["geometry"]["receiver_m"]], "s", color=GREEN)
    a0.text(ssp[:, 1].min(), c["geometry"]["source_m"] - 3, " source 30 m", color=ORANGE, fontsize=8)
    a0.text(ssp[:, 1].min(), c["geometry"]["receiver_m"] + 7, " hydrophone 50 m", color=GREEN, fontsize=8)
    a0.set_ylim(100, 0)
    a0.set_xlabel("sound speed (m/s)")
    a0.set_ylabel("depth (m)")
    a0.set_title("Profile, 100 m")
    d_ms = [a["delay_ms"] for a in c["arrivals"]]
    amp_db = [20 * np.log10(a["amp"]) for a in c["arrivals"]]
    a1.vlines(d_ms, -50, amp_db, color=CYAN, lw=1.2)
    a1.plot(d_ms, amp_db, "o", color=CYAN, ms=3.5)
    a1.text(95, -7, f"rms delay spread {c['rms_delay_ms']:.1f} ms,\n{100 - c['energy_after_ms']['100']:.1f}% of the energy within 100 ms", color=ORANGE, fontsize=8.5)
    a1.set_ylim(-50, 3)
    a1.set_xlabel("delay after the first arrival (ms)")
    a1.set_ylabel("arrival amplitude (dB)")
    a1.set_title(f"BELLHOP arrivals at 2 km ({len(d_ms)} paths)")
    rf = c["response_db"]
    a2.plot(np.array(rf["freq_hz"]) / 1e3, rf["mag_db"], color=VIOLET, lw=1.0)
    a2.set_xlabel("frequency (kHz)")
    a2.set_ylabel("|H(f)| (dB, mean power 0 dB)")
    a2.set_title("Frequency response over the 6.25 kHz band")
    save(fig, "bellhop_channel.png")

    # 10 ------------------------------------------------------------ FFB bit error rate
    rc = r["receiver"]
    s = rc["snr_db"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), sharey=True)
    for ax, label, title in ((axes[0], "awgn", "White noise channel"), (axes[1], "bellhop", "BELLHOP 2 km channel, perfect CSI")):
        for key, col, lab, ls in ((f"{label}_ibo5_none", RED, "No distortion removal", "-"),
                                  (f"{label}_ibo5_ffb_nn_1", ORANGE, "FFB, learned model, 1 pass", "-"),
                                  (f"{label}_ibo5_ffb_nn_3", GREEN, "FFB, learned model, 3 passes", "-"),
                                  (f"{label}_ibo5_ffb_exact_3", MAGENTA, "FFB, exact model, 3 passes", ":"),
                                  (f"{label}_linear", CYAN, "Perfectly linear amplifier", "--")):
            v = np.array(rc["curves"][key], dtype=float)
            v[v <= 0] = np.nan
            ax.semilogy(s, v, ls, color=col, marker="o", ms=3, label=lab)
        ax.set_title(f"{title}, 64-QAM, back-off 5 dB")
        ax.set_xlabel("SNR per subcarrier (dB)")
        ax.set_ylim(1e-6, 0.3)
    axes[0].set_ylabel("bit error rate")
    axes[0].legend(loc="lower left", fontsize=8.5)
    save(fig, "ffb_ber.png")

    # 11 ------------------------------------------------------------ back-off sweep
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    for ibo, col in (("3", MAGENTA), ("5", ORANGE), ("7", GREEN)):
        for suffix, ls, lab in (("none", "--", "without FFB"), ("ffb_nn_3", "-", "with FFB")):
            v = np.array(rc["curves"][f"bellhop_ibo{ibo}_{suffix}"], dtype=float)
            v[v <= 0] = np.nan
            ax.semilogy(s, v, ls, color=col, marker="o", ms=3, label=f"back-off {ibo} dB, {lab}")
    v = np.array(rc["curves"]["bellhop_linear"], dtype=float)
    v[v <= 0] = np.nan
    ax.semilogy(s, v, ":", color=CYAN, lw=2.2, label="linear amplifier")
    ax.set_xlabel("SNR per subcarrier (dB)")
    ax.set_ylabel("bit error rate")
    ax.set_title("64-QAM over the BELLHOP channel: FFB lets the amplifier run hard")
    ax.set_ylim(1e-4, 0.3)
    ax.legend(ncol=2, fontsize=8.5, loc="lower left")
    save(fig, "ffb_backoff.png")

    # 12 ------------------------------------------------------------ constellations
    k = rc["constellation"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.4))
    for ax, key, title, col in ((axes[0], "before", "Before FFB", RED), (axes[1], "after", "After three FFB passes", GREEN)):
        p = np.array(k[key])
        ax.plot(p[:, 0], p[:, 1], ".", color=col, ms=2.2, alpha=0.8)
        ax.set_aspect("equal")
        ax.set_xlim(-1.45, 1.45)
        ax.set_ylim(-1.45, 1.45)
        ax.set_title(title)
        ax.set_xlabel("in phase")
    axes[0].set_ylabel("quadrature")
    fig.suptitle("64-QAM, amplifier back-off 5 dB, SNR 35 dB", color=TEXT, fontsize=11)
    save(fig, "constellation.png")


if __name__ == "__main__":
    main()
