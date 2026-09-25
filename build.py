"""Build the report page from results/results.json.

    python build.py
        writes papr-reduction-report.html: a standalone copy of the page that
        opens in any browser from disk (the site stylesheet is inlined)

    python build.py --site C:/Users/bhutt/projects/waleedrazadev [--doi 10.5281/zenodo.NNN]
        also writes <site>/underwater-acoustics/papr-reduction/index.html with the
        waleedraza.dev navigation, footer and scripts, plus its figures

Both variants share report/content.html and report/page.css, so they look the
same. Every number comes from results.json through values(); the build stops if
the content asks for a value that does not exist, or if the text contains an em
dash, an en dash or a double hyphen.
"""
import argparse
import json
import math
import re
import shutil
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "report"
REPO_URL = "https://github.com/raza-waleed/underwater-acoustic-ofdm-papr-reduction"
SITE_DIR = "underwater-acoustics/papr-reduction"
GA_TAG = """<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-XKMNX734H9"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-XKMNX734H9');
</script>
"""

SITE_NAV = """<nav class="topnav">
<div class="wrap">
<div class="brand">Waleed Raza</div>
<button class="nav-toggle" aria-label="Toggle menu" aria-expanded="false">
<span></span><span></span><span></span>
</button>
<ul>
<li><a href='/#about'>Home</a></li>
<li><a href='/#research'>Research</a></li>
<li><a href='/simulations'>Simulations</a></li>
<li><a href='/route-planning'>Routes</a></li>
<li><a href='/aircraft-detection'>Detection</a></li>
<li><a class='active' href='/underwater-acoustics'>Acoustics</a></li>
<li><a href='/power-grid'>Grid</a></li>
<li><a href='/#education'>Education</a></li>
<li><a href='/#experience'>Experience</a></li>
<li><a href='/#publications'>Publications</a></li>
<li><a href='/#awards'>Awards</a></li>
<li><a href='/#service'>Service</a></li>
<li><a href='/#contact'>Contact</a></li>
<span id="nav-indicator"></span>
</ul>
</div>
</nav>"""

SITE_FOOTER = """<footer>
<div class="wrap">
<a class='back-link' href='/underwater-acoustics'>&larr; back to underwater acoustics</a><br>
&copy; <span id="year"></span> WALEED RAZA &middot; BUILT WITH HTML, CSS &amp; GSAP &middot; HOSTED ON NETLIFY
</div>
</footer>"""

SITE_SCRIPTS = """<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/ScrollTrigger.min.js"></script>
<script src="/app.js"></script>"""

SITE_HEAD = """<link rel="icon" href="/favicon.ico" sizes="any">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="apple-touch-icon" href="/apple-touch-icon.png">
<link rel="stylesheet" href="/style.css">
"""

STANDALONE_NAV = f"""<nav class="topnav">
<div class="wrap">
<div class="brand">Waleed Raza</div>
<ul class="mini">
<li><a href="https://waleedraza.dev">Portfolio</a></li>
<li><a class="active" href="https://waleedraza.dev/underwater-acoustics">Acoustics</a></li>
<li><a href="{REPO_URL}">Code</a></li>
</ul>
</div>
</nav>"""

STANDALONE_FOOTER = f"""<footer>
<div class="wrap">
<a class='back-link' href='https://waleedraza.dev/underwater-acoustics/papr-reduction/'>&rarr; live page on waleedraza.dev</a><br>
&copy; {date.today().year} WALEED RAZA
</div>
</footer>"""

# Without the site's GSAP script nothing would reveal .reveal elements, and the
# mobile menu could not open, so the standalone copy shows both directly.
STANDALONE_OVERRIDES = """
.reveal { opacity: 1; }
nav.topnav .wrap { flex-wrap: nowrap; }
nav.topnav ul.mini { display: flex; flex-direction: row; width: auto; flex-basis: auto; padding: 0; gap: 2px; }
nav.topnav ul.mini a { border: none; padding: 6px 7px; width: auto; }
"""


def f1(v):
    return f"{v:.1f}"


def f2(v):
    return f"{v:.2f}"


def up2(v):
    """Upper bound to two decimals, for "within X dB" statements (0.145 -> 0.15)."""
    return f"{math.ceil(v * 100 - 1e-9) / 100:.2f}"


def sci(v):
    """1.7x10^-2 as HTML, or 0 when no errors were counted."""
    if v <= 0:
        return "0"
    e = math.floor(math.log10(v))
    m = v / 10 ** e
    if round(m, 1) >= 10:
        m, e = m / 10, e + 1
    return f"{m:.1f}&times;10<sup>{e}</sup>"


def class_b(db):
    return 100 * (math.pi / 4) * 10 ** (-db / 20)


def values(r):
    rc, sw, rt, sl, pt, cp, ch, rx = (r["rcf_thesis"], r["rcf_sweep"], r["rct"], r["slm"], r["pts"], r["compare"], r["channel"], r["receiver"])
    rows = {(x["oversampling"], x["cr"]): x for x in sw["rows"]}
    cmp_rows = {x["method"]: x for x in cp["rows"]}
    net5 = rx["networks"]["5"]
    s = rx["summary"]
    snr = rx["snr_db"]
    cur = rx["curves"]
    at = lambda key, db: cur[key][snr.index(db)]
    ptsrow = {(x["subblocks"], x["phases"]): x for x in pt["rows"]}
    basics = r["basics"]
    return {
        # PAPR basics
        "n128_L4": f2(basics["L4"]["128"]), "n512_L4": f2(basics["L4"]["512"]),
        "theory_gap_max": up2(max(abs(basics["L1"][k] - basics["theory_L1"][k]) for k in basics["L1"])),
        "oversampling_gap_min": f2(min(basics["L4"][k] - basics["L1"][k] for k in basics["L1"])),
        "oversampling_gap_max": f2(max(basics["L4"][k] - basics["L1"][k] for k in basics["L1"])),
        "symbols_basics": f"{20000:,}",
        # RCF in the thesis setting
        "rcf_original": f2(rc["original"]), "rcf_p1": f2(rc["passes"][0]), "rcf_p2": f2(rc["passes"][1]),
        "rcf_p3": f2(rc["passes"][2]), "rcf_p4": f2(rc["passes"][3]), "rcf_gain": f2(rc["original"] - rc["passes"][3]),
        "rcf_clip_only": f2(rc["clip_only"]), "rcf_sdr": f1(rc["sdr_db"]), "rcf_symbols": f"{rc['config']['symbols']:,}",
        "eff_orig": f1(class_b(rc["original"])), "eff_rcf": f1(class_b(rc["passes"][3])),
        "eff_ratio_rcf": f2(class_b(rc["passes"][3]) / class_b(rc["original"])),
        # RCF parameter study
        "ref_snr": f2(sw["reference_snr_1e4"]), "sweep_symbols": f"{sw['symbols']:,}",
        "i2cr3_papr": f2(rows[(2, 3.0)]["papr_1e3"]), "i2cr3_pen": f2(rows[(2, 3.0)]["penalty_db"]),
        "i2cr2_papr": f2(rows[(2, 2.0)]["papr_1e3"]), "i2cr2_pen": f2(rows[(2, 2.0)]["penalty_db"]),
        "i1cr15_snr": f1(rows[(1, 1.5)]["snr_1e4"]),
        # rooting companding
        "rct_thesis_orig": f2(rt["thesis_setting"]["original"]), "rct_thesis_comp": f2(rt["thesis_setting"]["companded"]),
        "rct_L4_orig": f2(rt["L4"]["original"]), "rct_R03": f2(rt["L4"]["0.3"]), "rct_R05": f2(rt["L4"]["0.5"]),
        "rct_R07": f2(rt["L4"]["0.7"]), "rct_R09": f2(rt["L4"]["0.9"]),
        "rct_snr_05": f2(rt["ber"]["snr_1e4"]["0.5"]),
        "rct_pen_05": f2(rt["ber"]["snr_1e4"]["0.5"] - rt["ber"]["snr_1e4"]["none"]),
        "rct_oob": f1(cmp_rows["Rooting companding (R 0.5)"]["oob_db"]), "rct_sdr": f1(cmp_rows["Rooting companding (R 0.5)"]["sdr_db"]),
        "rct_oob_pct": f1(100 * 10 ** (cmp_rows["Rooting companding (R 0.5)"]["oob_db"] / 10)),
        "clip_oob_pct": f1(100 * 10 ** (cmp_rows["Clipping only (CR 3)"]["oob_db"] / 10)),
        # SLM and PTS
        "slm_u1": f2(sl["L4"]["1"]), "slm_u16": f2(sl["L4"]["16"]), "slm_gain16": f2(sl["L4"]["1"] - sl["L4"]["16"]),
        "slm_theory_dev": up2(max(abs(sl["L1"][u] - sl["theory_L1"][u]) for u in sl["L1"] if u != "1")),
        "pts_v4": f2(ptsrow[(4, 2)]["papr_1e3"]), "pts_v8": f2(ptsrow[(8, 2)]["papr_1e3"]),
        "pts_v4w4": f2(ptsrow[(4, 4)]["papr_1e3"]), "slmpts_symbols": f"{sl['symbols']:,}",
        "cmp_orig": f2(cmp_rows["Original OFDM"]["papr_1e3"]), "cmp_eff_orig": f1(cmp_rows["Original OFDM"]["class_b"]),
        # channel
        "ch_first": f"{ch['first_arrival_s']:.4f}", "ch_straight": f"{ch['straight_line_s']:.4f}",
        "ch_err": f"{ch['first_arrival_error_pct']:.3f}", "ch_rms": f1(ch["rms_delay_ms"]), "ch_max": f"{ch['max_delay_ms']:.0f}",
        "ch_in100": f1(100 - ch["energy_after_ms"]["100"]), "ch_n": str(len(ch["arrivals"])),
        "ch_fade_min": f1(ch["fade_stats"]["min_db"]), "ch_fade_10": f1(ch["fade_stats"]["below_10db_pct"]),
        # network
        "net5_epochs": str(net5["best_epoch"]), "net5_sec": f1(net5["seconds"]),
        "net5_val": sci(net5["validation_mse"]), "net5_test": sci(net5["test_mse"]),
        "net5_alpha": f"{net5['alpha_network']:.4f}", "net5_alpha_exact": f"{net5['alpha_exact']:.4f}",
        "net_cross_max": f1(100 * max(n["cross_sensitivity"] for n in rx["networks"].values())),
        "net_err_max": f"{max(n['max_error_below_3rms'] for n in rx['networks'].values()):.3f}",
        # receiver
        "rx_symbols": str(rx["symbols"]), "rx_bits": f"{rx['symbols'] * 512 * 6:,}",
        "awgn5_none40": sci(at("awgn_ibo5_none", 40.0)),
        "awgn5_snr_ffb": f2(s["awgn_ibo5_ffb_nn_3"]["snr_1e3"]), "awgn5_snr_lin": f2(s["awgn_linear"]["snr_1e3"]),
        "bh5_none40": sci(at("bellhop_ibo5_none", 40.0)), "bh5_ffb40": sci(at("bellhop_ibo5_ffb_nn_3", 40.0)),
        "bh5_exact40": sci(at("bellhop_ibo5_ffb_exact_3", 40.0)), "bh_lin40": sci(at("bellhop_linear", 40.0)),
        "bh3_none40": sci(at("bellhop_ibo3_none", 40.0)), "bh3_ffb40": sci(at("bellhop_ibo3_ffb_nn_3", 40.0)),
        "bh3_ratio": f"{at('bellhop_ibo3_none', 40.0) / at('bellhop_ibo3_ffb_nn_3', 40.0):.0f}",
        "bh_snr_lin": f1(s["bellhop_linear"]["snr_1e3"]), "bh3_snr_ffb": f1(s["bellhop_ibo3_ffb_nn_3"]["snr_1e3"]),
        "bh7_snr_none": f1(s["bellhop_ibo7_none"]["snr_1e3"]),
        "clip_none40": sci(at("bellhop_ibo5clip5_none", 40.0)), "clip_ffb40": sci(at("bellhop_ibo5clip5_ffb_nn_3", 40.0)),
        "eff3": f1(rx["efficiency_at_backoff"]["3"]), "eff7": f1(rx["efficiency_at_backoff"]["7"]),
        "eff_ratio_ffb": f2(rx["efficiency_at_backoff"]["3"] / rx["efficiency_at_backoff"]["7"]),
        "runtime": f"{r['runtime_s'] / 60:.0f}",
    }


def table_sweep(r):
    rows = [x for x in r["rcf_sweep"]["rows"] if x["oversampling"] in (1, 2, 4)]
    out = ['<div class="table-wrap"><table class="data"><thead><tr><th>Oversampling I</th><th>CR (dB)</th>'
           '<th>PAPR at 10<sup>-3</sup> (dB)</th><th>In-band SDR (dB)</th><th>SNR for BER 10<sup>-4</sup> (dB)</th>'
           '<th>SNR cost (dB)</th></tr></thead><tbody>']
    for x in rows:
        snr = f2(x["snr_1e4"]) if x["snr_1e4"] is not None else "floor above 10<sup>-4</sup>"
        cost = f2(x["penalty_db"]) if x["penalty_db"] is not None else "&infin;"
        out.append(f"<tr><td>{x['oversampling']}</td><td>{x['cr']:g} ({x['cr_db']:.2f})</td><td>{x['papr_1e3']:.2f}</td>"
                   f"<td>{x['sdr_db']:.1f}</td><td>{snr}</td><td>{cost}</td></tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def table_compare(r):
    out = ['<div class="table-wrap"><table class="data"><thead><tr><th>Method</th><th>PAPR at 10<sup>-3</sup> (dB)</th>'
           '<th>In-band SDR (dB)</th><th>Out-of-band power (dB)</th><th>Side information (bits)</th>'
           '<th>FFT work per symbol</th><th>Class B efficiency</th></tr></thead><tbody>']
    for x in r["compare"]["rows"]:
        sdr = "none" if x["sdr_db"] is None else f"{x['sdr_db']:.1f}"
        oob = "none" if x["oob_db"] is None else f"{x['oob_db']:.1f}"
        out.append(f"<tr><td>{x['method']}</td><td>{x['papr_1e3']:.2f}</td><td>{sdr}</td><td>{oob}</td><td>{x['side_bits']}</td>"
                   f"<td>{x['ifft_per_symbol']}</td><td>{x['class_b']:.1f}% ({x['battery_factor']:.2f}x)</td></tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def lab_data(r):
    rx = r["receiver"]
    return {
        "uwa": r["uwa"],
        "channel": {k: r["channel"][k] for k in ("geometry", "arrivals", "first_arrival_s", "straight_line_s", "first_arrival_error_pct",
                                                 "rms_delay_ms", "max_delay_ms", "energy_after_ms", "response_db")},
        "networks": {k: {"weights": v["weights"], "alpha_network": v["alpha_network"], "alpha_exact": v["alpha_exact"]} for k, v in rx["networks"].items()},
        "receiver": {"snr_db": rx["snr_db"], "curves": {k: [float(f"{x:.4g}") for x in v] for k, v in rx["curves"].items()}},
        "reference": r["js_reference"],
    }


def fill(template, slots, strict=True):
    missing = set()

    def sub(m):
        key = m.group(1)
        if key in slots:
            return slots[key]
        missing.add(key)
        return m.group(0)

    out = re.sub(r"\{\{([a-zA-Z0-9_]+)\}\}", sub, template)
    if strict and missing:
        raise SystemExit("the page asks for unknown values: " + ", ".join(sorted(missing)))
    return out


def check_text(html):
    """Project style: no em or en dashes and no double hyphens in visible text."""
    visible = re.sub(r"<script.*?</script>|<style.*?</style>|<!--.*?-->", "", html, flags=re.S)
    visible = re.sub(r"<pre.*?</pre>", "", visible, flags=re.S)
    if "\u2014" in visible or "\u2013" in visible or re.search(r"(?<![-!])--(?![->])", re.sub(r"<[^>]+>", "", visible)):
        raise SystemExit("dash check failed: remove em/en dashes and double hyphens from the page text")


def page(r, variant, doi=None):
    V = values(r)
    doi_button = (f' <a class="btn" href="https://doi.org/{doi}" target="_blank" rel="noopener">DOI {doi}</a>' if doi else "")
    content = fill((REPORT / "content.html").read_text(encoding="utf-8"), {
        **V, "table_sweep": table_sweep(r), "table_compare": table_compare(r), "fig": "figures/", "repo_url": REPO_URL,
        "doi_button": doi_button,
        "acoustics_url": "/underwater-acoustics" if variant == "site" else "https://waleedraza.dev/underwater-acoustics",
    })
    site = variant == "site"
    base_css = "" if site else (REPORT / "site-style.css").read_text(encoding="utf-8") + STANDALONE_OVERRIDES
    html = fill((REPORT / "shell.html").read_text(encoding="utf-8"), {
        "head_extra": GA_TAG if site else "",
        "head_links": SITE_HEAD if site else "",
        "base_css": base_css,
        "page_css": (REPORT / "page.css").read_text(encoding="utf-8"),
        "lab_css": (ROOT / "lab" / "papr-lab.css").read_text(encoding="utf-8"),
        "nav": SITE_NAV if site else STANDALONE_NAV,
        "footer": SITE_FOOTER if site else STANDALONE_FOOTER,
        "scripts": SITE_SCRIPTS if site else "",
        "content": content,
        "lab_data": json.dumps(lab_data(r), separators=(",", ":")).replace("</", "<\\/"),
        "lab_js": (ROOT / "lab" / "papr-lab.js").read_text(encoding="utf-8"),
    })
    check_text(html)
    return html


def build(site=None, doi=None):
    r = json.loads((ROOT / "results" / "results.json").read_text(encoding="utf-8"))
    if r.get("quick"):
        print("warning: results.json came from a --quick run")
    html = page(r, "standalone", doi)
    (ROOT / "papr-reduction-report.html").write_text(html, encoding="utf-8")
    print(f"wrote papr-reduction-report.html ({len(html) / 1024:.0f} KB)")
    if site:
        dest = Path(site) / SITE_DIR
        dest.mkdir(parents=True, exist_ok=True)
        html = page(r, "site", doi)
        (dest / "index.html").write_text(html, encoding="utf-8")
        figs = dest / "figures"
        figs.mkdir(exist_ok=True)
        for png in sorted((ROOT / "figures").glob("*.png")):
            shutil.copy2(png, figs / png.name)
        print(f"wrote {dest / 'index.html'} ({len(html) / 1024:.0f} KB) and {len(list(figs.glob('*.png')))} figures")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", help="waleedraza.dev source folder")
    ap.add_argument("--doi", help="Zenodo DOI, once the release is archived")
    a = ap.parse_args()
    build(a.site, a.doi)
