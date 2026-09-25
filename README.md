# PAPR Reduction and Neural Network Distortion Removal for Underwater Acoustic OFDM

My M.Eng. thesis work at Harbin Engineering University (2018 to 2021). Battery powered
underwater acoustic modems lose much of their energy in the transmit power amplifier,
because OFDM peaks sit about 11 dB above the average power. My thesis attacked that
from both ends of the link: peak-to-average power ratio (PAPR) reduction at the
transmitter, and a receiver that learns the power amplifier with a small neural network
and removes its distortion by frequentative decision feedback (FFB).

Read it with the interactive lab at
[waleedraza.dev/underwater-acoustics/papr-reduction](https://waleedraza.dev/underwater-acoustics/papr-reduction/),
or open [`papr-reduction-report.html`](papr-reduction-report.html) in any browser
(a standalone copy, no server needed).

## What is here

- **Transmitter methods:** repeated clipping and filtering (RCF) and rooting companding
  (RCT) from my thesis, plus selected mapping (SLM) and partial transmit sequences (PTS)
  from my published papers, in `paprlab/transmitter.py`.
- **Receiver:** a 6-12-6 neural network trained with Levenberg-Marquardt on measured
  amplifier samples (`paprlab/neural.py`, numpy only), used inside the FFB loop
  (`paprlab/receiver.py`).
- **Channel:** a BELLHOP ray tracing model of the 2 km shallow water link in 100 m of
  water, turned into the channel frequency response (`channel/`, `paprlab/channel.py`).
- **Lab:** the same engine in dependency free JavaScript (`lab/papr-lab.js`). On every
  page load it recomputes nine cases exported by Python and reports whether they match.

## Key results

- Four passes of repeated clipping and filtering (clipping ratio 3, twofold
  oversampling) take the PAPR at CCDF 10^-3 from 10.97 dB to 5.39 dB, which lifts an
  ideal class B amplifier from 22.2% to 42.2% average efficiency.
- The clipping ratio study shows where the tradeoff turns: below a clipping ratio of 2
  the in-band distortion grows faster than the peaks shrink.
- SLM with 16 candidates takes the PAPR from 11.15 dB to 8.03 dB and matches its closed
  form within 0.06 dB; rooting companding with R = 0.5 costs 2.7 dB of SNR and needs a
  band limiting filter.
- With 64-QAM at 3 dB amplifier back-off over the BELLHOP channel, FFB with the learned
  network cuts the bit error rate at 40 dB SNR from 1.7x10^-2 to 3.5x10^-4, the same as
  a perfectly linear amplifier. Of the 3, 5 and 7 dB back-off levels, only 7 dB reaches
  a bit error rate of 10^-3 without FFB; with FFB, 3 dB is enough, which an ideal class
  B amplifier turns into 55.6% instead of 35.1% average efficiency.

## Reproduce

```
pip install -r requirements.txt
python experiments/run_all.py        # a few minutes, writes results/results.json
python experiments/make_figures.py   # figures/*.png
python build.py                      # papr-reduction-report.html
python -m unittest discover -s tests -v
```

Every number on the page is filled in from `results/results.json` by `build.py`, which
refuses to build if a value is missing.

## Contents

```
papr-reduction-report.html   standalone copy of the page, with the lab inlined
paprlab/                     OFDM model, PAPR statistics, transmitter methods, amplifier,
                             BELLHOP channel, neural network, FFB receiver
experiments/                 run_all.py (every experiment) and make_figures.py
tests/                       22 unit tests
channel/                     BELLHOP input and arrivals output (see channel/README.md)
lab/                         the browser lab: papr-lab.js and papr-lab.css
report/                      page content, styles and shell filled in by build.py
                             (site-style.css is the waleedraza.dev stylesheet)
results/results.json         the experiment output every number comes from
figures/                     generated figures
```

## Not included

- BELLHOP (Michael B. Porter's Acoustics Toolbox): cited, not redistributed.
- The thesis document itself: summarised on the page, not published here.

## Related reports in this portfolio

- [Underwater Acoustic Transducer & Hydrophone Systems](https://github.com/raza-waleed/acoustic-measurement-report)
- [Underwater Acoustic OFDM/QPSK Communication](https://github.com/raza-waleed/underwater-acoustic-ofdm-lake-trial)
- [Digital Signal Processing for Underwater Acoustic Channels](https://github.com/raza-waleed/dsp-underwater-acoustics-report)
- [Ray-Tracing Simulation of Underwater Acoustic Channels](https://github.com/raza-waleed/bellhop-acoustic-modeling-report)
