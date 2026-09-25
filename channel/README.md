# BELLHOP channel for thesis Table 4.2

- `thesis_2km.env` is my BELLHOP input for the chapter 4 geometry in my thesis:
  100 m of water, transmitter at 30 m, hydrophone at 50 m, 2 km apart, 11 kHz (the
  centre of the 10 to 12 kHz band), the sound speed profile from my thesis
  (1482 m/s at the surface rising to 1498.4 m/s at 50 m, held below that) and a
  sandy bottom (1600 m/s, 1.8 g/cm3, 0.8 dB per wavelength).
- `thesis_2km.arr` is BELLHOP's ASCII arrivals output for that input: 23 multipath
  arrivals with amplitude, phase, delay, launch angle and bounce counts.
  `paprlab/channel.py` reads it and turns it into the channel frequency response.

The first arrival lands 0.048% after the straight line travel time at the
harmonic mean sound speed, the same kind of geometric check my BELLHOP report
uses (github.com/raza-waleed/bellhop-acoustic-modeling-report).

To regenerate the arrivals, put `bellhop.exe` from Michael B. Porter's Acoustics
Toolbox (oalib-acoustics.org) in this folder and run `bellhop.exe thesis_2km`.
BELLHOP is not included in this repository.
