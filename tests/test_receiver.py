"""BELLHOP channel, amplifier, neural network and the FFB receiver."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paprlab import amplifier as amp, channel as chn, neural, ofdm, receiver as rx  # noqa: E402


class BellhopChannel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = chn.read_arrivals(str(ROOT / "channel" / "thesis_2km.arr"))

    def test_arrivals_parse_for_the_table_4_2_geometry(self):
        d = self.d
        self.assertEqual((d["sd"], d["rd"], d["r"], d["freq"]), ([30.0], [50.0], [2000.0], 11000.0))
        self.assertEqual(len(d["arrivals"]), 23)
        delays = [a["delay_s"] for a in d["arrivals"]]
        self.assertEqual(delays, sorted(delays))

    def test_first_arrival_matches_straight_line_travel_time(self):
        z = np.linspace(30, 50, 2001)
        c = np.interp(z, [30, 40, 50], [1492.8, 1496.8, 1498.4])
        straight = np.hypot(2000.0, 20.0) * np.mean(1.0 / c)
        self.assertLess(abs(self.d["arrivals"][0]["delay_s"] - straight) / straight, 1e-3)

    def test_response_helpers(self):
        one = [{"amp": 2.0, "phase_deg": 30.0, "delay_s": 1.0}]
        H = chn.frequency_response(one, np.linspace(0, 100, 11))
        self.assertTrue(np.allclose(np.abs(H), 2.0))
        Hn = chn.normalized(chn.frequency_response(self.d["arrivals"], 11000 + ofdm.subcarrier_offsets(512, 6250 / 512)))
        self.assertAlmostEqual(float(np.mean(np.abs(Hn) ** 2)), 1.0, places=12)
        self.assertAlmostEqual(chn.energy_after(self.d["arrivals"], 10.0), 0.0)


class AmplifierAndNetwork(unittest.TestCase):
    def test_rapp_curve(self):
        T = amp.NonlinearTransmitter(ibo_db=5.0)
        a_sat = 10 ** (5 / 20)
        self.assertAlmostEqual(float(T.amplitude(a_sat)), a_sat * 2 ** -0.25, places=12)   # p = 2
        self.assertAlmostEqual(float(T.amplitude(1e-3)), 1e-3, places=9)                  # unit small signal gain
        self.assertLess(float(T.amplitude(50.0)), a_sat * (1 + 1e-6))                     # saturates

    def test_bussgang_gain(self):
        rng = np.random.default_rng(0)
        X, _ = ofdm.random_symbols(rng, 50, 256, 4)
        x = ofdm.to_time(X, 4)
        self.assertAlmostEqual(amp.bussgang_gain(x, 2.0 * x), 2.0, places=12)
        y = amp.NonlinearTransmitter(ibo_db=3.0)(x)
        a = amp.bussgang_gain(x, y)
        self.assertLess(abs(np.vdot(x, y - a * x)) / np.vdot(x, x).real, 1e-12)   # distortion uncorrelated with x

    def test_network_learns_the_amplifier(self):
        rng = np.random.default_rng(1)
        T = amp.NonlinearTransmitter(ibo_db=5.0)
        X, _ = ofdm.random_symbols(rng, 13, 512, 4)
        x = ofdm.to_time(X, 4).ravel()[:25800]
        net = neural.AmplifierNetwork(seed=12).fit(np.abs(x).reshape(-1, 6), np.abs(T(x)).reshape(-1, 6), seed=12)
        grid = np.linspace(0, 3.0, 61)
        self.assertLess(np.max(np.abs(net.predict_amplitude(grid) - T.amplitude(grid))), 0.06)
        self.assertLess(net.history["validation_mse"], 5e-4)
        self.assertEqual(net.n_params, 162)                     # 6 -> 12 -> 6, as in the thesis


class FfbReceiver(unittest.TestCase):
    def test_ffb_removes_noiseless_distortion_errors(self):
        rng = np.random.default_rng(3)
        T = amp.NonlinearTransmitter(ibo_db=3.0)
        X, bits = ofdm.random_symbols(rng, 20, 512, 64)
        y, Y = rx.transmit(X, 4, T)
        alpha = amp.bussgang_gain(ofdm.to_time(X, 4), y)
        H = np.ones(512, dtype=complex)
        dec = rx.ffb_detect(Y, H, 64, T, alpha, 4, iterations=3)
        errors = [int(np.sum(ofdm.demodulate(d, 64) != bits)) for d in dec]
        self.assertGreater(errors[0], 50)      # distortion alone breaks 64-QAM at 3 dB back-off
        self.assertEqual(errors[-1], 0)        # FFB with the true model removes every error

    def test_ffb_with_the_channel_undoes_equalisation_correctly(self):
        rng = np.random.default_rng(4)
        d = chn.read_arrivals(str(ROOT / "channel" / "thesis_2km.arr"))
        H = chn.normalized(chn.frequency_response(d["arrivals"], 11000 + ofdm.subcarrier_offsets(512, 6250 / 512)))
        T = amp.NonlinearTransmitter(ibo_db=5.0)
        X, bits = ofdm.random_symbols(rng, 10, 512, 16)
        y, Ytx = rx.transmit(X, 4, T)
        alpha = amp.bussgang_gain(ofdm.to_time(X, 4), y)
        dec = rx.ffb_detect(H * Ytx, H, 16, T, alpha, 4, iterations=2)   # noiseless: the channel must not matter
        self.assertEqual(int(np.sum(ofdm.demodulate(dec[-1], 16) != bits)), 0)


if __name__ == "__main__":
    unittest.main()
