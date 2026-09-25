"""OFDM model, PAPR statistics and the transmitter side methods.

Run from the repository root:  python -m unittest discover -s tests -v
"""
import itertools
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paprlab import ofdm, papr, transmitter as tx  # noqa: E402


class OfdmModel(unittest.TestCase):
    def test_qam_round_trip_and_unit_power(self):
        for order in (4, 16, 64):
            m = int(np.log2(order))
            every = np.array(list(itertools.product([0, 1], repeat=m)))
            points = ofdm.modulate(every.reshape(1, -1), order).ravel()
            self.assertAlmostEqual(np.mean(np.abs(points) ** 2), 1.0, places=12)
            self.assertTrue(np.array_equal(ofdm.demodulate(points, order), every.reshape(-1)))

    def test_gray_neighbours_differ_by_one_bit(self):
        every = np.array(list(itertools.product([0, 1], repeat=4)))
        points = ofdm.modulate(every.reshape(1, -1), 16).ravel()
        step = 2 / np.sqrt(10)
        for i, j in itertools.combinations(range(16), 2):
            if abs(abs(points[i] - points[j]) - step) < 1e-9:
                self.assertEqual(int(np.sum(every[i] != every[j])), 1)

    def test_time_frequency_round_trip_and_power(self):
        rng = np.random.default_rng(0)
        X, _ = ofdm.random_symbols(rng, 50, 128, 4)
        for L in (1, 2, 4, 1.5):
            x = ofdm.to_time(X, L)
            self.assertEqual(x.shape[1], int(round(128 * L)))
            self.assertAlmostEqual(np.mean(np.abs(x) ** 2), 1.0, places=10)
            self.assertLess(np.max(np.abs(ofdm.to_freq(x, 128) - X)), 1e-12)
            self.assertLess(ofdm.out_of_band_power(x, 128), 1e-25)

    def test_subcarrier_frequency_order_matches_fft(self):
        f = ofdm.subcarrier_offsets(8, 1.0)
        self.assertEqual(f.tolist(), [0, 1, 2, 3, -4, -3, -2, -1])


class PaprStatistics(unittest.TestCase):
    def test_papr_extremes(self):
        n = 64
        self.assertAlmostEqual(float(papr.papr_db(ofdm.to_time(np.ones((1, n)), 4))[0]), 10 * np.log10(n), places=9)
        tone = np.zeros((1, n)); tone[0, 3] = 1.0
        self.assertAlmostEqual(float(papr.papr_db(ofdm.to_time(tone, 4))[0]), 0.0, places=9)

    def test_nyquist_rate_simulation_matches_theory(self):
        rng = np.random.default_rng(1)
        X, _ = ofdm.random_symbols(rng, 20000, 64, 4)
        sim = papr.papr_at(papr.papr_db(ofdm.to_time(X, 1)), 1e-2)
        grid = np.arange(4, 14, 0.001)
        th = grid[np.argmin(np.abs(papr.ccdf_theory(grid, 64) - 1e-2))]
        self.assertLess(abs(sim - th), 0.3)

    def test_efficiency_rule(self):
        self.assertAlmostEqual(float(papr.class_b_efficiency(0.0)), np.pi / 4)
        self.assertAlmostEqual(float(papr.class_a_efficiency(0.0)), 0.5)


class TransmitterMethods(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(2)

    def test_clipper_limits_amplitude_and_keeps_phase(self):
        X, _ = ofdm.random_symbols(self.rng, 200, 128, 4)
        x = ofdm.to_time(X, 2)
        y = tx.clip(x, 3.0)
        limit = np.sqrt(3.0 * np.mean(np.abs(x) ** 2, axis=1, keepdims=True))
        self.assertTrue(np.all(np.abs(y) <= limit * (1 + 1e-12)))
        moved = np.abs(y - x) > 1e-12
        self.assertLess(np.max(np.abs(np.angle(y[moved] / x[moved]))), 1e-9)

    def test_filter_removes_out_of_band_only(self):
        X, _ = ofdm.random_symbols(self.rng, 20, 128, 4)
        y = tx.clip(ofdm.to_time(X, 2), 2.0)
        z = tx.band_limit(y, 128)
        self.assertLess(ofdm.out_of_band_power(z, 128), 1e-25)
        self.assertLess(np.max(np.abs(ofdm.to_freq(z, 128) - ofdm.to_freq(y, 128))), 1e-12)

    def test_rcf_thesis_setting(self):
        """128 subcarriers, I = 2, CR = 3, CCDF 1e-3: 10.97 dB, then 7.86 / 6.59 / 5.83 / 5.39 dB on the page."""
        X, _ = ofdm.random_symbols(np.random.default_rng(21), 20000, 128, 4)
        x = ofdm.to_time(X, 2)
        _, hist = tx.repeated_clipping_filtering(x, 128, 3.0, 4)
        self.assertLess(abs(papr.papr_at(papr.papr_db(x)) - 10.97), 0.25)
        values = [papr.papr_at(h) for h in hist]
        for got, page in zip(values, [7.86, 6.59, 5.83, 5.39]):
            self.assertLess(abs(got - page), 0.2)
        self.assertEqual(values, sorted(values, reverse=True))   # every pass lowers the PAPR

    def test_rooting_companding_thesis_setting(self):
        """Nyquist rate with the cyclic prefix, R = 0.5: 10.51 dB to 5.85 dB on the page."""
        X, _ = ofdm.random_symbols(np.random.default_rng(22), 20000, 128, 4)
        x = ofdm.add_cyclic_prefix(ofdm.to_time(X, 1), 32)
        self.assertLess(abs(papr.papr_at(papr.papr_db(tx.root_compand(x, 0.5))) - 5.85), 0.2)
        self.assertLess(abs(papr.papr_at(papr.papr_db(x)) - 10.51), 0.3)

    def test_rooting_inverse_is_exact(self):
        X, _ = ofdm.random_symbols(self.rng, 10, 64, 16)
        x = ofdm.to_time(X, 4)
        for R in (0.3, 0.5, 0.9):
            self.assertLess(np.max(np.abs(tx.root_expand(tx.root_compand(x, R), R) - x)), 1e-10)

    def test_slm_picks_the_best_candidate_and_matches_theory(self):
        X, _ = ofdm.random_symbols(self.rng, 5, 64, 4)
        phases = tx.slm_phase_sequences(self.rng, 4, 64)
        _, idx, p = tx.selected_mapping(X, phases, 4)
        brute = papr.papr_db(ofdm.to_time(X[:, None, :] * phases[None], 4))
        self.assertTrue(np.array_equal(idx, np.argmin(brute, axis=1)))
        self.assertTrue(np.allclose(p, brute.min(axis=1)))
        X, _ = ofdm.random_symbols(np.random.default_rng(23), 10000, 256, 4)
        phases = tx.slm_phase_sequences(np.random.default_rng(24), 4, 256)
        sim = papr.papr_at(tx.selected_mapping(X, phases, 1)[2], 1e-2)
        grid = np.arange(4, 14, 0.001)
        th = grid[np.argmin(np.abs(papr.slm_ccdf_theory(grid, 256, 4) - 1e-2))]
        self.assertLess(abs(sim - th), 0.3)

    def test_pts_exhaustive_search(self):
        X, _ = ofdm.random_symbols(self.rng, 6, 64, 4)
        sig, idx, p = tx.partial_transmit_sequences(X, 4, (1, -1, 1j, -1j), 2)
        weights = tx.pts_phase_combinations(4, (1, -1, 1j, -1j))
        self.assertEqual(weights.shape, (64, 4))
        width = 16
        for s in range(6):
            best = min(
                papr.papr_db(ofdm.to_time(np.concatenate([w[v] * X[s, v * width:(v + 1) * width] for v in range(4)])[None], 2))[0]
                for w in weights)
            self.assertAlmostEqual(float(p[s]), float(best), places=9)
        # the chosen signal carries the same data up to the known block weights
        rebuilt = ofdm.to_freq(sig, 64)
        for s in range(6):
            w = np.repeat(weights[idx[s]], width)
            self.assertLess(np.max(np.abs(rebuilt[s] / w - X[s])), 1e-12)


if __name__ == "__main__":
    unittest.main()
