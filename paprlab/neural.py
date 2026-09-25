"""The small neural network my thesis used to learn the power amplifier.

Architecture (thesis section 4.3.2): the time signal's amplitudes are reshaped
into vectors of 6 consecutive samples; one hidden layer of 12 sigmoid neurons
maps them to 6 output amplitudes (162 weights). Training uses the
Levenberg-Marquardt algorithm with a random 70/15/15 split and validation
based early stopping, the same recipe as MATLAB's trainlm that I used in 2020.
Written from scratch with numpy only.
"""
import time

import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class AmplifierNetwork:
    def __init__(self, inputs=6, hidden=12, seed=0):
        self.inputs, self.hidden = inputs, hidden
        rng = np.random.default_rng(seed)
        # Nguyen-Widrow style start: spread the hidden units across the input range
        w = rng.uniform(-1.0, 1.0, size=(hidden, inputs))
        beta = 0.7 * hidden ** (1.0 / inputs)
        self.w1 = beta * w / np.linalg.norm(w, axis=1, keepdims=True)
        self.b1 = beta * rng.uniform(-1.0, 1.0, size=hidden)
        self.w2 = rng.uniform(-0.5, 0.5, size=(inputs, hidden))
        self.b2 = np.zeros(inputs)
        self.in_range = (0.0, 1.0)
        self.out_range = (0.0, 1.0)
        self.history = {}

    # ------------------------------------------------------------ parameters
    @property
    def n_params(self):
        return self.w1.size + self.b1.size + self.w2.size + self.b2.size

    def _get(self):
        return np.concatenate([self.w1.ravel(), self.b1, self.w2.ravel(), self.b2])

    def _set(self, p):
        h, i = self.hidden, self.inputs
        k = 0
        self.w1 = p[k:k + h * i].reshape(h, i); k += h * i
        self.b1 = p[k:k + h]; k += h
        self.w2 = p[k:k + i * h].reshape(i, h); k += i * h
        self.b2 = p[k:k + i]

    # ------------------------------------------------------------ scaling (mapminmax)
    @staticmethod
    def _scale(v, rng_):
        lo, hi = rng_
        return 2.0 * (v - lo) / (hi - lo) - 1.0

    @staticmethod
    def _unscale(v, rng_):
        lo, hi = rng_
        return (v + 1.0) * (hi - lo) / 2.0 + lo

    # ------------------------------------------------------------ forward pass
    def _forward_scaled(self, xs):
        s = _sigmoid(xs @ self.w1.T + self.b1)
        return s @ self.w2.T + self.b2, s

    def predict_vectors(self, x):
        y, _ = self._forward_scaled(self._scale(x, self.in_range))
        return self._unscale(y, self.out_range)

    def predict_amplitude(self, a):
        """Amplitude in, amplitude out, for an array of any shape."""
        flat = np.ravel(a)
        pad = (-flat.size) % self.inputs
        padded = np.concatenate([flat, np.full(pad, flat[-1] if flat.size else 0.0)])
        out = self.predict_vectors(padded.reshape(-1, self.inputs)).ravel()[: flat.size]
        return out.reshape(np.shape(a))

    # ------------------------------------------------------------ training
    def _jacobian(self, xs):
        n = xs.shape[0]
        y, s = self._forward_scaled(xs)
        sp = s * (1.0 - s)
        o, h, i = self.inputs, self.hidden, self.inputs
        j_w1 = (self.w2[None, :, :, None] * sp[:, None, :, None] * xs[:, None, None, :]).reshape(n, o, h * i)
        j_b1 = self.w2[None, :, :] * sp[:, None, :]
        eye = np.eye(o)
        j_w2 = (eye[None, :, :, None] * s[:, None, None, :]).reshape(n, o, o * h)
        j_b2 = np.broadcast_to(eye, (n, o, o))
        return y, np.concatenate([j_w1, j_b1, j_w2, j_b2], axis=2).reshape(n * o, -1)

    def fit(self, x, t, seed=0, epochs=1000, mu=1e-3, mu_dec=0.1, mu_inc=10.0,
            mu_max=1e10, max_fail=6, min_grad=1e-7):
        """Levenberg-Marquardt on vectors x -> t (both shape (n, inputs))."""
        rng = np.random.default_rng(seed)
        order = rng.permutation(x.shape[0])
        n_train = int(round(0.70 * len(order)))
        n_val = int(round(0.15 * len(order)))
        tr, va, te = order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:]

        self.in_range = (float(x[tr].min()), float(x[tr].max()))
        self.out_range = (float(t[tr].min()), float(t[tr].max()))
        xs = self._scale(x, self.in_range)
        ts = self._scale(t, self.out_range)

        def mse_scaled(idx):
            y, _ = self._forward_scaled(xs[idx])
            return float(np.mean((y - ts[idx]) ** 2))

        def mse(idx):  # in amplitude units, like the thesis report
            return float(np.mean((self.predict_vectors(x[idx]) - t[idx]) ** 2))

        start = time.perf_counter()
        best = (np.inf, self._get(), 0)
        fails, epoch = 0, 0
        train_curve, val_curve = [mse(tr)], [mse(va)]
        stop = "max epochs"
        for epoch in range(1, epochs + 1):
            y, J = self._jacobian(xs[tr])
            e = (y - ts[tr]).ravel()
            sse = float(e @ e)
            g = J.T @ e
            if np.max(np.abs(g)) < min_grad:
                stop = "minimum gradient"
                break
            A = J.T @ J
            p = self._get()
            improved = False
            while mu <= mu_max:
                step = np.linalg.solve(A + mu * np.eye(A.shape[0]), -g)
                self._set(p + step)
                y_new, _ = self._forward_scaled(xs[tr])
                if float(np.sum((y_new - ts[tr]) ** 2)) < sse:
                    mu *= mu_dec
                    improved = True
                    break
                mu *= mu_inc
            if not improved:
                self._set(p)
                stop = "mu reached its maximum"
                break
            train_curve.append(mse(tr))
            v = mse(va)
            val_curve.append(v)
            if v < best[0]:
                best, fails = (v, self._get(), epoch), 0
            else:
                fails += 1
                if fails >= max_fail:
                    stop = "validation stopped improving"
                    break
        self._set(best[1])
        self.history = {
            "epochs_run": epoch, "best_epoch": best[2], "stop_reason": stop,
            "seconds": time.perf_counter() - start,
            "train_mse": mse(tr), "validation_mse": mse(va), "test_mse": mse(te),
            "train_curve": train_curve, "validation_curve": val_curve,
            "samples": int(x.size), "vectors": int(x.shape[0]),
            "split": [int(tr.size), int(va.size), int(te.size)],
        }
        return self

    # ------------------------------------------------------------ export
    def to_dict(self):
        return {
            "inputs": self.inputs, "hidden": self.hidden,
            "w1": self.w1.round(10).tolist(), "b1": self.b1.round(10).tolist(),
            "w2": self.w2.round(10).tolist(), "b2": self.b2.round(10).tolist(),
            "in_range": list(self.in_range), "out_range": list(self.out_range),
        }


class NeuralAmplifierModel:
    """Receiver side stand-in for the transmitter: network for the amplitude,
    phase passed through (the SSPA has no AM/PM distortion)."""

    def __init__(self, network):
        self.network = network

    def __call__(self, x):
        return self.network.predict_amplitude(np.abs(x)) * np.exp(1j * np.angle(x))
