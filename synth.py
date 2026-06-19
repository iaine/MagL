# ----------------------------
# Synth Engine
# ----------------------------
import numpy as np

# globals
SR = 44100
BLOCK = 512

class Synth:
    def __init__(self):
        self.phase = 0.0
        self.t = 0

    def normalize(self, arr):
        arr = np.array(arr, dtype=float)
        s = np.sum(arr)
        return arr / s if s > 0 else arr

    def cross_entropy(self, p, q):
        p = self.normalize(p)
        q = self.normalize(q)
        eps = 1e-9
        return -np.sum(p * np.log(q + eps))

    def sine(self, freq, amp, frames):
        t = (np.arange(frames) + self.t) / SR
        return amp * np.sin(2 * np.pi * freq * t)

    def distribution_sound(self, dist, frames):
        dist = self.normalize(dist)
        signal = np.zeros(frames)

        for i, w in enumerate(dist):
            freq = 220 * (i + 1)
            signal += self.sine(freq, w, frames)

        return signal

    def xent_sound(self, p, q, frames):
        H = self.cross_entropy(p, q)

        # Map entropy → instability
        base = self.distribution_sound(p, frames)

        # Modulation increases with entropy
        mod_freq = 2 + H * 10
        mod = np.sin(2 * np.pi * mod_freq * np.arange(frames) / SR)

        # Noise component
        noise = np.random.randn(frames) * (H * 0.1)

        return base * (1 + 0.5 * mod) + noise