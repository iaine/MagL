# ----------------------------
# Synth Engine
# ----------------------------
import numpy as np

# globals
SR = 44100
BLOCK = 512

# JS divergence is bounded by ln 2; used to normalize mismatch to [0, 1].
JS_MAX = np.log(2.0)


class Synth:
    def __init__(self):
        # Per-frequency running phase (radians). Carrying phase per oscillator
        # means a sustained tone continues smoothly across callback blocks
        # instead of restarting at phase 0 each block (which caused clicks).
        self.phases = {}
        # Phase for the modulation LFO, likewise carried across blocks.
        self.mod_phase = 0.0

    def normalize(self, arr):
        arr = np.array(arr, dtype=float)
        s = np.sum(arr)
        return arr / s if s > 0 else arr

    # --- distribution comparison measures -------------------------------

    def cross_entropy(self, p, q):
        """H(P, Q) = -Σ P·log Q. Kept for reference; note it does NOT vanish
        when P == Q (it equals the entropy of P) and is asymmetric."""
        p = self.normalize(p)
        q = self.normalize(q)
        eps = 1e-9
        return -np.sum(p * np.log(q + eps))

    def kl_divergence(self, p, q):
        """KL(P||Q) = Σ P·log(P/Q). Zero iff P == Q; asymmetric; unbounded."""
        p = self.normalize(p)
        q = self.normalize(q)
        eps = 1e-9
        return float(np.sum(p * np.log((p + eps) / (q + eps))))

    def js_divergence(self, p, q):
        """Jensen–Shannon divergence: symmetric, bounded in [0, ln 2], and zero
        iff P == Q. This is the mismatch driver -- identical distributions give
        0 (the sound resolves to consonance), maximal disagreement gives ln 2.
        Lengths are reconciled by zero-padding the shorter to the longer."""
        p = np.array(p, dtype=float)
        q = np.array(q, dtype=float)
        if p.shape != q.shape:                 # tolerate differing lengths
            n = max(len(p), len(q))
            p = np.pad(p, (0, n - len(p)))
            q = np.pad(q, (0, n - len(q)))
        p = self.normalize(p)
        q = self.normalize(q)
        m = 0.5 * (p + q)
        eps = 1e-9
        kl = lambda a, b: np.sum(a * np.log((a + eps) / (b + eps)))
        return float(0.5 * kl(p, m) + 0.5 * kl(q, m))

    # --- oscillators ----------------------------------------------------

    def sine(self, freq, amp, frames):
        # advance from this oscillator's stored phase, then store the end phase
        ph0 = self.phases.get(freq, 0.0)
        dphi = 2 * np.pi * freq / SR
        phase = ph0 + dphi * np.arange(frames)
        out = amp * np.sin(phase)
        # next block resumes exactly where this one ended (mod 2π)
        self.phases[freq] = (ph0 + dphi * frames) % (2 * np.pi)
        return out

    def distribution_sound(self, dist, frames):
        dist = self.normalize(dist)
        signal = np.zeros(frames)

        for i, w in enumerate(dist):
            freq = 220 * (i + 1)
            signal += self.sine(freq, w, frames)

        return signal

    def xent_sound(self, p, q, frames):
        # Mismatch driver is Jensen–Shannon divergence: symmetric, bounded, and
        # zero when p == q -> identical distributions resolve to a clean,
        # unmodulated tone (consonance / "silence at convergence").
        D = self.js_divergence(p, q)
        d = D / JS_MAX                          # normalized mismatch in [0, 1]

        base = self.distribution_sound(p, frames)

        # Modulation rate and depth rise with mismatch; at d == 0 there is no
        # modulation at all. The LFO phase is carried across blocks too.
        mod_freq = 2 + d * 18
        dphi = 2 * np.pi * mod_freq / SR
        mod_phase = self.mod_phase + dphi * np.arange(frames)
        self.mod_phase = (self.mod_phase + dphi * frames) % (2 * np.pi)
        mod = np.sin(mod_phase)

        # Noise scales with mismatch and vanishes at d == 0.
        noise = np.random.randn(frames) * (d * 0.25)

        out = base * (1 + d * 0.5 * mod) + noise
        # Soft limiter: JS is bounded so this can't run away, but tanh keeps the
        # output within [-1, 1] and prevents hard clipping on dense spectra.
        return np.tanh(out)
