# ----------------------------
# Synth Engine
# ----------------------------
import numpy as np

# globals
SR = 44100
BLOCK = 512

# JS divergence is bounded by ln 2; used to normalize mismatch to [0, 1].
JS_MAX = np.log(2.0)

# --- wavetable configuration ---
TABLE_SIZE = 2048      # samples per single-cycle wavetable
N_HARMONICS = 64       # harmonics in the band-limited sawtooth end of the morph
MORPH_STEPS = 33       # tables in the sine->saw bank (index 0 = sine, -1 = saw)
BASE_FREQ = 27.5       # lowest pitch the mip-map covers (A0)
N_OCTAVES = 10         # octaves of mip levels: 27.5 Hz -> ~28 kHz


class Synth:
    def __init__(self, table_size=TABLE_SIZE, n_harmonics=N_HARMONICS,
                 morph_steps=MORPH_STEPS, base_freq=BASE_FREQ,
                 n_octaves=N_OCTAVES):
        # Per-frequency running phase (radians). Carrying phase per oscillator
        # means a sustained tone continues smoothly across callback blocks
        # instead of restarting at phase 0 each block (which caused clicks).
        self.phases = {}
        # Phase for the modulation LFO, likewise carried across blocks.
        self.mod_phase = 0.0

        # --- wavetable state (purely additive; nothing above is changed) ---
        self.table_size = table_size
        self.n_harmonics = n_harmonics
        self.morph_steps = morph_steps
        self.base_freq = base_freq
        self.n_octaves = n_octaves
        # Wavetable read positions, in *table samples*, carried across blocks
        # for the same reason self.phases is: continuity, no clicks.
        self.wt_phases = {}
        # Precomputed sine->saw morph bank. Built once; reading it per block is
        # a table lookup, not a rebuild (rebuilding per block is far too slow).
        self._morph_bank = self._build_morph_bank()
        # Cache of tables derived from distributions, keyed by content hash.
        # Distributions change once per training step, not once per audio block,
        # so this hits on every block within a step.
        self._dist_table_cache = {}

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

    # --- wavetable engine -----------------------------------------------

    def _build_one_bank(self, max_harmonics):
        """Build a sine->saw morph bank capped at max_harmonics.

        A sine is harmonic 1 alone; a sawtooth is every harmonic at amplitude
        1/n. So the morph keeps harmonic 1 fixed and fades in harmonics 2..N
        scaled by w:

            table(w)[k] = sin(2*pi*k/N) + w * sum_{n=2..N} (1/n) sin(2*pi*n*k/N)

        At w=0 this is exactly a sine; at w=1 a band-limited sawtooth. Capping
        the harmonic count band-limits the table by construction -- unlike a
        naive 2*(t%1)-1 saw, which aliases badly.
        Each table is peak-normalized so the morph changes timbre, not loudness
        (loudness is the other, independent axis).
        """
        N = self.table_size
        k = np.arange(N)
        base = np.sin(2 * np.pi * k / N)                      # harmonic 1
        upper = np.zeros(N)
        for n in range(2, max(2, max_harmonics) + 1):
            upper += np.sin(2 * np.pi * n * k / N) / n        # saw's tail

        bank = np.zeros((self.morph_steps, N))
        for i in range(self.morph_steps):
            w = i / (self.morph_steps - 1)
            tbl = base + w * upper
            peak = np.max(np.abs(tbl))
            bank[i] = tbl / peak if peak > 0 else tbl
        return bank

    def _build_morph_bank(self):
        """Build a *mip-mapped* set of morph banks, one per octave.

        A single band-limited table is only band-limited at the pitch it was
        designed for. Play a 64-harmonic table at 3 kHz and harmonics above
        Nyquist fold back as audible alias tones. The standard fix is mip-
        mapping: each octave gets its own bank whose harmonic count is capped
        so that even the top harmonic stays below Nyquist. High notes therefore
        use fewer harmonics (and sound correspondingly purer -- which is
        physically right, not a compromise).
        """
        banks = []
        for octave in range(self.n_octaves):
            f_top = self.base_freq * (2 ** (octave + 1))      # top of this octave
            max_h = int((SR / 2) / max(f_top, 1e-9))          # keep under Nyquist
            max_h = max(1, min(self.n_harmonics, max_h))
            banks.append(self._build_one_bank(max_h))
        return banks

    def _bank_for_freq(self, freq):
        """Pick the mip level whose harmonic cap suits this pitch."""
        freq = max(float(freq), 1e-9)
        octave = int(np.floor(np.log2(freq / self.base_freq))) if freq > 0 else 0
        octave = int(np.clip(octave, 0, self.n_octaves - 1))
        return self._morph_bank[octave]

    def _morph_table(self, w, freq=None):
        """Interpolate between adjacent tables in the bank for continuous w.

        freq selects the mip level; omitting it uses the lowest (most
        harmonically rich) bank, which is correct for low fundamentals.
        """
        bank = self._bank_for_freq(freq) if freq is not None else self._morph_bank[0]
        w = float(np.clip(w, 0.0, 1.0))
        pos = w * (self.morph_steps - 1)
        i0 = int(np.floor(pos))
        i1 = min(i0 + 1, self.morph_steps - 1)
        frac = pos - i0
        return (1 - frac) * bank[i0] + frac * bank[i1]

    def _read_table(self, table, freq, amp, frames, voice):
        """Phase-continuous wavetable playback with linear interpolation.

        Mirrors the phase-carrying contract of sine(): the read position is
        stored per voice and resumed on the next block, so sustained tones do
        not click at block boundaries.
        """
        N = self.table_size
        step = freq * N / SR                       # table samples per audio sample
        ph0 = self.wt_phases.get(voice, 0.0)
        idx = ph0 + step * np.arange(frames)
        self.wt_phases[voice] = (ph0 + step * frames) % N

        idx = np.mod(idx, N)
        i0 = np.floor(idx).astype(int)
        i1 = (i0 + 1) % N
        frac = idx - i0
        return amp * ((1 - frac) * table[i0] + frac * table[i1])

    def wavetable(self, freq, amp, w, frames, voice=0):
        """One oscillator reading the sine->saw morph bank at position w.

        w = 0 -> pure sine; w = 1 -> band-limited sawtooth; in between is a
        real, continuous timbral position.

        The mip level is chosen from freq, so high pitches automatically use
        fewer harmonics and do not alias.
        """
        return self._read_table(self._morph_table(w, freq), freq, amp,
                                frames, voice)

    def loss_to_sound(self, loss, frames, base_freq=220.0, loss_scale=1.0,
                      voice="loss"):
        """A scalar loss -> audio. The headline mapping:

            low loss  -> quiet, pure sine
            high loss -> loud, sawtooth

        Volume and timbre are two independent axes driven by the same scalar.
        A raw loss is unbounded, so it is squashed with tanh into [0, 1);
        loss_scale sets what counts as "high" (see WavetableTracker, which
        calibrates this against a running max instead).
        """
        x = float(np.clip(loss, 0.0, None))
        level = float(np.tanh(x / max(loss_scale, 1e-9)))   # bounded [0, 1)
        amp = 0.05 + 0.75 * level                            # audible at zero loss
        out = self.wavetable(base_freq, amp, level, frames, voice=voice)
        return np.tanh(out)

    def _table_from_distribution(self, dist, fundamental=220.0):
        """Build a single-cycle table whose harmonic amplitudes ARE the
        distribution -- element i becomes harmonic i+1, weight becomes its
        amplitude. This is MagL's existing "distribution as spectrum" idea,
        rendered as a wavetable rather than a bank of live oscillators.

        Harmonics above Nyquist for the fundamental are dropped, so high-index
        elements cannot alias. Cached by content: distributions change per
        training step, not per audio block.
        """
        dist = self.normalize(dist)
        key = (round(fundamental, 3), dist.tobytes())
        cached = self._dist_table_cache.get(key)
        if cached is not None:
            return cached

        N = self.table_size
        k = np.arange(N)
        table = np.zeros(N)
        max_h = int((SR / 2) / max(fundamental, 1e-9))       # band limit
        for i, w in enumerate(dist):
            n = i + 1
            if n > max_h or w <= 0:
                continue
            table += w * np.sin(2 * np.pi * n * k / N)
        peak = np.max(np.abs(table))
        if peak > 0:
            table = table / peak
        self._dist_table_cache[key] = table
        return table

    def xent_sound(self, p, q, frames, mode="classic"):
        """Sonify the mismatch between two distributions.

        mode="classic"   -> the original sound: additive spectrum of P, with
                            modulation and noise scaled by the mismatch.
        mode="wavetable" -> the timbre of P is morphed *toward* the timbre of Q
                            by the mismatch itself (see _xent_sound_wavetable).

        Both are driven by the same bounded JS divergence and both fall silent
        at P == Q. "classic" is the default so existing behaviour is unchanged.
        """
        if mode == "wavetable":
            return self._xent_sound_wavetable(p, q, frames)
        return self._xent_sound_classic(p, q, frames)

    def _xent_sound_wavetable(self, p, q, frames, fundamental=220.0,
                              voice="xent"):
        """Wavetable xent: the mismatch IS the distance the timbre travels.

        P and Q are each rendered as a wavetable (their weights are the harmonic
        amplitudes), and the normalized JS divergence d sets how far P's table
        is morphed toward Q's:

            table = (1-d) * table_P + d * table_Q

        At d = 0 the morph does not move at all -- you hear P, untouched. That
        makes "silence at convergence" structural rather than a scalar dialled
        to zero: identical distributions are not merely unmodulated, they are
        not deformed. As d rises you hear one structure literally pulled toward
        the other, which is what "emit xent P Q" means.

        The modulation/noise instability layer from the classic mode is kept,
        subordinate to the morph, so d drives three coherent things at once:
        how far the timbre travels, how fast it modulates, and how noisy it is.
        """
        d = self.js_divergence(p, q) / JS_MAX     # bounded [0, 1], 0 iff p == q

        tbl_p = self._table_from_distribution(p, fundamental)
        tbl_q = self._table_from_distribution(q, fundamental)
        table = (1 - d) * tbl_p + d * tbl_q       # the structural core

        base = self._read_table(table, fundamental, 0.6, frames, voice=voice)

        # instability layer (same shape as classic; vanishes at d == 0)
        mod_freq = 2 + d * 18
        dphi = 2 * np.pi * mod_freq / SR
        mod_phase = self.mod_phase + dphi * np.arange(frames)
        self.mod_phase = (self.mod_phase + dphi * frames) % (2 * np.pi)
        mod = np.sin(mod_phase)
        noise = np.random.randn(frames) * (d * 0.25)

        out = base * (1 + d * 0.5 * mod) + noise
        return np.tanh(out)

    def _xent_sound_classic(self, p, q, frames):
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
