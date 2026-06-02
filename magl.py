import numpy as np
import sounddevice as sd
import time

# ----------------------------
# Global Config
# ----------------------------
SR = 44100
BLOCK = 512

# ----------------------------
# Core State
# ----------------------------
magnitudes = {}
distributions = {}

# ----------------------------
# Utility Functions
# ----------------------------

def normalize(arr):
    arr = np.array(arr, dtype=float)
    s = np.sum(arr)
    return arr / s if s > 0 else arr

def cross_entropy(p, q):
    p = normalize(p)
    q = normalize(q)
    eps = 1e-9
    return -np.sum(p * np.log(q + eps))

def magnitude_to_freq(m):
    # Log mapping preserves order
    base = 220.0
    return base * (2 ** (m / 5.0))

# ----------------------------
# Synth Engine
# ----------------------------

class Synth:
    def __init__(self):
        self.phase = 0.0
        self.t = 0

    def sine(self, freq, amp, frames):
        t = (np.arange(frames) + self.t) / SR
        return amp * np.sin(2 * np.pi * freq * t)

    def distribution_sound(self, dist, frames):
        dist = normalize(dist)
        signal = np.zeros(frames)

        for i, w in enumerate(dist):
            freq = 220 * (i + 1)
            signal += self.sine(freq, w, frames)

        return signal

    def xent_sound(self, p, q, frames):
        H = cross_entropy(p, q)

        # Map entropy → instability
        base = self.distribution_sound(p, frames)

        # Modulation increases with entropy
        mod_freq = 2 + H * 10
        mod = np.sin(2 * np.pi * mod_freq * np.arange(frames) / SR)

        # Noise component
        noise = np.random.randn(frames) * (H * 0.1)

        return base * (1 + 0.5 * mod) + noise

synth = Synth()

# ----------------------------
# Parser (Very Simple)
# ----------------------------

def parse_line(line):
    tokens = line.strip().split()

    if not tokens:
        return None

    if tokens[0] == "magnitude":
        name = tokens[1]
        val = float(tokens[3])
        magnitudes[name] = val

    elif tokens[0] == "distribution":
        name = tokens[1]
        raw = line.split("=")[1].strip()
        values = eval(raw)
        distributions[name] = normalize(values)

    elif tokens[0] == "emit":
        return tokens[1:]

    return None

# ----------------------------
# Audio Handlers
# ----------------------------

def render_command(cmd, frames):
    if cmd[0] == "tone":
        m = magnitudes[cmd[1]]
        freq = magnitude_to_freq(m)
        amp = m / (max(magnitudes.values()) + 1e-6)
        return synth.sine(freq, amp, frames)

    elif cmd[0] == "spectrum":
        d = distributions[cmd[1]]
        return synth.distribution_sound(d, frames)

    elif cmd[0] == "xent":
        p = distributions[cmd[1]]
        q = distributions[cmd[2]]
        return synth.xent_sound(p, q, frames)

    return np.zeros(frames)

# ----------------------------
# Real-Time Engine
# ----------------------------

class Engine:
    def __init__(self):
        self.current_cmd = None

    def callback(self, outdata, frames, time_info, status):
        if self.current_cmd:
            audio = render_command(self.current_cmd, frames)
        else:
            audio = np.zeros(frames)

        outdata[:] = audio.reshape(-1, 1)

    def run(self, program):
        lines = program.split("\n")

        with sd.OutputStream(callback=self.callback, channels=1, samplerate=SR, blocksize=BLOCK):
            for line in lines:
                cmd = parse_line(line)

                if cmd:
                    self.current_cmd = cmd
                    # play for a short duration
                    time.sleep(1.5)

            # fade out
            self.current_cmd = None
            time.sleep(1)


# ----------------------------
# Example Program
# ----------------------------

program = """
magnitude a = 2
magnitude b = 5

distribution P = [0.2, 0.3, 0.5]
distribution Q = [0.4, 0.4, 0.2]

emit tone a
emit tone b
emit spectrum P
emit xent P Q
"""

# ----------------------------
# Run
# ----------------------------

if __name__ == "__main__":
    engine = Engine()
    engine.run(program)
