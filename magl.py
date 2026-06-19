import numpy as np
import sounddevice as sd
import time
import ast

from synth import Synth

synth = Synth()

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
# Parser (Very Simple)
# ----------------------------

def parse_line(line):
    """Parse one line of MagL source.

    Returns an emit command (list of tokens after 'emit') or None. Malformed
    lines are reported and skipped rather than raising, so a single bad line
    can't tear down the audio stream or halt a program.
    """
    tokens = line.strip().split()

    if not tokens:
        return None

    head = tokens[0]

    # ignore comments and unknown leading tokens softly
    if head.startswith("#"):
        return None

    try:
        if head == "magnitude":
            # form:  magnitude <name> = <number>
            if len(tokens) < 4 or tokens[2] != "=":
                raise ValueError("expected 'magnitude <name> = <number>'")
            name = tokens[1]
            val = float(tokens[3])
            magnitudes[name] = val

        elif head == "distribution":
            # form:  distribution <name> = [<numbers>]
            if "=" not in line:
                raise ValueError("expected 'distribution <name> = [..]'")
            name = tokens[1]
            raw = line.split("=", 1)[1].strip()
            # literal_eval safely parses lists/tuples of numbers only --
            # no arbitrary code execution, unlike eval().
            values = ast.literal_eval(raw)
            if not isinstance(values, (list, tuple)) or not values:
                raise ValueError("distribution must be a non-empty list")
            if not all(isinstance(v, (int, float)) for v in values):
                raise ValueError("distribution elements must be numbers")
            distributions[name] = normalize(values)

        elif head == "emit":
            if len(tokens) < 2:
                raise ValueError("expected 'emit <command> ...'")
            return tokens[1:]

        else:
            print(f"[MagL] skipped unknown command: {head!r}")

    except (ValueError, SyntaxError, IndexError) as e:
        print(f"[MagL] skipping malformed line: {line.strip()!r} ({e})")

    return None

# ----------------------------
# Audio Handlers
# ----------------------------

def render_command(cmd, frames):
    """Render an emit command to audio. Unknown commands or references to
    undefined names produce silence (with a warning) instead of raising, so the
    realtime callback never crashes on bad input."""
    try:
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

        else:
            print(f"[MagL] unknown emit command: {cmd[0]!r}")

    except KeyError as e:
        print(f"[MagL] emit references undefined name: {e}")
    except IndexError:
        print(f"[MagL] emit missing argument(s): {cmd!r}")

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
