# 🎧 MagL+X  
### A Magnitude-Based Audio Programming Language with Cross-Entropy Sonification

MagL+X is an experimental programming language and runtime that **maps mathematical notions of magnitude and probabilistic structure to sound**.

It allows you to:
- Represent **magnitudes as audio signals**
- Encode **distributions as spectra**
- Hear **Cross-Entropy as sonic tension**
- Stream everything in **real time**

---

# 🚀 Features

## ✅ Core Concepts

| Concept | Meaning | Audio Mapping |
|--------|--------|--------------|
| Magnitude | Scalar quantity | Pitch + loudness |
| Distribution | Normalized magnitude field | Spectral mixture |
| CrossEntropy | Structural mismatch | Noise + modulation |
| Ratio | Multiplicative relation | Harmonic interval |
| Order | Comparability | Pitch hierarchy |

---

## 🔊 Audio Semantics

- **Larger cross-entropy `H(P, Q)`** → more modulation and noise (less stable sound)
- **Smaller cross-entropy `H(P, Q)`** → calmer, more harmonic sound
- **Magnitude increase** → louder + higher pitch
- **Distribution shift** → timbre change

> **⚠️ Note on what `xent` currently measures.** The `xent` emission computes the
> *raw cross-entropy* `H(P, Q) = -Σ P·log Q`. Because `H(P, P)` equals the entropy
> of `P` (not zero), **two identical distributions do not fall silent** — they
> still produce modulation and noise proportional to `P`'s entropy. So the current
> `xent` sound reflects "spread of `P`" mixed with "mismatch between `P` and `Q`",
> not pure mismatch. If you want a measure that vanishes when `P == Q` (true
> "consonance at convergence"), use KL or Jensen–Shannon divergence instead — see
> [Known limitations](#-known-limitations).

---

## 🎚 Real-Time Streaming

- Live audio output using `sounddevice`
- Emits each instruction as a playable segment
- Supports dynamic program execution

---

# 📦 Installation

```bash
pip install -r requirements.txt
```

or directly:

```bash
pip install numpy sounddevice torch pywebview
```

- `numpy`, `sounddevice` — core DSL, synth, and realtime audio
- `torch` — the PyTorch loss-sonification hook (`engine.py`, `example.py`, notebook)
- `pywebview` — optional web UI (`magic_webview.py`)

---

# ▶️ Running

The DSL interpreter lives in `magl.py`. Run it directly to play the built-in
example program:

```bash
python magl.py
```

**New to MagL+X? Start with the notebook.** `maglx_notebook.ipynb` is a
self-contained, runnable walkthrough — it trains a small PyTorch model and
sonifies its loss in real time, and is the recommended way to see the system end
to end.

---

# 🧠 Language Overview

## Magnitudes

```magl
magnitude a = 2
magnitude b = 5
```

- Scalars
- Mapped to frequency + amplitude

---

## Distributions

```magl
distribution P = [0.2, 0.3, 0.5]
distribution Q = [0.4, 0.4, 0.2]
```

- Automatically normalized
- Represent spectral energy

---

## Emission

```magl
emit tone a
emit spectrum P
emit xent P Q
```

---

# 🎧 Example Program

```magl
magnitude a = 2
magnitude b = 5

distribution P = [0.2, 0.3, 0.5]
distribution Q = [0.4, 0.4, 0.2]

emit tone a
emit tone b
emit spectrum P
emit xent P Q
```

---

# 🔊 What You’ll Hear

1. **tone a**  
   - Low pitch, moderate amplitude  

2. **tone b**  
   - Higher pitch, louder  

3. **spectrum P**  
   - Harmonic blend of frequencies  

4. **xent P Q**  
   - Audible mismatch:
     - modulation
     - noise
     - instability  

---

# 🧮 Cross-Entropy Interpretation

MagL+X renders cross-entropy as sound:

> The **raw cross-entropy** of a target field `P` relative to a comparison field `Q`,
> mapped to modulation and noise.

Mathematically:

```
H(P, Q) = -Σ P(x) log Q(x)
```

Audio mapping (driven by the value of `H`):

| `H(P, Q)` | Perception |
|-----------|------------|
| Lower | calmer, more harmonic |
| Medium | beating / detuning |
| Higher | noise / chaos |

> **Cross-entropy is not a true distance.** It is not zero when `P == Q` (it equals
> the entropy of `P`), and it is asymmetric (`H(P, Q) ≠ H(Q, P)`). The
> `CrossEntropyTracker` calls it as `H(previous, current)`, so the order matters.
> For a symmetric measure that *does* vanish at `P == Q` — the behavior you want if
> you intend "silence at convergence" — use Jensen–Shannon divergence. See
> [Known limitations](#-known-limitations).

---

# 🏗 Architecture

MagL+X has **two front ends** that feed one synth.

**1. The text DSL** (`magl.py`): you write MagL source, it is parsed into
magnitude/distribution state and rendered.

```
MagL Code
   ↓
Parser  (magl.py)
   ↓
Magnitude / Distribution State
   ↓
Audio Mapping
   ↓
Real-Time Synth (sounddevice)
   ↓
Speaker Output
```

**2. The PyTorch loss hook** (`engine.py`, `live_engine.py`): a training loop's
per-sample loss tensors are turned directly into audio, bypassing the text
parser. This is what lets you *hear a model learn*.

```
PyTorch per-sample loss tensor
   ↓
loss_to_audio_fn  /  CrossEntropyTracker   (live_engine.py)
   ↓
Audio callback function
   ↓
Real-Time Synth (sounddevice)
   ↓
Speaker Output
```

`engine.py` additionally offers `loss_to_magl()` / `xent_to_magl()`, which emit
**MagL source strings** from loss tensors — a bridge from the tensor world back
into the text DSL.

---

# 🎛 Internals

### Magnitude → Frequency

```python
f = base * 2 ** (m / k)
```

### Distribution → Sound

- Each element → harmonic partial
- Weight → amplitude

### CrossEntropy → Sound

Both modulation rate and noise level scale with the value of `H(P, Q)`:

```python
mod_freq = 2 + H * 10     # faster modulation as H rises
noise    = randn * H * 0.1  # more noise as H rises
```

Because `H(P, Q)` mixes the entropy of `P` with the mismatch between `P` and
`Q`, the resulting sound reflects both (see [Known limitations](#-known-limitations)).

---

# 📁 Project Files

| File | Role |
|------|------|
| `magl.py` | The text DSL: parser, synth, cross-entropy, and a blocking realtime engine. Run `python magl.py` for the built-in demo. |
| `engine.py` | PyTorch → MagL **source** helpers (`loss_to_magl`, `xent_to_magl`, `loss_to_magnitudes`) plus a `MagLHook`. |
| `live_engine.py` | PyTorch → **audio** helpers (`loss_to_audio_fn`, `CrossEntropyTracker`) and a non-blocking `LiveEngine`. *Snippet:* relies on `Synth`, `SR`, `sd` from `magl.py`'s scope — paste alongside it rather than importing standalone. |
| `example.py` | Illustrative training-loop **snippet** showing where the audio hook goes. Pseudocode (`model = ...`), not runnable as-is. |
| `magic_webview.py` | Optional browser UI (pywebview) with its own synth and two demo buttons. |
| `maglx_notebook.ipynb` | **Recommended starting point.** Self-contained, runnable: trains a model and sonifies its loss. |
| `example.md` | Deep-dive on the loss→audio hook and the sounds it produces. |

> **Naming.** The project is **MagL+X**; the interpreter module is **`magl.py`**.
> (Earlier drafts referred to `maglx.py` — that file does not exist; use `magl.py`.)

---

# 🧠 Conceptual Insight

MagL+X reframes audio as:

> A **projection of abstract mathematical structure into perception**

Instead of:
- playing notes

You are:
- expressing **relationships between magnitudes**
- hearing **prediction error**

---

<a id="-known-limitations"></a>
# ⚠️ Known limitations

These are current behaviors to be aware of. They are documented honestly here
rather than hidden; several are good first issues.

- **`xent` is raw cross-entropy, not a divergence.** `H(P, Q) = -Σ P·log Q` does
  not vanish when `P == Q` (it equals the entropy of `P`) and is asymmetric. If
  you want a measure that falls silent at convergence, swap in **KL divergence**
  `Σ P·log(P/Q)` or, for a symmetric and bounded result, **Jensen–Shannon
  divergence**. The `CrossEntropyTracker` calls `H(previous, current)`, so order
  currently matters.

- **Loss vectors must be the same length to compare.** `xent_sound(p, q)`
  multiplies element-wise, so feeding the `CrossEntropyTracker` two consecutive
  batches of *different* sizes (e.g. a smaller final batch in an epoch) raises a
  broadcasting error. Bin both loss vectors into a fixed number of value-bins
  before comparing to make this robust (and order-invariant — see below).

- **Spectrum depends on sample order.** In `distribution_sound`, element *i* of
  the loss vector drives the harmonic at `220·(i+1)` Hz, so shuffling the
  dataloader changes the sound even when the model state is identical.
  Histogramming losses by *value* (rather than indexing by position) would make
  the sonification order-invariant.

- **No output limiting.** `xent_sound` returns `base·(1 + 0.5·mod) + noise` with
  no normalization, so large `H` can clip. Until a limiter is added, keep volume
  low — especially on headphones (see Notes).

- **Realtime path is not phase-continuous.** In `magl.py`, the synth's time index
  is not advanced between callback blocks, so each block restarts the waveform at
  *t = 0*, which can produce a periodic click on sustained tones.

- **Parser is minimal.** `distribution` right-hand sides are parsed with `eval`;
  only use trusted MagL source. There is no error handling for unknown commands
  or references to undefined names.

---

# 🧩 Future Work

- Full parser + grammar
- Stereo spatialization (P vs Q)
- Live coding REPL
- ML model sonification
- Visual interface
- Swap raw cross-entropy for KL / Jensen–Shannon divergence (see Known limitations)

---

# ⚠️ Notes

- Audio output depends on your system device
- High CrossEntropy can generate loud/noisy output

---

# 📜 License

Experimental / research use.

---

# 💡 TL;DR

MagL+X lets you:

> **Hear mathematics. Hear probability. Hear error.**
