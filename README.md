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

- **Low CrossEntropy** → stable, harmonic tones  
- **High CrossEntropy** → noisy, unstable sound  
- **Magnitude increase** → louder + higher pitch  
- **Distribution shift** → timbre change  

---

## 🎚 Real-Time Streaming

- Live audio output using `sounddevice`
- Emits each instruction as a playable segment
- Supports dynamic program execution

---

# 📦 Installation

```bash
pip install numpy sounddevice scipy
```

---

# ▶️ Running

Save the interpreter script as:

```bash
maglx.py
```

Run:

```bash
python maglx.py
```

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

MagL+X treats CrossEntropy as:

> A **distance between structured magnitude fields**, rendered as sound.

Mathematically:

H(P, Q) = -Σ P(x) log Q(x)

Audio mapping:

| Value | Perception |
|------|------------|
| Low | consonance |
| Medium | beating / detuning |
| High | noise / chaos |

---

# 🏗 Architecture

```
MagL Code
   ↓
Parser
   ↓
Magnitude / Distribution State
   ↓
Audio Mapping
   ↓
Real-Time Synth (SoundDevice)
   ↓
Speaker Output
```

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

- Modulation rate ∝ entropy  
- Noise level ∝ entropy  
- Instability ∝ mismatch  

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

# 🧩 Future Work

- Full parser + grammar
- Stereo spatialization (P vs Q)
- Live coding REPL
- ML model sonification
- Visual interface

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
