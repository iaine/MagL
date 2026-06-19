# 📘 MagL+X Example: PyTorch Loss → Real-Time Audio Sonification

This document describes how to use the **MagL+X PyTorch hook** to convert per-sample losses into real-time audio and explains the sound characteristics produced.

---

# 🧠 Function Overview

## `loss_to_audio_fn(loss_tensor, synth)`

### Purpose

Converts a PyTorch loss tensor (with `reduction='none'`) into an **audio function** suitable for real-time playback.

---

## ✅ Definition

```python
def loss_to_audio_fn(loss_tensor, synth):
    values = loss_tensor.detach().cpu().flatten().numpy()
    values = np.clip(values, 0, None)

    if values.sum() > 0:
        values = values / values.sum()

    def audio_fn(frames):
        return synth.distribution_sound(values, frames)

    return audio_fn
```

---

# ⚙️ How It Works

## Step-by-step

1. **Extract values**
   - Converts tensor → NumPy array
   - Flattens multi-dimensional loss

2. **Sanitize values**
   - Clips negatives to 0

3. **Normalize**
   - Converts loss into a probability distribution

4. **Wrap as audio function**
   - Returns callable for streaming engine

---

# 🔊 Audio Mapping

The function interprets the loss tensor as a **spectral distribution**.

| Loss Property | Audio Effect |
|--------------|------------|
| Magnitude of values | Loudness per frequency |
| Number of elements | Number of harmonics |
| Distribution shape | Timbre |
| Uniform distribution | Smooth, even tone |
| Peaked distribution | Strong dominant pitch |

---

# 🎧 Sound Types Produced

## 1. Uniform Loss

```python
tensor([0.5, 0.5, 0.5, 0.5])
```

### Sound
- Balanced harmonic spectrum
- Smooth and stable
- “Neutral” tone

---

## 2. Sparse / Peaked Loss

```python
tensor([0.0, 0.0, 1.0, 0.0])
```

### Sound
- Strong single frequency
- Clear pitch
- Minimal harmonic complexity

---

## 3. High Variance Loss

```python
tensor([0.1, 0.9, 0.2, 0.8])
```

### Sound
- Irregular harmonic structure
- Slight beating or roughness
- Perceived instability

---

## 4. Dense High-Dimensional Loss

```python
tensor([... many values ...])
```

### Sound
- Rich, noisy spectrum
- Broadband frequency content
- Approaches noise as dimensionality increases

---

# 🔁 Cross-Entropy Audio (`CrossEntropyTracker`)

## Definition

```python
class CrossEntropyTracker:
    def __init__(self):
        self.prev = None

    def to_audio_fn(self, current_loss, synth):
        curr = current_loss.detach().cpu().flatten().numpy()
        curr = np.clip(curr, 0, None)

        if curr.sum() > 0:
            curr = curr / curr.sum()

        if self.prev is None:
            self.prev = curr
            return lambda frames: np.zeros(frames)

        prev = self.prev
        self.prev = curr

        def audio_fn(frames):
            return synth.xent_sound(prev, curr, frames)

        return audio_fn
```

---

# 🔊 Cross-Entropy Sound Types

> **What these describe.** The table below describes the *intended* mapping from
> how much the loss distribution **changes** between steps to how the sound
> behaves. Note a current caveat: `xent_sound` uses raw cross-entropy
> `H(P, Q) = -Σ P·log Q`, which does **not** reach zero at convergence (it equals
> the entropy of the distribution). So a converged model will sound *calmer* but
> not fully silent or perfectly stable — residual modulation/noise proportional
> to the distribution's entropy remains. For true "silence at convergence," drive
> the sound with KL or Jensen–Shannon divergence instead. See the README's
> *Known limitations*.

## 1. Low Change (Convergence)

- Stable tone
- Minimal modulation
- Harmonic consistency

## 2. Moderate Change

- Beating / phasing
- Mild amplitude modulation
- Slow fluctuations

## 3. High Change (Learning Shock)

- Audible noise bursts
- Rapid modulation
- Chaotic fluctuations

## 4. Training Progression

| Phase | Sound |
|------|------|
| Initial | Noise, chaos |
| Mid-training | Fluctuating tones |
| Converged | Stable harmony |

---

# 🎯 Example Usage in Training Loop

```python
criterion = nn.CrossEntropyLoss(reduction="none")
tracker = CrossEntropyTracker()

for x, y in dataloader:
    logits = model(x)
    loss = criterion(logits, y)

    audio_fn = tracker.to_audio_fn(loss, synth)
    engine.current_audio_fn = audio_fn

    loss.mean().backward()
```

---

# 🧠 Conceptual Meaning

This system converts:

- **Loss values → spectral structure**
- **Loss changes → sonic tension**

Resulting in:

> A real-time auditory representation of model learning dynamics.

---

# ⚠️ Notes

- Large tensors produce dense/noisy spectra
- Very high loss variance may create loud output
- Normalize carefully to avoid clipping

---

# 💡 Summary

The MagL+X loss hook enables:

- Real-time sonification of training
- Detection of instability via sound
- Perceptual understanding of optimization dynamics

> You are not just observing loss — you are hearing learning.
