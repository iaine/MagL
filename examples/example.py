import torch
import torch.nn as nn
import numpy as np

from synth import Synth
from live_engine import LiveEngine, CrossEntropyTracker

# Your existing synth + engine
synth = Synth()
engine = LiveEngine()
engine.start()

tracker = CrossEntropyTracker()

model = ...
optimizer = ...
criterion = nn.CrossEntropyLoss(reduction="none")

num_epochs = 50

for epoch in range(num_epochs):
    for x, y in dataloader:
        optimizer.zero_grad()

        logits = model(x)

        # PER-SAMPLE LOSS
        loss_per_sample = criterion(logits, y)

        # TRAINING STEP
        loss_per_sample.mean().backward()
        optimizer.step()

        # -------------------------
        # 🔊 AUDIO HOOK
        # -------------------------

        # Option A: hear raw distribution
        audio_fn = loss_to_audio_fn(loss_per_sample, synth)

        # Option B: hear learning dynamics (recommended)
        audio_fn = tracker.to_audio_fn(loss_per_sample, synth)

        # Push into engine
        engine.current_audio_fn = audio_fn
