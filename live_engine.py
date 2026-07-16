# Add to your existing engine
import numpy as np
import sounddevice as sd

class LiveEngine:
    def __init__(self, sr, blocksize):
        self.current_audio_fn = None
        self.SR = sr
        self.BLOCK = blocksize

    def callback(self, outdata, frames, time_info, status):
        if self.current_audio_fn:
            audio = self.current_audio_fn(frames)
        else:
            audio = np.zeros(frames)

        outdata[:] = audio.reshape(-1, 1)

    def start(self):
        self.stream = sd.OutputStream(
            callback=self.callback,
            channels=1,
            samplerate=self.SR,
            blocksize=self.BLOCK,
        )
        self.stream.start()

    def stop(self):
        self.stream.stop()

def loss_to_audio_fn(loss_tensor, synth):
    values = loss_tensor.detach().cpu().flatten().numpy()
    values = np.clip(values, 0, None)

    if values.sum() > 0:
        values = values / values.sum()

    def audio_fn(frames):
        return synth.distribution_sound(values, frames)

    return audio_fn

class CrossEntropyTracker:
    """Sonifies the *change* between consecutive loss distributions.

    mode="classic"   -> the original sound (additive spectrum + modulation).
    mode="wavetable" -> the previous step's timbre is morphed toward the
                        current step's, by the mismatch itself.
    Default is "classic", so existing notebooks sound exactly as before.
    """

    def __init__(self, mode="classic"):
        self.prev = None
        self.mode = mode

    def to_audio_fn(self, current_loss, synth, mode=None):
        curr = current_loss.detach().cpu().flatten().numpy()
        curr = np.clip(curr, 0, None)

        if curr.sum() > 0:
            curr = curr / curr.sum()

        if self.prev is None:
            self.prev = curr
            return lambda frames: np.zeros(frames)

        prev = self.prev
        self.prev = curr
        use_mode = mode or self.mode

        def audio_fn(frames):
            return synth.xent_sound(prev, curr, frames, mode=use_mode)

        return audio_fn


class WavetableTracker:
    """Sonifies loss *magnitude* as volume + timbre (MagL's Path A mapping):

        low loss  -> quiet, pure sine
        high loss -> loud, sawtooth

    A raw loss has no absolute meaning ("is 2.3 high?"), so this calibrates
    against a running maximum: loss is heard relative to the worst seen so far.
    That makes the sonification self-calibrating over a training run -- early
    steps sound loud and harsh, and as the model improves you hear it settle
    toward a quiet sine. You are hearing the trajectory, not the raw number.
    """

    def __init__(self, decay=0.999, floor=1e-6):
        self.running_max = floor
        self.decay = decay          # lets the reference slowly relax downward
        self.floor = floor

    def to_audio_fn(self, current_loss, synth, base_freq=220.0):
        # accept a scalar tensor or a per-sample tensor (mean it)
        val = current_loss.detach().cpu().flatten().numpy()
        loss = float(np.clip(val.mean(), 0.0, None))

        # running max: the reference against which "high" is judged
        self.running_max = max(loss, self.running_max * self.decay, self.floor)
        scale = self.running_max

        def audio_fn(frames):
            return synth.loss_to_sound(loss, frames, base_freq=base_freq,
                                       loss_scale=scale, voice="loss")

        return audio_fn
