# Add to your existing engine

class LiveEngine:
    def __init__(self):
        self.current_audio_fn = None

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
            samplerate=SR,
            blocksize=BLOCK,
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
