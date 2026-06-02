import webview
import json
import numpy as np
import threading
import time

SR = 44100

# ----------------------------
# Synth (Python side)
# ----------------------------
class Synth:
    def sine(self, freq, amp, t):
        return amp * np.sin(2 * np.pi * freq * t)

    def distribution(self, dist, duration=0.5):
        t = np.linspace(0, duration, int(SR * duration), endpoint=False)
        signal = np.zeros_like(t)

        dist = np.array(dist)
        if dist.sum() > 0:
            dist = dist / dist.sum()

        for i, w in enumerate(dist):
            signal += self.sine(220 * (i + 1), w, t)

        return signal

    def xent(self, p, q, duration=0.5):
        eps = 1e-9
        p = np.array(p)
        q = np.array(q)

        p = p / (p.sum() + eps)
        q = q / (q.sum() + eps)

        H = -np.sum(p * np.log(q + eps))

        base = self.distribution(p, duration)

        t = np.linspace(0, duration, len(base), endpoint=False)
        mod = np.sin(2 * np.pi * (2 + H * 10) * t)
        noise = np.random.randn(len(base)) * (H * 0.1)

        return base * (1 + 0.5 * mod) + noise


synth = Synth()

# ----------------------------
# API exposed to JS
# ----------------------------
class API:
    def play_distribution(self, dist):
        dist = json.loads(dist)
        audio = synth.distribution(dist).astype(np.float32)
        return audio.tolist()

    def play_xent(self, p, q):
        p = json.loads(p)
        q = json.loads(q)
        audio = synth.xent(p, q).astype(np.float32)
        return audio.tolist()


api = API()

# ----------------------------
# HTML + JS UI
# ----------------------------
html = """
<!DOCTYPE html>
<html>
<head>
    <title>MagL+X Audio</title>
</head>
<body>
    <h2>🎧 MagL+X Web Audio</h2>

    <button onclick="playDist()">Play Distribution</button>
    <button onclick="playXent()">Play CrossEntropy</button>

    <script>
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();

    async function playBuffer(data) {
        let buffer = audioCtx.createBuffer(1, data.length, 44100);
        let channel = buffer.getChannelData(0);

        for (let i = 0; i < data.length; i++) {
            channel[i] = data[i];
        }

        let source = audioCtx.createBufferSource();
        source.buffer = buffer;
        source.connect(audioCtx.destination);
        source.start();
    }

    async function playDist() {
        let dist = JSON.stringify([0.1, 0.3, 0.6]);
        let result = await window.pywebview.api.play_distribution(dist);
        playBuffer(result);
    }

    async function playXent() {
        let p = JSON.stringify([0.2, 0.3, 0.5]);
        let q = JSON.stringify([0.5, 0.3, 0.2]);
        let result = await window.pywebview.api.play_xent(p, q);
        playBuffer(result);
    }
    </script>
</body>
</html>
"""

# ----------------------------
# Run App
# ----------------------------
def start():
    webview.create_window("MagL+X", html=html, js_api=api)
    webview.start()

if __name__ == "__main__":
    start()
