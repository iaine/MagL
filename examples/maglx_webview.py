import webview
import json
import numpy as np
import threading
import time

from ..synth import Synth

SR = 44100

# ----------------------------
# Synth (Python side)
# ----------------------------
synth = Synth()

# ----------------------------
# API exposed to JS
# ----------------------------
class API:
    def play_distribution(self, dist):
        dist = json.loads(dist)
        audio = synth.distribution_sound(dist).astype(np.float32)
        return audio.tolist()

    def play_xent(self, p, q):
        p = json.loads(p)
        q = json.loads(q)
        audio = synth.xent_sound(p, q).astype(np.float32)
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
    webview.create_window("MagL+X", html="ui/index.html", js_api=api)
    webview.start()

if __name__ == "__main__":
    start()
