---
title: DuET Piano to Score
emoji: 🎹
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
python_version: 3.12
app_file: demo/app.py
license: mit
---
Private DuET ZeroGPU demo. Audio longer than 240 seconds is truncated to its first
240 seconds before transcription. MIDI limit remains 120 seconds.
Code: MIT. DuET epoch83 weights: CC BY-NC 4.0, loaded from a private model repository.
