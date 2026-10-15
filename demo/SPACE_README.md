---
title: DuET Piano to Score
emoji: 🎹
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
license: mit
---
DuET demo: piano audio to score MIDI, MusicXML and PDF.
Only the first 240 seconds are processed; longer audio is truncated.
Code is MIT; DuET epoch83 weights are separately licensed CC BY-NC 4.0.
The private model is loaded at runtime via a Space secret; it is not served as a download.
