---
license: cc-by-nc-4.0
library_name: duet-score
tags:
- music
- midi
- music-transcription
---

# DuET: Note--Frame Dual Encoding for Piano Performance-to-Score Transcription

## Model

DuET transcribes piano performance MIDI into structured musical scores with a note encoder, a frame encoder, and a note-aligned autoregressive decoder. The model jointly learns score prediction and frame-level beat and downbeat tracking, using TimeRoPE and TPRoPE to represent relative time and pitch relationships.

[GitHub: code, installation, and usage](https://github.com/wei-zeng98/DuET)
