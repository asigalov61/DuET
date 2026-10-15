# Fixed inference architecture

| Component | Configuration |
|---|---|
| Encoder | 8 dual-stream fusion blocks |
| Decoder | 8 autoregressive blocks, alternating note/frame cross-attention |
| Width / attention | 512 / 8 heads |
| FFN | SwiGLU, intermediate width 1408 |
| Frame input | 50 FPS, two kernel-5 convolutions, no downsampling |
| Frame features | held, sustained, onset, onset velocity; 88 pitches; two pedal channels |
| Note features | pitch embedding, continuous duration and velocity |
| Note self/cross RoPE | 32 time dimensions (.1–64 s), 32 pitch dimensions (12–256 semitones), pitch origin 60 |
| Frame and fusion RoPE | continuous time, scaled by .02 seconds |
| Decoder self RoPE | sequence index |
| New region / history | 32 seconds / 8 seconds, fixed performance clock |
| Output | one compound score record per input note; copied input pitch |

Frame and note times use the same window-left origin. Carry-in sounding notes
remain in the frame roll; note tokens require an onset inside the window.
History contains previous generated records whose input onsets fall in the
preceding eight seconds. There is no reference-score history at inference.

Retained auxiliary modules are the pitch head, beat head, beat-index head and two
frame-phase heads. They belong to the release architecture. The inference library
contains no losses, optimizer, training loop, augmentation, ablation routing,
pointer/EOS machinery or compatibility loaders for earlier architectures.
