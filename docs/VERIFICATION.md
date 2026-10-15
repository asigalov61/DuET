# Release verification

The inference-only bundle was extracted read-only from retained epoch83.
The original checkpoint SHA256 was verified before conversion. Only the unused
training decoder-mask embedding was omitted; all remaining tensor names and
shapes loaded strictly. Optimizer and other training metadata were not exported.

Checks against the reference implementation:

- A newly authored synthetic MIDI covers three windows, simultaneous attacks,
  short notes, exact window boundaries, pedal and carry-in sounding notes.
- Frame/note features match element-for-element for all three windows.
- Encoder frame and note memory maximum absolute difference: **0**.
- Cached and uncached generated rows agree with the reference model.
- Full-piece generated records match exactly, including generated history.
- A real performance spanning **7 windows / 2,000 notes** produces identical
  records to the retained epoch83 export. This is an implementation regression
  check, not a new generalization or quality claim. The performance is not shipped.

See `verification.json` for the machine-readable result. CPU regression tests
cover input cleanup, half-open boundaries, padding, RoPE null pitch handling,
overlap cleanup, two MIDI hands/channel0, and cross-bar MusicXML round-trip.
The wheel includes the frozen numerical vocabulary and contains no model weights.

The numerical comparison used float32 CUDA inference with TF32 disabled.
Other hardware/PyTorch versions may differ numerically. CPU package import and
regression tests were checked; full-piece CPU performance was not benchmarked.
