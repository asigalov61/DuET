# Private Hugging Face deployment

This Gradio ZeroGPU Space wraps the fixed epoch83 inference library with Gradio. It
supports MIDI, audio through Transkun v2, and single YouTube links through yt-dlp.
Verovio renders the generated MusicXML as paginated staff notation. MusicXML
remains available if preview rendering fails. No training code is deployed.

Authenticate with `hf auth login` using permission to create/write repositories,
then run from the repository root:

```bash
python demo/deploy.py --owner YOUR_HF_USERNAME
```

The script uploads `artifacts/epoch83` to a **private** model repository,
checks the remote weight SHA256, and deploys a **private** Gradio ZeroGPU Space. It refuses
public existing repositories, pins the model revision and puts the active HF
token in a runtime Space secret (never source code). Prefer replacing that secret
with a model-read-only token after deployment. No paid hardware is requested.
Account subscription/quota requirements are controlled by Hugging Face.

Private personal Spaces are accessed by the owner; collaborators may need an
organization Space and its access controls. Do not make the Space public merely
to share with another member. Code is MIT; DuET weights remain CC BY-NC 4.0.

Audio is decoded on CPU and truncated to its first 240 seconds before GPU
transcription. YouTube downloads request only the first 240 seconds. The
100 MB upload limit remains; MIDI retains its 120 second / 5 MB limit. The demo queues one
inference at a time, and keeps temporary results for up to 24 hours. Old job files
are removed on the next run. Use the library for full pieces. ZeroGPU uses the visitor GPU quota and may queue. Use `--backend cpu` for the
CPU Basic Docker fallback. No dedicated GPU is selected.

FFmpeg decodes audio. YouTube downloads can fail because of hosting-IP blocks,
login requirements, regional availability, or site changes; upload authorized
audio instead. Cookies, login bypasses, and DRM circumvention are not provided.
Only video IDs on HTTPS YouTube URLs are accepted. Tokens and model files are
outside the directory allowed for result downloads.

Validation: unit tests cover MIDI export and audio-stage orchestration; CI also
executes the real packaged Transkun model on generated audio. A deployment must
additionally be checked for a running Space and an end-to-end sample request.
