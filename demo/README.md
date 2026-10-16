# Demo and maintainer deployment

The demo accepts **piano audio only**. Transkun v2 produces performance MIDI,
then DuET produces score MIDI. MuseScore 4.6.5 imports that score MIDI to
produce MusicXML, PDF and paginated SVG previews. MIDI input is supported by
the DuET library and CLI, but not by this demo's upload interface.

Audio is limited to 100 MB. Only its first 240 seconds are processed; longer
recordings are truncated automatically. One inference runs at a time. Temporary
results are eligible for cleanup after 24 hours, on the next request.

Downloads include `score.mid`, `score.musicxml`, `score.pdf`, `report.json`,
and `performance.mid` when available. If engraving fails, score MIDI and the
library-generated MusicXML remain available. Successful engraving replaces the
MusicXML with MuseScore's MIDI-import result; it can differ from the library's
direct MusicXML export.

## Maintainer deployment tool

`deploy.py` is a maintainer tool, not a prerequisite for using DuET. It requires
a verified release bundle in `artifacts/epoch83/` containing `config.json`,
`model.safetensors`, `LICENSE-MODEL` and `README.md`. These files are not in Git.
The tool only creates or updates **private** model and Space repositories and
refuses existing public repositories. It does not publish a release.

Authenticate using `hf auth login` with repository write permissions. Separately
create a fine-grained token with read access to the model repository. Supply it
without putting its value in command history:

```bash
read -r -s -p "Model read token: " DUET_HF_READ_TOKEN
export DUET_HF_READ_TOKEN
python demo/deploy.py --owner YOUR_HF_USERNAME
unset DUET_HF_READ_TOKEN
```

The read token is stored as the Space's `HF_TOKEN` secret. The tool never copies
the active write/login token. It verifies the weight checksum and pins the model
revision. The default backend is ZeroGPU; use `--backend cpu` for CPU Basic.
Account access and hardware eligibility are managed by Hugging Face.

GitHub and the Space are separate repositories. Pushing code to GitHub does not
update the Space; deployment explicitly uploads a selected set of runtime files.
The Docker build also excludes weights, credentials, Git history and outputs.
No training code is deployed. Third-party terms are documented in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Verification

Run the repository tests and CI checks before deployment. After deployment,
check the Space runtime and an end-to-end audio request, including generated
MusicXML, MIDI, PDF and preview. CI tests use generated inputs.

GPU duration is estimated **after** decoding and truncation, from the prepared
WAV length: `max(30, ceil(15 + 0.65 * seconds))`. This initial heuristic requests
30–171 seconds and is not a runtime guarantee. Visitor limits still apply.
`report.json` records the requested duration and inference wall time measured
inside the GPU function (excluding queueing and engraving) for calibration.
