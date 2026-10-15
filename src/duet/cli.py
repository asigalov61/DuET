"""Command-line entry point."""

import argparse
import json
import sys
from pathlib import Path
import torch
from .pipeline import DuET


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe piano audio or performance MIDI into MusicXML and score MIDI."
    )
    parser.add_argument("input", help="Input audio or performance MIDI")
    parser.add_argument(
        "--input-type", choices=["auto", "midi", "audio"], default="auto"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face model repository or downloaded model directory",
    )
    parser.add_argument("--output", required=True, help="New output directory")
    parser.add_argument(
        "--device", default=None, help="Default: CUDA when available, otherwise CPU"
    )
    parser.add_argument("--revision", help="Hugging Face commit or tag")
    parser.add_argument(
        "--offline", action="store_true", help="Use only local/cached model files"
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        model = DuET.from_pretrained(
            args.model,
            device=args.device,
            revision=args.revision,
            local_files_only=args.offline,
        )
        is_midi = args.input_type == "midi" or (
            args.input_type == "auto"
            and Path(args.input).suffix.lower() in {".mid", ".midi"}
        )
        transcribe = model.transcribe if is_midi else model.transcribe_audio
        report = transcribe(
            args.input, args.output, progress=lambda w: print(json.dumps(w), flush=True)
        )
        print(f"Saved {report['input_notes']} note predictions to {args.output}")
    except Exception as exc:
        print(f"DuET: {exc}", file=sys.stderr)
        return 1
    return 0
