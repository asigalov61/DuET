"""Upload the fixed release and deploy a private Space with the active HF account."""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", required=True)
    parser.add_argument("--backend", choices=["zero", "cpu"], default="zero")
    parser.add_argument("--model-name", default="DuET")
    parser.add_argument("--space-name", default="DuET-demo")
    parser.add_argument(
        "--runtime-token-env", default="DUET_HF_READ_TOKEN",
        help="Environment variable containing a separate model-read-only HF token",
    )
    args = parser.parse_args()
    runtime_token = os.environ.get(args.runtime_token_env, "").strip()
    if not runtime_token:
        parser.error(
            "Set the runtime token environment variable to a separate model-read-only "
            "HF token before deployment; the login/write token is never copied."
        )
    root = Path(__file__).resolve().parents[1]
    bundle = root / "artifacts/epoch83"
    config = json.loads((bundle / "config.json").read_text())
    digest = hashlib.sha256((bundle / "model.safetensors").read_bytes()).hexdigest()
    if digest != config["weights_sha256"]:
        raise ValueError("Local weight checksum mismatch.")
    api = HfApi()
    identity = api.whoami()["name"]
    if identity != args.owner:
        raise ValueError("Active account differs from requested owner.")
    model_repo = f"{args.owner}/{args.model_name}"
    space_repo = f"{args.owner}/{args.space_name}"
    api.create_repo(model_repo, private=True, exist_ok=True)
    if not api.model_info(model_repo).private:
        raise ValueError("Refusing to upload to a public model repository.")
    with tempfile.TemporaryDirectory(prefix="duet-model-") as temp:
        stage = Path(temp)
        for name in ["config.json", "model.safetensors", "LICENSE-MODEL"]:
            shutil.copyfile(bundle / name, stage / name)
        card = (
            (bundle / "README.md")
            .read_text()
            .replace(
                "Private release preparation; weights have not been published on Hugging Face.",
                "Private epoch83 release. Model weights: CC BY-NC 4.0.",
            )
        )
        (stage / "README.md").write_text(card)
        api.upload_folder(
            repo_id=model_repo,
            folder_path=stage,
            commit_message="Retained epoch83 inference release",
        )
    info = api.model_info(model_repo, files_metadata=True)
    remote = next(f for f in info.siblings if f.rfilename == "model.safetensors")
    if remote.lfs.sha256 != digest or not info.private:
        raise ValueError("Remote weight integrity/privacy check failed.")
    print("Private model uploaded and verified:", model_repo, flush=True)
    api.create_repo(
        space_repo,
        repo_type="space",
        space_sdk="gradio" if args.backend == "zero" else "docker",
        private=True,
        exist_ok=True,
    )
    if not api.repo_info(space_repo, repo_type="space").private:
        raise ValueError("Refusing to deploy to a public Space.")
    # Runtime only: no token is written into a file or Docker layer.
    api.add_space_secret(space_repo, key="HF_TOKEN", value=runtime_token)
    api.add_space_variable(space_repo, key="MODEL_REPO", value=model_repo)
    api.add_space_variable(space_repo, key="MODEL_REVISION", value=info.sha)
    with tempfile.TemporaryDirectory(prefix="duet-space-") as temp:
        stage = Path(temp)
        for name in [
            "pyproject.toml",
            "LICENSE",
            "LICENSE-MODEL",
            "THIRD_PARTY_NOTICES.md",
        ]:
            shutil.copyfile(root / name, stage / name)
        shutil.copytree(
            root / "src",
            stage / "src",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
        )
        (stage / "demo").mkdir()
        shutil.copyfile(root / "demo/app.py", stage / "demo/app.py")
        shutil.copyfile(root / "demo/engraving.py", stage / "demo/engraving.py")
        shutil.copyfile(root / "demo/musescore_runtime.py", stage / "demo/musescore_runtime.py")
        if args.backend == "zero":
            shutil.copyfile(root / "demo/SPACE_ZERO_README.md", stage / "README.md")
            shutil.copyfile(
                root / "demo/requirements-zero.txt", stage / "requirements.txt"
            )
            shutil.copyfile(root / "demo/packages.txt", stage / "packages.txt")
        else:
            shutil.copyfile(root / "demo/Dockerfile", stage / "Dockerfile")
            shutil.copyfile(root / "demo/SPACE_README.md", stage / "README.md")
        api.upload_folder(
            repo_id=space_repo,
            repo_type="space",
            folder_path=stage,
            commit_message="Deploy private audio score demo",
        )
    api.request_space_hardware(
        space_repo, hardware="zero-a10g" if args.backend == "zero" else "cpu-basic"
    )
    assert api.repo_info(space_repo, repo_type="space").private
    receipt = {
        "model_repo": model_repo,
        "model_revision": info.sha,
        "weights_sha256": digest,
        "space_repo": space_repo,
        "private": True,
    }
    (root / "artifacts/deployment.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
