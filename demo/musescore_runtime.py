"""Install the pinned official MuseScore AppImage without FUSE or a display."""
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import urllib.request

VERSION = "4.6.5"
URL = "https://github.com/musescore/MuseScore/releases/download/v4.6.5/MuseScore-Studio-4.6.5.253511702-x86_64.AppImage"
SHA256 = "193daa0ea18bcfa90a47145a842275b8069b7b2b8d153e562b15fab5fe50fcaf"


def ensure_musescore():
    cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "duet" / ("musescore-" + VERSION)
    executable = cache / "squashfs-root" / "AppRun"
    if (cache / "verified").is_file() and executable.is_file():
        return str(executable)
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache) as tmp:
        archive = Path(tmp) / "MuseScore.AppImage"
        digest = hashlib.sha256()
        with urllib.request.urlopen(URL, timeout=120) as src, archive.open("wb") as dst:
            while chunk := src.read(1024 * 1024):
                digest.update(chunk)
                dst.write(chunk)
        if digest.hexdigest() != SHA256:
            raise RuntimeError("MuseScore download checksum mismatch")
        archive.chmod(0o755)
        subprocess.run([str(archive), "--appimage-extract"], cwd=tmp, check=True, stdout=subprocess.DEVNULL, timeout=120)
        import shutil
        shutil.rmtree(cache / "squashfs-root", ignore_errors=True)
        (Path(tmp) / "squashfs-root").rename(cache / "squashfs-root")
    (cache / "verified").write_text(SHA256)
    return str(executable)
