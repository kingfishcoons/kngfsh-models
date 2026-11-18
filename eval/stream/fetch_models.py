#!/usr/bin/env python3
"""Downloads one bake-off engine's files into DIR and checks or records their sha256 (models.json, D126).

    python3 eval/stream/fetch_models.py --models eval/stream/models.json --engine NAME --dir DIR --lock lock.json
"""
import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def get(url, dest, want=None):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f)
            break
        except OSError as e:
            if attempt == 3:
                raise
            print(f"retrying {url}: {e}")
    got = sha256(dest)
    if want and got != want:
        raise SystemExit(f"{url}: sha256 {got}, expected {want}")
    return got


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--models", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--dir", required=True)
    parser.add_argument("--lock", required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.models).read_text())["engines"][args.engine]
    fetch = spec["fetch"]
    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    lock = {"engine": args.engine, "files": {}}
    if "moonshine" in fetch:
        from moonshine_voice import download as d

        info = d.find_model_info(fetch["moonshine"]["language"], d.ModelArch(fetch["moonshine"]["arch"]))
        path, _ = d.download_model_from_info(info, cache_root=out / "cache")
        target = out
        for f in Path(path).iterdir():
            shutil.copy2(f, target / f.name)
        lock["source"] = info["download_url"]
        files = [f for f in target.iterdir() if f.is_file()]
    elif "tar" in fetch:
        archive = out / "model.tar.bz2"
        lock["archive_sha256"] = get(fetch["tar"], archive, fetch["sha256"])
        lock["source"] = fetch["tar"]
        with tarfile.open(archive) as t:
            t.extractall(out, filter="data")
        files = [f for f in (out / spec["subdir"]).rglob("*") if f.is_file()]
    elif "zip" in fetch:
        archive = out / "model.zip"
        lock["archive_sha256"] = get(fetch["zip"], archive, fetch["sha256"])
        lock["source"] = fetch["zip"]
        with zipfile.ZipFile(archive) as z:
            z.extractall(out)
        files = [f for f in (out / spec["subdir"]).rglob("*") if f.is_file()]
    elif "hf" in fetch:
        base = f"https://huggingface.co/{fetch['hf']}/resolve/{fetch['revision']}/"
        lock["source"] = f"{fetch['hf']}@{fetch['revision']}"
        for name, want in fetch["files"].items():
            get(base + name, out / name, want)
        files = [out / name for name in fetch["files"]]
    else:
        raise SystemExit(f"{args.engine}: nothing to fetch")
    for f in sorted(files):
        if f.name.startswith("model."):
            continue
        lock["files"][str(f.relative_to(out))] = {"bytes": f.stat().st_size, "sha256": sha256(f)}
    lock["bytes"] = sum(v["bytes"] for v in lock["files"].values())
    Path(args.lock).write_text(json.dumps(lock, indent=1))
    print(json.dumps({k: v for k, v in lock.items() if k != "files"}))


if __name__ == "__main__":
    main()
