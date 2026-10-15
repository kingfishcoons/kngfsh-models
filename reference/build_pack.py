#!/usr/bin/env python3
"""Builds the reference clip pack from the FLEURS test split (google/fleurs, CC-BY-4.0).

Run by .github/workflows/reference.yml. Writes, under --out:
  audio/<lang>-<fleurs id>.wav   16 kHz mono PCM16, one utterance each
  manifest.json                  the clips, their transcripts and attribution
  transcripts-<lang>.tsv         file, duration and the raw transcript, one line per clip

Selection is seeded, so a rebuild at the same revision picks the same clips. Transcripts are copied unchanged.
"""
import argparse
import csv
import io
import json
import math
import os
import random
import tarfile
from pathlib import Path

import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download
from scipy.signal import resample_poly

REPO = "google/fleurs"
# The dataset revision the pack was built from. Pinning it keeps a rebuild reproducible.
REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
# (FLEURS config, app language code). FLEURS has no Castellano config: es_419 is Latin American Spanish.
# ca_es is Catalan as spoken in Catalonia, not a Valencian-specific set.
CONFIGS = [("es_419", "es"), ("ca_es", "ca")]
RATE = 16_000
MIN_S, MAX_S = 4.0, 15.0
PER_LANG = 30
SEED = 20261015
LICENSE = "CC-BY-4.0"


def to_16k_mono(data: np.ndarray, rate: int) -> np.ndarray:
    x = data.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if rate != RATE:
        g = math.gcd(rate, RATE)
        x = resample_poly(x, RATE // g, rate // g).astype(np.float32)
    return np.clip(x, -1.0, 1.0)


def read_transcripts(path: str) -> dict:
    """FLEURS TSV, no header: id, file name, raw transcription, normalized transcription, samples, gender."""
    rows = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) >= 4:
                rows[row[1]] = {"fleurs_id": row[0], "raw": row[2], "normalized": row[3]}
    return rows


def build_language(config: str, lang: str, out: Path) -> list:
    tsv = hf_hub_download(REPO, f"data/{config}/test.tsv", repo_type="dataset", revision=REVISION)
    tar_path = hf_hub_download(REPO, f"data/{config}/audio/test.tar.gz", repo_type="dataset", revision=REVISION)
    transcripts = read_transcripts(tsv)
    print(f"{config}: {len(transcripts)} utterances in the test split")

    # Decode only the utterances in the duration window; keep them in memory until the selection is made.
    eligible = {}
    seen = 0
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = os.path.basename(member.name)
            if name not in transcripts:
                continue
            seen += 1
            data, rate = sf.read(io.BytesIO(tar.extractfile(member).read()), always_2d=False)
            samples = to_16k_mono(data, rate)
            seconds = len(samples) / RATE
            if MIN_S <= seconds <= MAX_S:
                eligible[name] = samples
    print(f"{config}: decoded {seen} utterances, {len(eligible)} in {MIN_S}-{MAX_S} s")
    if len(eligible) < PER_LANG:
        raise SystemExit(f"{config}: only {len(eligible)} eligible utterances, need {PER_LANG}")

    rng = random.Random(SEED)
    chosen = sorted(rng.sample(sorted(eligible, key=lambda n: transcripts[n]["fleurs_id"]), PER_LANG),
                    key=lambda n: int(transcripts[n]["fleurs_id"]))

    clips = []
    for name in chosen:
        info = transcripts[name]
        samples = eligible[name]
        clip_id = f"{lang}-{info['fleurs_id']}"
        rel = f"audio/{clip_id}.wav"
        sf.write(out / rel, samples, RATE, subtype="PCM_16")
        clips.append({
            "id": clip_id,
            "file": rel,
            "language": lang,
            "fleurs_config": config,
            "fleurs_id": info["fleurs_id"],
            "duration_ms": round(len(samples) * 1000 / RATE),
            "transcript": info["raw"],
            "transcript_normalized": info["normalized"],
            "license": LICENSE,
        })
    return clips


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reference")
    args = parser.parse_args()
    out = Path(args.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)

    all_clips = []
    for config, lang in CONFIGS:
        clips = build_language(config, lang, out)
        all_clips.extend(clips)
        with open(out / f"transcripts-{lang}.tsv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(["file", "duration_ms", "transcript"])
            for c in clips:
                w.writerow([c["file"], c["duration_ms"], c["transcript"]])
        print(f"{lang}: wrote {len(clips)} clips, {sum(c['duration_ms'] for c in clips) / 60000:.1f} min")

    manifest = {
        "pack": "kngfsh reference clips",
        "built_from": {"dataset": REPO, "revision": REVISION, "split": "test"},
        "format": "16 kHz, mono, PCM16 WAV",
        "license": LICENSE,
        "clips": all_clips,
    }
    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
