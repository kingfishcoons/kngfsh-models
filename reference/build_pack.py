#!/usr/bin/env python3
"""Builds the reference clip pack from the FLEURS test split (google/fleurs, CC-BY-4.0).

Run by .github/workflows/reference.yml. Writes, under --out:
  audio/<lang>-<fleurs id>.wav   16 kHz mono PCM16, one utterance each; a second or third recording of the same
                                 FLEURS id (the same sentence read by another speaker) is <lang>-<fleurs id>-b.wav, -c, ...
  manifest.json                  the clips, their transcripts, English references and attribution
  transcripts-<lang>.tsv         file, duration, the raw transcript and the English reference, one line per clip

The English reference is the FLEURS en_us transcription of the same id (the same sentence, read in English). Where
an id has no English in any en_us split, the English fields are null and the build lists the ids; it does not fail.

Selection is seeded, so a rebuild at the same revision picks the same clips. Transcripts are copied unchanged.
The build fails if two clips share an id or a file, if an id has two different English texts, or if a file or an
English field does not match its manifest row.
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
# The English reference for each id. Looked up in these splits, in this order (test first, the clip's own split).
ENGLISH_CONFIG = "en_us"
ENGLISH_SPLITS = ["test", "dev", "train"]
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


def clip_ids(lang: str, fleurs_ids: list) -> list:
    """One id per chosen utterance, in order: <lang>-<fleurs id>.

    FLEURS gives the same id to every recording of a sentence, and a sentence can be read by more than one speaker.
    The first recording keeps the plain id; later ones get a letter, -b, -c, ..., so no clip overwrites another.
    """
    seen = {}
    ids = []
    for fleurs_id in fleurs_ids:
        n = seen.get(fleurs_id, 0)
        seen[fleurs_id] = n + 1
        if n > 25:
            raise SystemExit(f"{lang}-{fleurs_id}: more than 26 recordings of one FLEURS id")
        ids.append(f"{lang}-{fleurs_id}" + ("" if n == 0 else "-" + chr(ord("a") + n)))
    return ids


def read_english(wanted: set) -> dict:
    """FLEURS id -> (split, English raw transcription), for the ids in `wanted`, over the en_us splits.

    Stops if one id has two different English texts. An id with no English is left out of the result.
    """
    texts = {}
    for split in ENGLISH_SPLITS:
        tsv = hf_hub_download(REPO, f"data/{ENGLISH_CONFIG}/{split}.tsv", repo_type="dataset", revision=REVISION)
        with open(tsv, encoding="utf-8", newline="") as f:
            for row in csv.reader(f, delimiter="\t"):
                if len(row) >= 4 and row[0] in wanted:
                    texts.setdefault(row[0], {}).setdefault(row[2], split)
    conflicts = {i: list(t) for i, t in texts.items() if len(t) > 1}
    if conflicts:
        raise SystemExit("ids with two different English texts: " + "; ".join(f"{i}: {t}" for i, t in sorted(conflicts.items())))
    found = {}
    for i, t in texts.items():
        ((text, split),) = t.items()
        found[i] = (split, text)
    print(f"{ENGLISH_CONFIG}: English found for {len(found)} of {len(wanted)} ids")
    return found


def attach_english(clips: list, english: dict) -> list:
    """Sets english_reference and english_source on each clip (null when there is no English). Returns the ids without."""
    missing = []
    for c in clips:
        hit = english.get(c["fleurs_id"])
        if hit is None:
            c["english_reference"] = None
            c["english_source"] = None
            missing.append(c["id"])
        else:
            split, text = hit
            c["english_reference"] = text
            c["english_source"] = f"google/fleurs {ENGLISH_CONFIG}, id {c['fleurs_id']}, {split}"
    return missing


def verify(out: Path, clips: list) -> None:
    """Fails the build if two clips share an id or a file, if audio/ and the manifest disagree on which files
    exist, if a file's format or length does not match its manifest row, or if an English field is missing or
    empty (null is allowed: no English for that id)."""
    ids = [c["id"] for c in clips]
    files = [c["file"] for c in clips]
    if len(set(ids)) != len(ids):
        raise SystemExit("two clips share an id: " + ", ".join(sorted({i for i in ids if ids.count(i) > 1})))
    if len(set(files)) != len(files):
        raise SystemExit("two clips share a file: " + ", ".join(sorted({f for f in files if files.count(f) > 1})))
    on_disk = {p.name for p in (out / "audio").glob("*.wav")}
    listed = {Path(f).name for f in files}
    if on_disk != listed:
        raise SystemExit(f"audio/ and the manifest disagree: only on disk {sorted(on_disk - listed)}, only in the manifest {sorted(listed - on_disk)}")
    for c in clips:
        info = sf.info(out / c["file"])
        ms = round(info.frames * 1000 / info.samplerate)
        if info.samplerate != RATE or info.channels != 1 or info.subtype != "PCM_16":
            raise SystemExit(f"{c['file']}: expected 16 kHz mono PCM16, found {info.samplerate} Hz, {info.channels} ch, {info.subtype}")
        if abs(ms - c["duration_ms"]) > 1:
            raise SystemExit(f"{c['file']}: the manifest says {c['duration_ms']} ms, the file holds {ms} ms")
        for key in ("english_reference", "english_source"):
            if key not in c:
                raise SystemExit(f"{c['id']}: the manifest row has no {key}")
            if c[key] == "":
                raise SystemExit(f"{c['id']}: {key} is empty; write null when there is no English")
        if (c["english_reference"] is None) != (c["english_source"] is None):
            raise SystemExit(f"{c['id']}: english_reference and english_source must both be null or both set")
    for lang in sorted({c["language"] for c in clips}):
        with open(out / f"transcripts-{lang}.tsv", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            if "english" not in (reader.fieldnames or []):
                raise SystemExit(f"transcripts-{lang}.tsv has no english column")
            rows = list(reader)
        expected = [c for c in clips if c["language"] == lang]
        if len(rows) != len(expected):
            raise SystemExit(f"transcripts-{lang}.tsv has {len(rows)} rows, the manifest has {len(expected)} clips")
        for r, c in zip(rows, expected):
            if r["english"] == "":
                raise SystemExit(f"transcripts-{lang}.tsv: {c['id']} has an empty english cell; write null when there is no English")


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
    for name, clip_id in zip(chosen, clip_ids(lang, [transcripts[n]["fleurs_id"] for n in chosen])):
        info = transcripts[name]
        samples = eligible[name]
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
        print(f"{lang}: wrote {len(clips)} clips, {sum(c['duration_ms'] for c in clips) / 60000:.1f} min")

    missing = attach_english(all_clips, read_english({c["fleurs_id"] for c in all_clips}))
    if missing:
        print("no English in any en_us split (english is null): " + ", ".join(missing))

    for _, lang in CONFIGS:
        clips = [c for c in all_clips if c["language"] == lang]
        with open(out / f"transcripts-{lang}.tsv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(["file", "duration_ms", "transcript", "english"])
            for c in clips:
                w.writerow([c["file"], c["duration_ms"], c["transcript"], "null" if c["english_reference"] is None else c["english_reference"]])
        print(f"{lang}: English for {sum(c['english_reference'] is not None for c in clips)} of {len(clips)} clips")

    verify(out, all_clips)
    print(f"verified: {len(all_clips)} clips, every id and file distinct, every file matches its manifest row, every English field set or null")

    manifest = {
        "pack": "kngfsh reference clips",
        "built_from": {"dataset": REPO, "revision": REVISION, "split": "test", "english_config": ENGLISH_CONFIG},
        "format": "16 kHz, mono, PCM16 WAV",
        "license": LICENSE,
        "clips": all_clips,
    }
    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
