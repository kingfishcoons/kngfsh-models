#!/usr/bin/env python3
"""Builds the clip sets of the recognizer comparison (asr-compare.yml): 16 kHz mono PCM16 WAVs and one manifest.

Sets (all pinned to a dataset revision; the selection is seeded, so a rebuild picks the same clips):

- fleurs-ca, fleurs-es: the 60 clips of this repository's reference pack (reference/manifest.json; FLEURS, CC-BY-4.0),
  read speech; Catalan as read in Catalonia and Latin American Spanish. The app's own test clips.
- corts: Valencian parliament speech, the clean test split of projecte-aina/corts_valencianes_asr_a (CC-BY-NC-4.0),
  short segments of 3 to 15 s. Real speakers talking, mostly Valencian. One of the fine-tuned models was trained on this
  corpus's train split (other sessions, possibly the same speakers), so it has a head start here.
- codeswitch: BSC-LT/BSCs_Code_Switching_CA-ES_ASR_Test (CC-BY-NC-4.0), hand-checked Catalan-Spanish code-switching from
  the Corts Valencianes (anonymized) and the Parlament de Catalunya, 3 to 15 s, half from each. The language is left to
  the app's gate ("auto").

The audio of the two CC-BY-NC sets is not committed anywhere; the workflow keeps it in an artifact for the run.

    python3 eval/asr/prepare_sets.py --out clips [--per-set 120]
"""
import argparse
import csv
import io
import json
import random
import shutil
import subprocess
import tarfile
import wave
from pathlib import Path

from huggingface_hub import hf_hub_download

CORTS = ("projecte-aina/corts_valencianes_asr_a", "bbd27052cf14a4bcd7d1490bb0b8f658eb2745e8")
CODESWITCH = ("BSC-LT/BSCs_Code_Switching_CA-ES_ASR_Test", "a8618b942a698ce784e1061f6c86b2c86fe92432")
MIN_MS, MAX_MS = 3_000, 15_000
COLUMNS = ["id", "set", "language", "file", "duration_ms", "reference", "source", "license"]


def wav_ms(path):
    with wave.open(str(path)) as w:
        return int(w.getnframes() * 1000 / w.getframerate())


def to_pcm16(src, dst):
    """Copies a WAV that is already 16 kHz mono PCM16; converts anything else with ffmpeg."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with wave.open(str(src)) as w:
            ok = w.getframerate() == 16_000 and w.getnchannels() == 1 and w.getsampwidth() == 2
    except (wave.Error, EOFError):
        ok = False
    if ok:
        shutil.copyfile(src, dst)
    else:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-ac", "1", "-ar", "16000",
                        "-sample_fmt", "s16", str(dst)], check=True)


def extract(archive, wanted, into):
    """Extracts the members of a .tar.gz whose file stem is in wanted; returns {stem: path}."""
    found = {}
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            stem = Path(member.name).stem
            if member.isfile() and stem in wanted:
                target = into / f"{stem}{Path(member.name).suffix}"
                with tar.extractfile(member) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
                found[stem] = target
    return found


def fleurs(reference, out, rows):
    manifest = json.loads(Path(reference).read_text(encoding="utf-8"))
    base = Path(reference).parent
    for clip in manifest["clips"]:
        target = out / "audio" / f"fleurs-{clip['language']}" / f"{clip['id']}.wav"
        to_pcm16(base / clip["file"], target)
        rows.append({
            "id": clip["id"], "set": f"fleurs-{clip['language']}", "language": clip["language"],
            "file": str(target.relative_to(out)), "duration_ms": wav_ms(target), "reference": clip["transcript"],
            "source": f"google/fleurs {clip['fleurs_config']} id {clip['fleurs_id']} (this repository's reference pack)",
            "license": clip["license"],
        })


def corts(out, work, per_set, rng, rows):
    repo, rev = CORTS
    index = hf_hub_download(repo, "corpus/files/clean_test_corts_short.csv", repo_type="dataset", revision=rev)
    archive = hf_hub_download(repo, "corpus/speech/clean_test_short/clean_test_short_1.tar.gz", repo_type="dataset", revision=rev)
    with open(index, encoding="utf-8", newline="") as handle:
        entries = {r["identifier"]: r["text"] for r in csv.DictReader(handle)}
    raw = work / "corts"
    raw.mkdir(parents=True, exist_ok=True)
    found = extract(archive, set(entries), raw)
    candidates = sorted(i for i, p in found.items() if MIN_MS <= wav_ms(p) <= MAX_MS and entries[i].strip())
    rng.shuffle(candidates)
    for clip_id in sorted(candidates[:per_set]):
        target = out / "audio" / "corts" / f"{clip_id}.wav"
        to_pcm16(found[clip_id], target)
        rows.append({
            "id": f"corts-{clip_id[:12]}", "set": "corts", "language": "ca", "file": str(target.relative_to(out)),
            "duration_ms": wav_ms(target), "reference": entries[clip_id],
            "source": f"{repo}@{rev[:7]} clean_test_short {clip_id}", "license": "CC-BY-NC-4.0",
        })


def codeswitch(out, work, per_set, rng, rows):
    repo, rev = CODESWITCH
    index = hf_hub_download(repo, "corpus/files/metadata.tsv", repo_type="dataset", revision=rev)
    with open(index, encoding="utf-8", newline="") as handle:
        meta = [r for r in csv.DictReader(handle, delimiter="\t")
                if MIN_MS <= float(r["duration"]) * 1000 <= MAX_MS and r["normalized_text"].strip()]
    chosen = []
    for source in sorted({r["source"] for r in meta}):
        group = sorted((r for r in meta if r["source"] == source), key=lambda r: r["audio_id"])
        rng.shuffle(group)
        chosen += group[: per_set // 2]
    wanted = {r["audio_id"] for r in chosen}
    raw = work / "codeswitch"
    raw.mkdir(parents=True, exist_ok=True)
    found = {}
    for name in ("corts_valencianes_anonymized", "parlament_parla"):
        archive = hf_hub_download(repo, f"corpus/speech/{name}.tar.gz", repo_type="dataset", revision=rev)
        found.update(extract(archive, wanted, raw))
    for r in sorted(chosen, key=lambda r: r["audio_id"]):
        if r["audio_id"] not in found:
            print(f"codeswitch: {r['audio_id']} not in the archives, left out")
            continue
        target = out / "audio" / "codeswitch" / f"{r['audio_id']}.wav"
        to_pcm16(found[r["audio_id"]], target)
        rows.append({
            "id": f"cs-{r['audio_id'][:12]}", "set": "codeswitch", "language": "auto", "file": str(target.relative_to(out)),
            "duration_ms": wav_ms(target), "reference": r["normalized_text"].strip(),
            "source": f"{repo}@{rev[:7]} {r['source']} {r['audio_id']}", "license": "CC-BY-NC-4.0",
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--reference", default="reference/manifest.json")
    parser.add_argument("--per-set", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20261111)
    args = parser.parse_args()
    out, work = Path(args.out), Path(args.out + "-work")
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    rows = []
    fleurs(args.reference, out, rows)
    corts(out, work, args.per_set, random.Random(args.seed), rows)
    codeswitch(out, work, args.per_set, random.Random(args.seed + 1), rows)
    shutil.rmtree(work)
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("two clips share an id")
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for r in rows:
        writer.writerow({k: str(r[k]).replace("\t", " ").replace("\n", " ") for k in COLUMNS})
    (out / "clips.tsv").write_text(buf.getvalue(), encoding="utf-8")
    for name in sorted({r["set"] for r in rows}):
        group = [r for r in rows if r["set"] == name]
        print(f"{name}: {len(group)} clips, {sum(r['duration_ms'] for r in group) / 60000:.1f} min")


if __name__ == "__main__":
    main()
