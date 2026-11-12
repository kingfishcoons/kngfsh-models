#!/usr/bin/env python3
"""Translates the reference pack's transcripts with Opus-MT and writes a CSV in the format the kngfsh-translate app
exports (Diagnostics > Translation quality), so eval/score_translations.py and eval/compare_translations.py read the
output of both the same way.

Opus-MT (Helsinki-NLP, CC-BY-4.0) is the candidate translator of the app's Phase 2b. This is its quality ceiling: the
full-precision PyTorch model, beam search of 4, one sentence at a time. The app would ship an int8 ONNX export of the
same model, so a result worth acting on is checked again on that export. Time per sentence is the runner's CPU, not a
phone's, and is only recorded for completeness.

Run by .github/workflows/opus-mt.yml (the cloud sessions of the app cannot download models; Actions runners can).
"""
import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time

import torch
from huggingface_hub import HfApi
from transformers import MarianMTModel, MarianTokenizer

MODELS = {"es": "Helsinki-NLP/opus-mt-es-en", "ca": "Helsinki-NLP/opus-mt-ca-en"}
COLUMNS = [
    "id", "language", "fleurs_id", "source", "reference", "hypothesis", "status", "error", "translate_ms",
    "acronyms_lost", "engine", "pack", "device", "soc", "android", "app", "at_epoch_ms",
]
GUARDED = "=+-@\t\r"


def guard(cell):
    """The app's spreadsheet guard, so the scorer reads these files exactly as it reads the app's."""
    return "'" + cell if cell and cell[0] in GUARDED else cell


def git_short_sha_of(path):
    out = subprocess.run(["git", "log", "-1", "--format=%h", "--", path], capture_output=True, text=True)
    return out.stdout.strip() or "unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--manifest", default="reference/manifest.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--beams", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    with open(args.manifest, encoding="utf-8") as handle:
        clips = json.load(handle)["clips"]
    pack = "kngfsh-models@" + git_short_sha_of(args.manifest)
    torch.set_num_threads(os.cpu_count() or 2)
    torch.manual_seed(0)
    started = int(time.time() * 1000)
    device = f"GitHub Actions {platform.system()} {platform.machine()}, {os.cpu_count()} CPUs, PyTorch {torch.__version__} fp32"

    rows = []
    for language, repo in MODELS.items():
        revision = HfApi().model_info(repo).sha
        print(f"{repo} at {revision}", flush=True)
        tokenizer = MarianTokenizer.from_pretrained(repo, revision=revision)
        model = MarianMTModel.from_pretrained(repo, revision=revision).eval()
        engine = f"opus-mt:{repo}@{revision[:7]}"
        for clip in (c for c in clips if c["language"] == language):
            row = {
                "id": clip["id"], "language": language, "fleurs_id": clip["fleurs_id"], "source": clip["transcript"],
                "reference": clip["english_reference"], "hypothesis": "", "status": "failed", "error": "",
                "translate_ms": "", "acronyms_lost": 0, "engine": engine, "pack": pack, "device": device,
                "soc": platform.processor() or platform.machine(), "android": "", "app": "eval/translate_opus_mt.py",
                "at_epoch_ms": started,
            }
            try:
                batch = tokenizer([clip["transcript"]], return_tensors="pt")
                begin = time.perf_counter()
                with torch.no_grad():
                    generated = model.generate(**batch, num_beams=args.beams, max_new_tokens=args.max_new_tokens)
                row["translate_ms"] = f"{(time.perf_counter() - begin) * 1000:.1f}"
                row["hypothesis"] = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
                row["status"] = "ok"
            except Exception as error:  # a failed sentence is a row, as in the app's export
                row["error"] = f"{type(error).__name__}: {error}"
            rows.append(row)
        del model, tokenizer

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\r\n")
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow([guard(str(row[c])) if c in ("source", "reference", "hypothesis", "error") else row[c] for c in COLUMNS])
    failed = [r for r in rows if r["status"] != "ok"]
    print(f"{len(rows) - len(failed)} of {len(rows)} sentences translated, written to {args.out}")
    for r in failed:
        print(f"  failed {r['id']}: {r['error']}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
