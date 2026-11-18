#!/usr/bin/env python3
"""Builds the audio streams of the draft-text bake-off (kngfsh-translate D126) and what is known about each.

Three sets, all from this repository's FLEURS reference pack (CC-BY-4.0):

- `clips`: the 60 reference clips, each streamed on its own.
- `mixed`: the mixed phone-test track (D81): 30 clips at the times in `reference/tracks/kngfsh-ref-mixed.csv`
  (Castellano and Valencian alternating, about 2.5 s apart), rebuilt here from the clips, plus 3 s of silence at the end.
- `continuous`: the continuous-speech track (D113), `reference/tracks/kngfsh-ref-continuous.wav`: 15 Castellano then 15
  Valencian sentences 0.4 s apart.

For every stream, `items.json` lists its utterances (clip id, language, reference text, and where its speech starts
and ends in the stream) and the segments the app's detector and segmenter would cut it into at the 8 s cap (Silero
VAD v6.2.0 with the app's thresholds and `Segmenter` rules, as `reference/build_tracks.py` does). The segments are what
the Whisper re-decoding drafter decodes again and again as they grow.

    python3 eval/stream/sets.py --vad silero_vad.onnx --out sets
"""
import argparse
import csv
import json
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reference"))
import build_tracks as bt  # noqa: E402  (Silero, segment(), load(), the app's segmenter settings)

SR = 16_000


def speech_span(vad, pcm16):
    """(on_ms, off_ms) of the speech the app's detector finds: first frame at 0.5 or more, last frame still speaking."""
    probs = vad.probabilities(bt.to_float(pcm16))
    speaking, first, last = False, None, None
    for k, p in enumerate(probs):
        speaking = p >= 0.35 if speaking else p >= 0.5
        if speaking:
            first = k if first is None else first
            last = k
    if first is None:
        return 0, len(pcm16) * 1000 // SR
    return first * bt.FRAME_MS, (last + 1) * bt.FRAME_MS


def segments(vad, pcm16):
    probs = vad.probabilities(bt.to_float(pcm16))
    return [{"start_ms": s, "end_ms": e, "reason": r} for s, e, r in bt.segment(probs, **bt.MODES["normal-8s"])]


def write_wav(path, pcm16):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm16.astype("<i2").tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--vad", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    vad = bt.Silero(args.vad)
    manifest = json.loads((ROOT / "reference/manifest.json").read_text())
    clips = {c["id"]: c for c in manifest["clips"]}
    audio = {cid: bt.load(ROOT / "reference" / c["file"]) for cid, c in clips.items()}
    span = {cid: speech_span(vad, pcm) for cid, pcm in audio.items()}
    items = []

    for cid in sorted(clips):
        c = clips[cid]
        on, off = span[cid]
        items.append({
            "set": "clips", "id": cid, "wav": str(ROOT / "reference" / c["file"]), "duration_ms": len(audio[cid]) * 1000 // SR,
            "utterances": [{"id": cid, "language": c["language"], "reference": c["transcript"], "on_ms": on, "off_ms": off}],
            "segments": segments(vad, audio[cid]),
        })

    rows = list(csv.DictReader(open(ROOT / "reference/tracks/kngfsh-ref-mixed.csv", encoding="utf-8")))
    length = max(int(r["end_ms"]) for r in rows) + 3000
    track = np.zeros(length * SR // 1000, np.int16)
    utterances = []
    for r in rows:
        pcm = audio[r["clip_id"]]
        at = int(r["start_ms"]) * SR // 1000
        if int(r["end_ms"]) - int(r["start_ms"]) != len(pcm) * 1000 // SR:
            raise SystemExit(f"{r['clip_id']}: the mixed track's CSV and the clip disagree on its length")
        track[at:at + len(pcm)] = pcm
        on, off = span[r["clip_id"]]
        utterances.append({"id": r["clip_id"], "language": r["language"], "reference": r["transcript"],
                           "on_ms": int(r["start_ms"]) + on, "off_ms": int(r["start_ms"]) + off})
    write_wav(out / "audio/mixed.wav", track)
    items.append({"set": "mixed", "id": "mixed", "wav": str((out / "audio/mixed.wav").resolve()), "duration_ms": length,
                  "utterances": utterances, "segments": segments(vad, track)})

    path = ROOT / "reference/tracks/kngfsh-ref-continuous.wav"
    track = bt.load(path)
    rows = list(csv.DictReader(open(ROOT / "reference/tracks/kngfsh-ref-continuous.csv", encoding="utf-8")))
    margin = int(bt.MARGIN_S * 1000)
    utterances = [{"id": r["clip_id"], "language": r["language"], "reference": r["transcript"],
                   "on_ms": int(r["start_ms"]) + margin, "off_ms": int(r["end_ms"]) - margin} for r in rows]
    items.append({"set": "continuous", "id": "continuous", "wav": str(path),
                  "duration_ms": len(track) * 1000 // SR, "utterances": utterances, "segments": segments(vad, track)})

    (out / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=1))
    for name in ("clips", "mixed", "continuous"):
        sel = [i for i in items if i["set"] == name]
        print(f"{name}: {len(sel)} streams, {sum(len(i['utterances']) for i in sel)} utterances, "
              f"{sum(len(i['segments']) for i in sel)} segments, {sum(i['duration_ms'] for i in sel) / 1000:.0f} s")


if __name__ == "__main__":
    main()
