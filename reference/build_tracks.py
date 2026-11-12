#!/usr/bin/env python3
"""Builds the continuous-speech phone-test track from the reference pack, and the segments the app should cut it into.

The phone-test tracks of kngfsh-translate (its DIVERGENCES D81) leave 2.5 s of silence between clips, so every clip
closes its own segment and the app's hard cap is almost never reached (3 of 58 clips). In a meeting it is reached all
the time: half of the segments of the first real meeting ended at the cap (D105). This track tests the cap: the same
30 clips as the mixed track (the first 15 Spanish and the first 15 Catalan clips in id order, without es-1780 and
ca-1992), each trimmed to its speech and joined to the next with a short pause, so the speech runs on unbroken:

    4 s silence | 15 es clips, 0.4 s apart | 4 s silence | 15 ca clips, 0.4 s apart | 3 s silence

A clip is trimmed to the speech the app's detector finds in it, plus 0.1 s either side (as kngfsh-translate's
`tools/segmentation_sim.py` does), so two sentences are about 0.6 s of quiet apart: under the 1.5 s pause that always
ends a segment, and at the 0.6 s that ends one only after its 6 s minimum. The 4 s between the languages ends the
Spanish monologue, so the language gate sees one language per segment except by its own mistakes.

Outputs, in --out:

- `kngfsh-ref-continuous.wav`: the track, 16 kHz mono PCM16.
- `kngfsh-ref-continuous.csv`: one row per clip: where its kept audio sits in the track, the transcript and English.
- `kngfsh-ref-continuous-segments.csv`: the segments the app's detector and segmenter cut the track into, for the
  normal mode at the 8 s cap (the app since build 53), at the 12 s cap (before it) and for live mode (6 s), with the
  clips each segment covers and whether it starts or ends inside a sentence. This is the clean signal; a phone hears
  the track through a speaker and a room, so its cuts will move.

The detector is Silero VAD v6.2.0 (the onnx file inside the silero-vad 6.2.0 wheel; the app runs the same model version
through whisper.cpp's ggml conversion, so its probabilities can differ slightly) with the app's thresholds (speech from 0.5, until it falls below 0.35, 512-sample windows); the segmenter follows
`core/segmenter/Segmenter.kt` with its adaptive minimum at its floor. Rebuilding from the same pack and model gives
the same bytes.

    pip install onnxruntime==1.29.0 numpy
    pip download --no-deps silero-vad==6.2.0 && unzip -j silero_vad-6.2.0-py3-none-any.whl silero_vad/data/silero_vad.onnx
    python3 reference/build_tracks.py --vad silero_vad.onnx --out reference/tracks
"""
import argparse
import csv
import json
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort

SR = 16_000
WINDOW = 512
CONTEXT = 64  # samples of the previous window that Silero v5/v6 expects in front of each window at 16 kHz
FRAME_MS = WINDOW * 1000 // SR  # 32
LEFT_OUT = {"es-1780", "ca-1992"}  # not on the phone-test tracks (D81)
PER_LANGUAGE = 15
PAUSE_S = 0.4
MARGIN_S = 0.1
LEAD_S, BETWEEN_S, TAIL_S = 4.0, 4.0, 3.0
NAME = "kngfsh-ref-continuous"
VAD_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"  # silero-vad 6.2.0, silero_vad.onnx

# Segmenter settings (core/segmenter/Segmenter.kt): the normal mode at today's and the earlier cap, and live mode.
MODES = {
    "normal-8s": dict(min_speech=300, preferred_min=6000, long_pause=1500, end_silence=600, cap=8000),
    "normal-12s": dict(min_speech=300, preferred_min=6000, long_pause=1500, end_silence=600, cap=12000),
    "live-6s": dict(min_speech=300, preferred_min=2500, long_pause=1000, end_silence=400, cap=6000),
}
INSIDE_S = 0.3  # a cut further than this inside a clip's speech is a cut inside a sentence


class Silero:
    def __init__(self, model):
        self.session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])

    def probabilities(self, pcm):
        state = np.zeros((2, 1, 128), np.float32)
        context = np.zeros((1, CONTEXT), np.float32)
        out = []
        for i in range(0, len(pcm) - WINDOW + 1, WINDOW):
            window = pcm[i:i + WINDOW][None, :]
            p, state = self.session.run(
                None, {"input": np.concatenate([context, window], axis=1), "state": state, "sr": np.array(SR, np.int64)}
            )
            out.append(float(p[0][0]))
            context = window[:, -CONTEXT:]
        return out


def segment(probabilities, min_speech, preferred_min, long_pause, end_silence, cap):
    """Segmenter.onFrame over SpeechHysteresis decisions; returns (start_ms, end_ms, reason) per kept segment."""
    segments, is_open, speaking = [], False, False
    start = last = speech = 0
    for k, p in enumerate(probabilities):
        speaking = p >= 0.35 if speaking else p >= 0.5
        frame_start, frame_end = k * FRAME_MS, (k + 1) * FRAME_MS
        if not is_open:
            if speaking:
                is_open, start, last, speech = True, frame_start, frame_end, FRAME_MS
            continue
        if speaking:
            speech += FRAME_MS
            last = frame_end
        silence = frame_end - last
        if frame_end - start >= cap:
            reason = "HARD_CAP"
        elif silence >= long_pause:
            reason = "LONG_PAUSE"
        elif silence >= end_silence and last - start >= preferred_min:
            reason = "SILENCE"
        else:
            continue
        if speech >= min_speech:
            segments.append((start, last, reason))
        is_open = False
    if is_open and speech >= min_speech:
        segments.append((start, last, "FLUSH"))
    return segments


def load(path):
    with wave.open(str(path)) as w:
        if w.getframerate() != SR or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise SystemExit(f"{path}: expected 16 kHz mono PCM16")
        return np.frombuffer(w.readframes(w.getnframes()), np.int16)


def to_float(pcm16):
    return pcm16.astype(np.float32) / 32768


def track_clips(manifest):
    """The mixed track's clips (D81): the first 15 of each language in id order, second recordings and D72's two ids
    left out; Spanish first."""
    clips = [c for c in manifest["clips"] if not c["id"].endswith("-b") and c["id"] not in LEFT_OUT]
    by = lambda language: sorted((c for c in clips if c["language"] == language), key=lambda c: c["id"])
    return by("es")[:PER_LANGUAGE], by("ca")[:PER_LANGUAGE]


def trim(vad, pcm16):
    """The clip's speech as the app's detector and segmenter see it with silence around it, plus MARGIN_S either side."""
    pad = np.zeros(int(2.5 * SR), np.int16)
    segments = segment(vad.probabilities(to_float(np.concatenate([pad, pcm16, pad]))), **MODES["normal-12s"])
    if not segments:
        raise SystemExit("a clip holds no speech the detector finds")
    first = segments[0][0] / 1000 - 2.5 - MARGIN_S
    last = segments[-1][1] / 1000 - 2.5 + MARGIN_S
    a, b = max(0, int(round(first * SR))), min(len(pcm16), int(round(last * SR)))
    return pcm16[a:b], a


def build(vad, pack):
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    es, ca = track_clips(manifest)
    parts, rows, t = [np.zeros(int(LEAD_S * SR), np.int16)], [], int(LEAD_S * SR)
    for block, gap_after in ((es, BETWEEN_S), (ca, TAIL_S)):
        for i, clip in enumerate(block):
            kept, offset = trim(vad, load(pack / clip["file"]))
            rows.append({
                "playlist": NAME, "clip_id": clip["id"], "language": clip["language"],
                "start_ms": round(t * 1000 / SR), "end_ms": round((t + len(kept)) * 1000 / SR),
                "trimmed_from_ms": round(offset * 1000 / SR), "kept_ms": round(len(kept) * 1000 / SR),
                "transcript": clip["transcript"], "english_reference": clip["english_reference"] or "",
            })
            parts.append(kept)
            t += len(kept)
            pause = PAUSE_S if i < len(block) - 1 else gap_after
            parts.append(np.zeros(int(round(pause * SR)), np.int16))
            t += int(round(pause * SR))
    return np.concatenate(parts), rows


def cut_segments(vad, track, rows):
    probabilities = vad.probabilities(to_float(track))
    spans = [(r["start_ms"] + MARGIN_S * 1000, r["end_ms"] - MARGIN_S * 1000, r["clip_id"]) for r in rows]
    inside = lambda ms: any(a + INSIDE_S * 1000 < ms < b - INSIDE_S * 1000 for a, b, _ in spans)
    out = []
    for mode, rules in MODES.items():
        for n, (a, b, reason) in enumerate(segment(probabilities, **rules), 1):
            covered = [cid for s, e, cid in spans if s < b and e > a]
            out.append({
                "mode": mode, "segment": n, "start_ms": a, "end_ms": b, "length_ms": b - a, "reason": reason,
                "clips": " ".join(covered), "starts_inside": int(inside(a)), "ends_inside": int(inside(b)),
            })
    return out


def summary(segments):
    lines = []
    for mode in MODES:
        rows = [s for s in segments if s["mode"] == mode]
        caps = sum(s["reason"] == "HARD_CAP" for s in rows)
        cut = sum(1 for s in rows if s["starts_inside"] or s["ends_inside"])
        lengths = sorted(s["length_ms"] for s in rows)
        median = lengths[len(lengths) // 2] / 1000
        lines.append(f"{mode:10s} {len(rows):3d} segments, {caps:3d} at the hard cap, {cut:3d} cut inside a sentence, "
                     f"median length {median:.1f} s")
    return "\n".join(lines)


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--vad", type=Path, required=True, help="silero_vad.onnx from the silero-vad 6.2.0 wheel")
    parser.add_argument("--pack", type=Path, default=Path(__file__).resolve().parent, help="the reference pack")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "tracks")
    args = parser.parse_args()

    import hashlib
    digest = hashlib.sha256(args.vad.read_bytes()).hexdigest()
    if digest != VAD_SHA256:
        raise SystemExit(f"{args.vad}: sha256 {digest}, expected silero-vad 6.2.0's {VAD_SHA256}")
    vad = Silero(args.vad)
    track, rows = build(vad, args.pack)
    segments = cut_segments(vad, track, rows)

    args.out.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.out / f"{NAME}.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(track.tobytes())
    write_csv(args.out / f"{NAME}.csv", rows)
    write_csv(args.out / f"{NAME}-segments.csv", segments)
    print(f"{NAME}.wav: {len(track) / SR:.1f} s, {len(rows)} clips")
    print(summary(segments))


if __name__ == "__main__":
    main()
