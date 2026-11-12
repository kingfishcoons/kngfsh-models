#!/usr/bin/env python3
"""Scores the English of live sessions of the reference tracks, end to end (DIVERGENCES D100).

R1 (`score_translations.py`) scores ML Kit on the pack's perfect transcripts. This scores what the app showed: each
line of a session's table export is matched to the pack clip it came from by its text (character 3-grams of the
recognized source against each transcript), the lines of a clip are joined, and the clip gets:

- `wer`: the recognized source against the transcript (lower case, punctuation removed);
- `chrf`: the English shown against the clip's English reference;
- `clean`: the chrF of ML Kit's translation of the perfect transcript (the R1 CSV), when given;
- causes: `lang` (a line decoded in the other language), `split` (the clip came out as more than one line),
  `extra` (a line holds text the clip does not have: a loop's leftovers, an invented ending, a prompt leak),
  `asr` (WER 15 % or more), none of these: `ok`.

Lines that match no clip (a "Gracias." tail, a prompt leak, a runaway) are listed as junk; a line of fewer than four
words can only belong to the clip of the line before it. The clip figures leave out what a clip-by-clip match cannot
see: a clip with no line of its own (lost, or swallowed by a merged line) and the junk lines. The whole-session line
counts them: the track's clip order (D81) gives the clips played up to the last one heard, and all the English shown
is scored as one text against all their references, beside ML Kit's perfect-transcript English for the same clips.

    pip install sacrebleu jiwer
    python3 tools/score_session.py kngfsh-<session>.csv [more sessions] \\
        [--r1 docs/measurements/r1-translations-oneplus13-1792158917008.csv] [--worst 6]
"""
import argparse
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

import jiwer
from sacrebleu.metrics import CHRF

REFERENCE = Path(__file__).resolve().parent.parent / "app/src/main/assets/bench/reference-sentences.json"
GUARDED = "=+-@\t\r"
MATCH_MIN = 0.5  # share of a line's 3-grams that must be in a transcript for the line to belong to that clip
EXTRA_BELOW = 0.8  # a matched line with less than this share in its clip carries text the clip does not have
SHORT_WORDS = 4  # a line of fewer words is matched only to the clip of the line before it
TRACK_LEFT_OUT = {"es-1780", "ca-1992"}  # not on the phone-test tracks (D81)
ASR_WER = 0.15
CHRF_METRIC = CHRF()


def unguard(cell):
    return cell[1:] if len(cell) > 1 and cell[0] == "'" and cell[1] in GUARDED else cell


def fold(text):
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def words(text):
    return re.sub(r"[^\w]+", " ", text.lower()).split()


def grams(text):
    t = f" {fold(text)} "
    return {t[i:i + 3] for i in range(len(t) - 2)}


def chrf(hyp, ref):
    return CHRF_METRIC.sentence_score(hyp, [ref]).score


def load_clips(path=REFERENCE):
    """The app's bundled reference sentences, or the pack's `reference/manifest.json` (kngfsh-models)."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if "sentences" in document:
        sentences = document["sentences"]
    else:
        sentences = [{"id": c["id"], "language": c["language"], "source": c["transcript"],
                      "reference": c["english_reference"]} for c in document["clips"]]
    clips = [s for s in sentences if not s["id"].endswith("-b")]  # second recordings share the transcript
    for c in clips:
        c["grams"] = grams(c["source"])
    return clips


def track_clips(clips, per_clip):
    """The clips the session's track played, in order, up to the last one a line matched (D81: the es and ca tracks
    are every clip of the language in id order without es-1780 and ca-1992; the mixed track alternates the first 15
    of each, es first). Empty when the matched clips fit no track."""
    def by_language(language):
        return sorted((c for c in clips if c["language"] == language and c["id"] not in TRACK_LEFT_OUT),
                      key=lambda c: c["id"])
    es, ca = by_language("es"), by_language("ca")
    mixed = [c for pair in zip(es[:15], ca[:15]) for c in pair]
    heard = set(per_clip)
    for track in (es, ca, mixed):
        ids = [c["id"] for c in track]
        if heard and heard <= set(ids):
            last = max(ids.index(h) for h in heard)
            return track[:last + 1]
    return []


def load_r1(path):
    if not path:
        return {}
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return {r["id"]: unguard(r["hypothesis"]) for r in csv.DictReader(handle) if r["status"] == "ok"}


def score_session(path, clips, r1, worst):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "source" not in rows[0]:
        sys.exit(f"{path}: not a session table export")
    session = rows[0]["session"]
    dropped = [r for r in rows if r["status"] == "DROPPED"]
    junk, per_clip, shown, previous = [], {}, [], None
    for r in rows:
        source = unguard(r["source"])
        if r["status"] != "DONE" or not source.strip():
            continue
        shown.append(r)
        g = grams(source)
        if len(words(source)) < SHORT_WORDS:
            # Three 3-grams of "Gracias." can be found in almost any sentence, so a short line only belongs to the
            # clip of the line before it (the tail of a cut sentence), and is junk otherwise.
            best = previous
            share = len(g & best["grams"]) / max(1, len(g)) if best else 0.0
        else:
            best, share = None, 0.0
            for c in clips:
                s = len(g & c["grams"]) / max(1, len(g))
                if s > share:
                    best, share = c, s
        if share < MATCH_MIN:
            junk.append(r)
            continue
        previous = best
        per_clip.setdefault(best["id"], []).append((r, share))

    played = track_clips(clips, per_clip)
    lost = [c for c in played if c["id"] not in per_clip]
    results = []
    for clip in clips:
        lines = per_clip.get(clip["id"])
        if not lines:
            continue
        source = " ".join(unguard(r["source"]) for r, _ in lines)
        english = " ".join(unguard(r["english"]) for r, _ in lines)
        wer = jiwer.wer(" ".join(words(clip["source"])), " ".join(words(source)) or "-")
        causes = []
        if any(r["language"] != clip["language"] for r, _ in lines):
            causes.append("lang")
        if len(lines) > 1:
            causes.append("split")
        if any(s < EXTRA_BELOW for _, s in lines):
            causes.append("extra")
        if wer >= ASR_WER:
            causes.append("asr")
        results.append({
            "clip": clip, "lines": lines, "source": source, "english": english, "wer": wer,
            "chrf": chrf(english, clip["reference"]),
            "clean": chrf(r1[clip["id"]], clip["reference"]) if clip["id"] in r1 else None,
            "causes": causes or ["ok"],
        })

    modes = sorted({r["mode"] for r in rows})
    print(f"\n== {session} ({Path(path).name}): {len(rows)} lines, mode {', '.join(modes)}; "
          f"{len(results)} clips matched, {len(dropped)} dropped, {len(junk)} junk lines")
    if played:
        # Document level: all English shown, in order, against the references of every clip the track played up to
        # the last one heard. Character n-gram counts do not care where lines were cut, so merged, split, lost and
        # junk lines all count, which the clip-level figures below cannot do.
        refs = " ".join(c["reference"] for c in played)
        live = chrf(" ".join(unguard(r["english"]) for r in shown), refs)
        clean = chrf(" ".join(r1.get(c["id"], "") for c in played), refs) if r1 else float("nan")
        print(f"  whole session: {len(played)} clips played, {len(lost)} with no line of their own "
              f"({', '.join(c['id'] for c in lost) or 'none'}); chrF live {live:5.1f}  clean {clean:5.1f}  "
              f"loss {clean - live:5.1f}")
    for language in ("es", "ca"):
        rs = [x for x in results if x["clip"]["language"] == language]
        if not rs:
            continue
        live = CHRF_METRIC.corpus_score([x["english"] for x in rs], [[x["clip"]["reference"] for x in rs]]).score
        both = [x for x in rs if x["clean"] is not None]
        clean = CHRF_METRIC.corpus_score([r1[x["clip"]["id"]] for x in both],
                                         [[x["clip"]["reference"] for x in both]]).score if both else float("nan")
        refs = " ".join(" ".join(words(x["clip"]["source"])) for x in rs)
        hyps = " ".join(" ".join(words(x["source"])) for x in rs)
        print(f"  {language}: {len(rs):2d} clips  chrF live {live:5.1f}  clean (R1, same clips) {clean:5.1f}  "
              f"loss {clean - live:5.1f}   WER {100 * jiwer.wer(refs, hyps):4.1f} %")
    tally = {}
    for x in results:
        for cause in x["causes"]:
            t = tally.setdefault(cause, [0, 0.0])
            t[0] += 1
            t[1] += (x["clean"] - x["chrf"]) if x["clean"] is not None else 0.0
    print("  causes (clips; mean chrF lost against clean): " + ", ".join(
        f"{c} {n} ({s / n:+.1f})" for c, (n, s) in sorted(tally.items(), key=lambda kv: -kv[1][0])))
    for r in junk:
        print(f"  junk {r['line']} [{r['language']}, {r['flags'] or 'no flags'}]: {unguard(r['source'])[:90]!r} -> "
              f"{unguard(r['english'])[:70]!r}")
    for r in dropped:
        print(f"  dropped {r['line']} ({r['drop_reason']}): {unguard(r['raw_text'])[:60]!r}")
    ranked = sorted((x for x in results if x["clean"] is not None), key=lambda x: x["chrf"] - x["clean"])
    for x in ranked[:worst]:
        print(f"  {x['clip']['id']} chrF {x['chrf']:.0f} (clean {x['clean']:.0f}) WER {100 * x['wer']:.0f} % "
              f"[{' '.join(x['causes'])}] lines {' '.join(r['line'] for r, _ in x['lines'])}")
        print(f"      heard:  {x['source'][:200]}")
        print(f"      shown:  {x['english'][:200]}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sessions", nargs="+", help="table exports (kngfsh-<session>.csv)")
    parser.add_argument("--r1", help="the R1 translations CSV, for the clean-transcript comparison")
    parser.add_argument("--worst", type=int, default=6)
    parser.add_argument("--reference", default=REFERENCE,
                        help="reference sentences: the app's bench JSON (default) or the pack's manifest.json")
    args = parser.parse_args()
    clips, r1 = load_clips(args.reference), load_r1(args.r1)
    for path in args.sessions:
        score_session(path, clips, r1, args.worst)


if __name__ == "__main__":
    main()
