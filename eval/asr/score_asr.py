#!/usr/bin/env python3
"""Scores the recognizer comparison (asr-compare.yml): word error rate per set and model, speed, how sure each model
was, and a side-by-side for reading.

    python3 eval/asr/score_asr.py --clips clips.tsv --hyps DIR --out DIR

`DIR/<model>.tsv` are asr_host's decode outputs; `DIR/<model>.json` (optional) say how the model was made. Both sides
are normalized the same way before scoring: lower case, digits spelled out in the clip's language (the Corts and
code-switching references spell numbers; Whisper writes digits), punctuation and hyphens removed, apostrophes and the
Catalan middle dot kept. So a model that writes no punctuation (the fine-tunes) is not rewarded or punished for it here;
the report counts it separately, because the app needs punctuation (sentence ends join and split lines, and both
translators read it).
"""
import argparse
import csv
import json
import re
import statistics
import unicodedata
from pathlib import Path

import jiwer
from num2words import num2words

BASELINE = "turbo-q8_0"
BAD_WER = 0.30
LOW_P = 0.85  # the app's LineConfidence threshold, calibrated on the turbo model
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
KEEP = re.compile(r"[^\w'·\s]")


def spell(match, language):
    text = match.group(0)
    try:
        if "," in text or "." in text:
            whole, frac = re.split(r"[.,]", text, maxsplit=1)
            if len(frac) == 3:  # thousands separator, "2.000"
                return num2words(int(whole + frac), lang=language)
            return f"{num2words(int(whole), lang=language)} coma {' '.join(num2words(int(d), lang=language) for d in frac)}"
        return num2words(int(text), lang=language)
    except (ValueError, NotImplementedError, OverflowError):
        return text


def normalize(text, language):
    text = unicodedata.normalize("NFC", text).lower().replace("’", "'").replace("`", "'").replace("‘", "'")
    text = text.replace("l.l", "l·l").replace("ŀl", "l·l")
    text = NUMBER.sub(lambda m: spell(m, language), text)
    text = text.replace("-", " ").replace("%", " per cent " if language == "ca" else " por ciento ")
    text = KEEP.sub(" ", text)
    text = re.sub(r"(?<!\w)'|'(?!\w)", " ", text)  # quotes, not elisions
    return " ".join(text.split())


def read_tsv(path, raw=False):
    """clips.tsv is written with csv's quoting; asr_host writes its text unquoted (raw)."""
    # A model's text can hold a broken UTF-8 sequence (a multibyte character split across tokens in a loop); it is
    # replaced rather than stopping the report.
    with open(path, encoding="utf-8", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE if raw else csv.QUOTE_MINIMAL))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--clips", required=True)
    parser.add_argument("--hyps", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    clips = {c["id"]: c for c in read_tsv(args.clips)}
    models = sorted(p.stem for p in Path(args.hyps).glob("*.tsv"))
    if BASELINE in models:
        models = [BASELINE] + [m for m in models if m != BASELINE]
    hyps = {m: {h["id"]: h for h in read_tsv(Path(args.hyps) / f"{m}.tsv", raw=True)} for m in models}
    info = {m: json.loads((Path(args.hyps) / f"{m}.json").read_text()) if (Path(args.hyps) / f"{m}.json").exists() else {} for m in models}
    sets = sorted({c["set"] for c in clips.values()})

    rows, report = [], []
    report.append("# Recognizer comparison: Valencian, Catalan, Spanish and code-switching (kngfsh-translate D104)\n")
    report.append("Word error rate after normalizing both sides (lower case, digits spelled out, punctuation removed). "
                  f"'bad' = clips with a word error rate of {BAD_WER:.0%} or more. Speed is on the GitHub runner's CPU "
                  "(4 cores, same threads for every model), as a multiple of the audio's length: a ratio between models, "
                  "not the phone's time. 'punct' = share of clips whose text has any punctuation.\n")
    for m in models:
        if info[m]:
            report.append(f"- **{m}**: {info[m].get('about', '')}")
    report.append("")

    per_model = {}
    for s in sets:
        ids = sorted(i for i, c in clips.items() if c["set"] == s)
        report.append(f"## {s} ({len(ids)} clips, {sum(int(clips[i]['duration_ms']) for i in ids) / 60000:.1f} min)\n")
        report.append("| model | WER | bad clips | median clip WER | speed (x real time) | punct | token_p median | flagged < 0.85 (of them bad) |")
        report.append("|---|---|---|---|---|---|---|---|")
        for m in models:
            refs, outs, clip_wers, ms, audio, punct, ps, flagged, flagged_bad = [], [], [], 0, 0, 0, [], 0, 0
            missing = 0
            for i in ids:
                h = hyps[m].get(i)
                c = clips[i]
                if h is None:
                    missing += 1
                    continue
                language = h["language"] or c["language"]
                ref = normalize(c["reference"], language)
                hyp = normalize(h["text"], language)
                if not ref:
                    continue
                w = jiwer.wer(ref, hyp or "-")
                refs.append(ref)
                outs.append(hyp or "-")
                clip_wers.append(w)
                ms += float(h["ms"])
                audio += int(c["duration_ms"])
                punct += bool(re.search(r"[.,;:?!¿¡]", h["text"]))
                p = float(h["token_p"]) if h["token_p"] else None
                if p is not None:
                    ps.append(p)
                    if p < LOW_P:
                        flagged += 1
                        flagged_bad += w >= BAD_WER
                rows.append({"set": s, "id": i, "model": m, "language": language, "wer": f"{w:.3f}", "token_p": h["token_p"],
                             "ms": h["ms"], "reference": c["reference"], "hypothesis": h["text"]})
            if not refs:
                report.append(f"| {m} | (no output) | | | | | | |")
                continue
            wer = jiwer.wer(refs, outs)
            bad = sum(w >= BAD_WER for w in clip_wers)
            per_model[(s, m)] = {"wer": wer, "clip": dict(zip([i for i in ids if i in hyps[m]], clip_wers))}
            note = f" ({missing} missing)" if missing else ""
            report.append(
                f"| {m}{note} | {wer:.1%} | {bad} ({bad / len(clip_wers):.0%}) | {statistics.median(clip_wers):.1%} | "
                f"{ms / max(1, audio):.2f} | {punct / len(clip_wers):.0%} | "
                f"{statistics.median(ps):.3f} | {flagged} ({flagged_bad}) |" if ps else
                f"| {m}{note} | {wer:.1%} | {bad} | {statistics.median(clip_wers):.1%} | {ms / max(1, audio):.2f} | {punct / len(clip_wers):.0%} | | |"
            )
        if (s, BASELINE) in per_model:
            base = per_model[(s, BASELINE)]["clip"]
            for m in models[1:]:
                if (s, m) not in per_model:
                    continue
                cand = per_model[(s, m)]["clip"]
                shared = [i for i in base if i in cand]
                better = sum(cand[i] <= base[i] - 0.10 for i in shared)
                worse = sum(cand[i] >= base[i] + 0.10 for i in shared)
                report.append(f"\n{m} against {BASELINE}: {better} clips better by 10 points or more, {worse} worse, "
                              f"{len(shared) - better - worse} within 10 points.")
        report.append("")

    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    with open(out / "clips-by-model.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["set"])
        writer.writeheader()
        writer.writerows(rows)

    # For reading: each clip's reference and every model's text, the clips where the models disagree most first.
    side = ["# Side by side (reference, then each model with its word error rate)\n"]
    for s in sets:
        ids = [i for i, c in clips.items() if c["set"] == s]
        def spread(i):
            ws = [per_model[(s, m)]["clip"].get(i) for m in models if (s, m) in per_model]
            ws = [w for w in ws if w is not None]
            return max(ws) - min(ws) if ws else 0
        side.append(f"## {s}\n")
        for i in sorted(ids, key=spread, reverse=True):
            side.append(f"### {i} ({int(clips[i]['duration_ms']) / 1000:.1f} s)")
            side.append(f"    reference      {clips[i]['reference']}")
            for m in models:
                h = hyps[m].get(i)
                w = per_model.get((s, m), {}).get("clip", {}).get(i)
                if h is not None:
                    side.append(f"    {m:<14} {'' if w is None else f'[{w:.0%}] '}{h['text']}")
            side.append("")
    (out / "side-by-side.md").write_text("\n".join(side) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
