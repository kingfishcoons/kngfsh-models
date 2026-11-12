#!/usr/bin/env python3
"""Scores a translations CSV exported by the app (Diagnostics > Translation quality; DIVERGENCES D89).

R1c of the plan: BLEU and chrF per direction (es to en, ca to en) against the English reference of the same FLEURS
id, with a bootstrap interval and the worst sentences. The decision rule of PLAN "Direction after Phase 1" reads the
chrF column: both directions at about 60 or better parks Opus-MT; only ca to en weak means Phase 2b for that direction.

    pip install sacrebleu
    python3 tools/score_translations.py kngfsh-translations-<epoch>.csv [--worst 8] [--include-repeats]

A clip that is a second recording of the same FLEURS id (`es-1780-b`) has the same transcript, so it has the same
translation; it is left out of the scores unless --include-repeats is given. A failed row (status not "ok") is left out
and counted. A text cell that begins with `'` followed by = + - @ tab or CR carries the app's spreadsheet guard; one
apostrophe is removed from it before scoring.
"""
import argparse
import csv
import random
import sys

from sacrebleu.metrics import BLEU, CHRF

GUARDED = "=+-@\t\r"
THRESHOLD = 60.0  # PLAN: chrF about 0.60, on sacreBLEU's 0 to 100 scale
BOOTSTRAP_ROUNDS = 1000
SEED = 20261016
DIRECTIONS = (("es", "Castellano to English"), ("ca", "Valencian/Catalan to English"))


def unguard(cell):
    return cell[1:] if len(cell) > 1 and cell[0] == "'" and cell[1] in GUARDED else cell


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    needed = {"id", "language", "source", "reference", "hypothesis", "status"}
    if not rows or not needed <= set(rows[0]):
        sys.exit(f"{path}: not a translations CSV (needs the columns {sorted(needed)})")
    for row in rows:
        for key in ("source", "reference", "hypothesis"):
            row[key] = unguard(row[key])
    return rows


def bootstrap_chrf(hyps, refs, rounds=BOOTSTRAP_ROUNDS):
    """The 95 % interval of corpus chrF when the sentences are resampled with replacement."""
    rng = random.Random(SEED)
    chrf = CHRF()
    n = len(hyps)
    scores = sorted(
        chrf.corpus_score([hyps[i] for i in pick], [[refs[i] for i in pick]]).score
        for pick in ([rng.randrange(n) for _ in range(n)] for _ in range(rounds))
    )
    return scores[int(0.025 * rounds)], scores[int(0.975 * rounds) - 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("csv")
    parser.add_argument("--worst", type=int, default=8, help="how many of the lowest-scoring sentences to list per direction")
    parser.add_argument("--include-repeats", action="store_true", help="also score the -b second recordings")
    args = parser.parse_args()

    rows = read(args.csv)
    meta = rows[0]
    print(f"{args.csv}")
    print(f"  {meta.get('engine', '?')} on {meta.get('device', '?')}, {meta.get('app', '?')}, sentences {meta.get('pack', '?')}")
    failed = [r for r in rows if r["status"] != "ok"]
    if failed:
        print(f"  {len(failed)} rows failed and are left out: " + ", ".join(f"{r['id']} ({r['error']})" for r in failed[:5]))

    bleu, chrf, chrfpp = BLEU(), CHRF(), CHRF(word_order=2)
    for code, name in DIRECTIONS:
        group = [r for r in rows if r["language"] == code and r["status"] == "ok"]
        if not args.include_repeats:
            group = [r for r in group if not r["id"].endswith(("-b", "-c", "-d"))]
        print(f"\n{name} ({code}): {len(group)} sentences")
        if not group:
            print("  nothing to score")
            continue
        hyps = [r["hypothesis"] for r in group]
        refs = [r["reference"] for r in group]
        score_chrf = chrf.corpus_score(hyps, [refs]).score
        low, high = bootstrap_chrf(hyps, refs)
        verdict = "at or above" if score_chrf >= THRESHOLD else "below"
        print(f"  chrF   {score_chrf:5.1f}   95% interval {low:.1f} to {high:.1f}   ({verdict} the {THRESHOLD:.0f} line)")
        print(f"  chrF++ {chrfpp.corpus_score(hyps, [refs]).score:5.1f}")
        print(f"  BLEU   {bleu.corpus_score(hyps, [refs]).score:5.1f}   ({bleu.corpus_score(hyps, [refs])})")
        ranked = sorted(group, key=lambda r: chrf.sentence_score(r["hypothesis"], [r["reference"]]).score)
        print(f"  the {min(args.worst, len(ranked))} lowest sentences by chrF:")
        for r in ranked[: args.worst]:
            s = chrf.sentence_score(r["hypothesis"], [r["reference"]]).score
            print(f"    {r['id']}  chrF {s:4.1f}")
            print(f"      source:     {r['source']}")
            print(f"      reference:  {r['reference']}")
            print(f"      translated: {r['hypothesis']}")


if __name__ == "__main__":
    main()
