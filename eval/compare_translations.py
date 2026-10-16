#!/usr/bin/env python3
"""Compares two translations CSVs of the same sentences (the kngfsh-translate app's export and the output of
eval/translate_opus_mt.py), sentence by sentence and per direction.

    pip install sacrebleu
    python3 eval/compare_translations.py A.csv B.csv [--names A B] [--show 5]

For each direction: corpus chrF, chrF++ and BLEU of both systems, the difference B minus A with a paired bootstrap
(the same sentences are resampled for both systems, so the interval is about the difference and is much tighter than
the two separate intervals), the share of resamples in which B is ahead, and how many sentences each system wins by
sentence-level chrF (a gap under 1 point is a tie). Then the sentences where B gains the most and loses the most.
Only sentences that are `ok` in both files and are not a second recording (-b, -c) are used.
"""
import argparse
import csv
import random
import sys

from sacrebleu.metrics import BLEU, CHRF

GUARDED = "=+-@\t\r"
ROUNDS = 1000
SEED = 20261016
TIE = 1.0
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
    return {r["id"]: r for r in rows if r["status"] == "ok" and not r["id"].endswith(("-b", "-c", "-d"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--names", nargs=2, default=("A", "B"))
    parser.add_argument("--show", type=int, default=5)
    args = parser.parse_args()
    name_a, name_b = args.names
    rows_a, rows_b = read(args.a), read(args.b)
    chrf, chrfpp, bleu = CHRF(), CHRF(word_order=2), BLEU()
    rng = random.Random(SEED)

    only = sorted(set(rows_a) ^ set(rows_b))
    print(f"A = {name_a} ({args.a})\nB = {name_b} ({args.b})")
    if only:
        print(f"  not in both files and left out: {', '.join(only)}")
    for code, title in DIRECTIONS:
        ids = sorted(i for i in rows_a if i in rows_b and rows_a[i]["language"] == code)
        print(f"\n{title} ({code}): {len(ids)} sentences")
        if not ids:
            continue
        for i in ids:  # the two files must hold the same sentences, or the comparison means nothing
            if rows_a[i]["source"] != rows_b[i]["source"] or rows_a[i]["reference"] != rows_b[i]["reference"]:
                sys.exit(f"{i}: the source or reference differs between the files")
        refs = [[rows_a[i]["reference"] for i in ids]]
        hyp_a = [rows_a[i]["hypothesis"] for i in ids]
        hyp_b = [rows_b[i]["hypothesis"] for i in ids]
        for name, hyp in ((name_a, hyp_a), (name_b, hyp_b)):
            print(f"  {name:<10} chrF {chrf.corpus_score(hyp, refs).score:5.1f}   chrF++ {chrfpp.corpus_score(hyp, refs).score:5.1f}   BLEU {bleu.corpus_score(hyp, refs).score:5.1f}")
        n = len(ids)
        diffs = []
        for _ in range(ROUNDS):
            pick = [rng.randrange(n) for _ in range(n)]
            r = [[refs[0][k] for k in pick]]
            diffs.append(chrf.corpus_score([hyp_b[k] for k in pick], r).score - chrf.corpus_score([hyp_a[k] for k in pick], r).score)
        diffs.sort()
        base = chrf.corpus_score(hyp_b, refs).score - chrf.corpus_score(hyp_a, refs).score
        ahead = sum(d > 0 for d in diffs) / ROUNDS
        print(f"  chrF {name_b} minus {name_a}: {base:+.1f}   95% interval {diffs[int(0.025 * ROUNDS)]:+.1f} to {diffs[int(0.975 * ROUNDS) - 1]:+.1f}   {name_b} ahead in {ahead:.0%} of resamples")
        gain = []
        for i in ids:
            a = chrf.sentence_score(rows_a[i]["hypothesis"], [rows_a[i]["reference"]]).score
            b = chrf.sentence_score(rows_b[i]["hypothesis"], [rows_b[i]["reference"]]).score
            gain.append((b - a, i, a, b))
        wins = sum(g > TIE for g, *_ in gain)
        losses = sum(g < -TIE for g, *_ in gain)
        print(f"  by sentence: {name_b} better on {wins}, {name_a} better on {losses}, level on {len(gain) - wins - losses}")
        gain.sort()
        for label, picks in ((f"{name_b} gains the most", gain[::-1][: args.show]), (f"{name_b} loses the most", gain[: args.show])):
            print(f"  {label}:")
            for g, i, a, b in picks:
                print(f"    {i}  {name_a} {a:4.1f} -> {name_b} {b:4.1f}")
                print(f"      source:    {rows_a[i]['source']}")
                print(f"      reference: {rows_a[i]['reference']}")
                print(f"      {name_a}: {rows_a[i]['hypothesis']}")
                print(f"      {name_b}: {rows_b[i]['hypothesis']}")


if __name__ == "__main__":
    main()
