#!/usr/bin/env python3
"""Writes ML Kit's English and each candidate's English next to each other, for reading: the perfect transcripts
(the app's Translation quality CSVs) and the lines of the phone sessions (session exports whose `english` column holds
each system's output). A score cannot say which lines turned into nonsense or changed their meaning; a reader can.

    python3 eval/side_by_side.py --mlkit eval/results/mlkit-oneplus13-1792158917008.csv \\
        --candidate NAME=translations.csv=sessions_dir [...] --sessions eval/sessions --out side-by-side.txt
"""
import argparse
import csv
import glob
import os
from pathlib import Path

GUARDED = "=+-@\t\r"


def unguard(cell):
    return cell[1:] if len(cell) > 1 and cell[0] == "'" and cell[1] in GUARDED else cell


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mlkit", required=True)
    parser.add_argument("--candidate", action="append", required=True, help="NAME=translations.csv=sessions_dir")
    parser.add_argument("--sessions", default="eval/sessions")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    candidates = [c.split("=", 2) for c in args.candidate]
    width = max(len("reference"), *(len(n) for n, _, _ in candidates)) + 1

    out = ["# Perfect transcripts (the 58 sentences of R1; second recordings left out)", ""]
    mlkit = {r["id"]: r for r in read(args.mlkit)}
    others = [(name, {r["id"]: r for r in read(path)}) for name, path, _ in candidates]
    for sentence_id, row in mlkit.items():
        if sentence_id.endswith("-b"):
            continue
        out.append(f"## {sentence_id}")
        out.append(f"{'source':<{width}} {unguard(row['source'])}")
        out.append(f"{'reference':<{width}} {unguard(row['reference'])}")
        out.append(f"{'ML Kit':<{width}} {unguard(row['hypothesis'])}")
        for name, rows in others:
            out.append(f"{name:<{width}} {unguard(rows[sentence_id]['hypothesis']) if sentence_id in rows else '(missing)'}")
        out.append("")

    out += ["# Phone sessions (the text the recognizer heard, in the language the app decoded it in)", ""]
    for path in sorted(glob.glob(os.path.join(args.sessions, "kngfsh-*.csv"))):
        name = Path(path).name
        session = read(path)
        theirs = [(n, {r["line"]: r for r in read(os.path.join(d, name))}) for n, _, d in candidates]
        out.append(f"## {session[0]['session']} ({session[0]['mode']})")
        for line in session:
            if line["status"] != "DONE" or not unguard(line["source"]).strip():
                continue
            out.append(f"{line['line']} [{line['language']}] {unguard(line['source'])}")
            out.append(f"  {'ML Kit':<{width}} {unguard(line['english'])}")
            for n, rows in theirs:
                out.append(f"  {n:<{width}} {unguard(rows[line['line']]['english']) if line['line'] in rows else '(missing)'}")
        out.append("")
    Path(args.out).write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
