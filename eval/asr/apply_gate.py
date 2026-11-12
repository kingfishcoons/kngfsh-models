#!/usr/bin/env python3
"""Gives every clip whose language is "auto" the language the app's gate chose (asr_host gate: the higher of p_es and
p_ca, as the app does when it has to choose), so every model decodes a clip in the same language. Rewrites clips.tsv.

    python3 eval/asr/apply_gate.py --clips clips.tsv --gate languages.tsv
"""
import argparse
import csv
import io


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--clips", required=True)
    parser.add_argument("--gate", required=True)
    args = parser.parse_args()
    with open(args.gate, encoding="utf-8", newline="") as handle:
        gate = {r["id"]: r for r in csv.DictReader(handle, delimiter="\t")}
    with open(args.clips, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields, clips = reader.fieldnames, list(reader)
    chosen = {"es": 0, "ca": 0}
    for c in clips:
        if c["language"] != "auto":
            continue
        g = gate.get(c["id"])
        if g is None:
            raise SystemExit(f"{c['id']}: the gate has no answer for it")
        c["language"] = "es" if float(g["p_es"]) > float(g["p_ca"]) else "ca"
        chosen[c["language"]] += 1
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(clips)
    with open(args.clips, "w", encoding="utf-8") as handle:
        handle.write(buf.getvalue())
    print(f"gate: {chosen['es']} clips decoded as Spanish, {chosen['ca']} as Catalan/Valencian")


if __name__ == "__main__":
    main()
