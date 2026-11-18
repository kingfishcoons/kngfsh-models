#!/usr/bin/env python3
"""Scores the draft-text bake-off (kngfsh-translate D127): how soon each drafter shows words, how often it changes
what it showed, how right the words are, and what it costs.

    python3 eval/stream/score_stream.py --items sets/items.json --results DIR --models eval/stream/models.json \
        --large-v3 eval/results/asr/large-v3-q8_0.tsv --out eval/results/stream

DIR holds `<engine>.jsonl` from run_stream.py (streaming engines) and `whisper-<model>.tsv` from `asr_host redecode`
(the Whisper re-decoding drafters). Every engine becomes a list of (display time, shown text) per stream:

- **Streaming engines:** chunk i (100 ms) arrives at (i + 1) x 100 ms and is processed once the previous chunk is done,
  taking the wall time it took on one runner core; its text shows when it is done.
- **Whisper re-decoding:** each segment the app would cut (8 s cap) is decoded as it grows, at most once a second. A
  pass starts when the drafter is free and uses all the audio of the segment up to the last whole second (or the
  segment's end); its cost is the measured wall time (base, one core) or Qualcomm's NPU figures for turbo (312 ms + 6.76
  ms a token). `raw` shows each pass's text; `la2` shows only the words the last two passes agree on (LocalAgreement-2,
  Whisper-Streaming), and the whole text after the segment's last pass. The language of each segment is given, as a
  locked chip would.

Measures, per engine and set:

- **WER**: word error of the final text against the references (normalized as eval/asr/score_asr.py does), and on
  the clips also against large-v3's text from D104 (how far the draft is from the line that will replace it).
- **appear lag**: for each word of the final text matched to a reference word, the time its position was first shown
  minus when the reference word was said; **settle lag**: the same from when it stopped changing. When a reference word
  was said is estimated by spreading the utterance's words evenly over its detected speech, so single words carry about
  a word's error; medians over hundreds of words are what to read.
- **first words**: on the clips, from the start of speech to the first shown text.
- **revisions**: shown words later taken back, per 100 final words.
- **CPU**: CPU seconds per second of audio on one runner core (an x86 runner core, not a phone core), and the
  largest queue (how far it fell behind real time).
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "asr"))
from score_asr import normalize  # noqa: E402

NPU_ENCODER_MS, NPU_TOKEN_MS = 312.0, 6.76


def words(text, language):
    return normalize(text or "", language).split()


def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def align(hyp, ref):
    """Levenshtein alignment: for each hyp index the ref index it matches or substitutes (or None), and the counts."""
    n, m = len(hyp), len(ref)
    prev = list(range(m + 1))
    back = [[0] * (m + 1) for _ in range(n + 1)]
    for j in range(1, m + 1):
        back[0][j] = 2
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        back[i][0] = 1
        h = hyp[i - 1]
        for j in range(1, m + 1):
            sub = prev[j - 1] + (h != ref[j - 1])
            ins = prev[j] + 1
            dele = cur[j - 1] + 1
            best = min(sub, ins, dele)
            cur[j] = best
            back[i][j] = 0 if best == sub else (1 if best == ins else 2)
        prev = cur
    i, j, to_ref, ops = n, m, [None] * n, []
    while i > 0 or j > 0:
        b = back[i][j]
        if i > 0 and j > 0 and b == 0:
            to_ref[i - 1] = j - 1
            ops.append(("S" if hyp[i - 1] != ref[j - 1] else "C", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and (j == 0 or b == 1):
            ops.append(("I", i - 1, None))
            i -= 1
        else:
            ops.append(("D", None, j - 1))
            j -= 1
    return to_ref, list(reversed(ops))


def streaming_timeline(rec, duration_ms):
    done, events = 0.0, []
    changes = {i: t for i, t in rec["changes"]}
    max_queue = 0.0
    for i, w in enumerate(rec["wall_ms"]):
        arrive = (i + 1) * rec["chunk_ms"]
        done = max(arrive, done) + w
        max_queue = max(max_queue, done - arrive)
        if i in changes:
            events.append((done, changes[i]))
    done = max(duration_ms, done) + rec["finish_wall_ms"]
    events.append((done, rec["final"]))
    cpu_s = (sum(rec["cpu_ms"]) + rec["finish_cpu_ms"]) / 1000
    per = 30_000 // rec["chunk_ms"]  # 30 s windows
    cpu = rec["cpu_ms"]
    peak = max((sum(cpu[i:i + per]) / 30_000 for i in range(0, max(1, len(cpu) - per + 1), per // 3)), default=0.0)
    return events, cpu_s, max_queue, peak


def looped(n_tokens, seconds):
    """A pass that wrote far more tokens than the audio can hold (a repetition loop): more than 8 + 6 a second. Normal
    speech in the reference sets runs at about 3 to 5 tokens a second."""
    return n_tokens > 8 + 6 * seconds


def whisper_timeline(passes, segments, mode, cost):
    """passes: {seg index: [(point_ms, ms, n_tokens, text)]}; returns the display events, the busy seconds, the
    largest queue and the number of passes dropped as loops. A looped pass costs its time but changes nothing shown
    (a drafter would need the same guard the app has); a segment whose last pass looped keeps its last good text."""
    events, free, finals, busy, max_queue, dropped = [], 0.0, [], 0.0, 0.0, 0
    for s, seg in enumerate(segments):
        plist = sorted(passes.get(s, []))
        if not plist:
            continue
        j, history, shown = 0, [], ""
        while j < len(plist):
            t = max(free, plist[j][0])
            k = max(x for x in range(j, len(plist)) if plist[x][0] <= t)
            point, ms, n_tok, text = plist[k]
            c = ms if cost == "measured" else NPU_ENCODER_MS + NPU_TOKEN_MS * n_tok
            free = t + c
            busy += c
            max_queue = max(max_queue, free - point)
            last = k == len(plist) - 1
            if looped(n_tok, (point - seg["start_ms"]) / 1000):
                dropped += 1
                if last:
                    shown = history[-1] if history else ""
            else:
                history.append(text)
                if mode == "raw" or last:
                    shown = text
                else:
                    a, b = (history[-2].split() if len(history) > 1 else []), text.split()
                    shown = " ".join(b[:lcp(a, b)])
            events.append((free, " ".join(finals + ([shown] if shown else []))))
            j = k + 1
        finals.append(shown)
    return events, busy / 1000, max_queue, dropped


def score_item(item, events, lang_of_text):
    """Lags, WER pieces, revisions for one stream. events: [(time_ms, text)] in time order, the last being final."""
    utts = item["utterances"]
    ref, ref_utt, ref_time = [], [], []
    for u_i, u in enumerate(utts):
        ws = words(u["reference"], u["language"])
        for p, w in enumerate(ws):
            ref.append(w)
            ref_utt.append(u_i)
            ref_time.append(u["on_ms"] + (p + 1) / len(ws) * (u["off_ms"] - u["on_ms"]))
    shown = [(t, words(x, lang_of_text)) for t, x in events]
    final = shown[-1][1]
    to_ref, ops = align(final, ref)
    # Per-utterance errors; an insertion goes to the utterance of the reference word before it (or the first).
    per = [{"S": 0, "D": 0, "I": 0, "N": 0} for _ in utts]
    for r, u_i in enumerate(ref_utt):
        per[u_i]["N"] += 1
    last_u = 0
    for op, h, r in ops:
        if r is not None:
            last_u = ref_utt[r]
        if op in ("S", "D", "I"):
            per[last_u][op] += 1
    # appear and settle times per final word
    appear = [None] * len(final)
    for t, ws in shown:
        for k in range(len(ws)):
            if k < len(final) and appear[k] is None:
                appear[k] = t
    settle = [None] * len(final)
    run_min = len(final)
    for idx in range(len(shown) - 1, -1, -1):
        t, ws = shown[idx]
        run_min = min(run_min, lcp(ws, final))
        for k in range(run_min):
            settle[k] = t
    appear_lag, settle_lag = [], []
    for k, r in enumerate(to_ref):
        if r is None or appear[k] is None:
            continue
        appear_lag.append(appear[k] - ref_time[r])
        settle_lag.append((settle[k] if settle[k] is not None else shown[-1][0]) - ref_time[r])
    revisions = sum(len(a) - lcp(a, b) for (_, a), (_, b) in zip(shown, shown[1:]))
    first = None
    if len(utts) == 1:
        nonempty = [t for t, ws in shown if ws]
        first = (nonempty[0] - utts[0]["on_ms"]) if nonempty else None
    return {"per": per, "appear": appear_lag, "settle": settle_lag, "revisions": revisions, "final_words": len(final),
            "first": first, "final": events[-1][1]}


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(round(q / 100 * len(xs) + 0.5)) - 1))]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--items", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--large-v3", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    items = json.loads(Path(args.items).read_text())
    models = json.loads(Path(args.models).read_text())
    by_id = {(i["set"], i["id"]): i for i in items}
    large = {}
    with open(args.large_v3, encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            large[row["id"]] = row["text"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    engines = {}  # name -> list of (item, events, cpu_s, max_queue)
    loops = {}  # whisper drafter -> set -> passes dropped as loops
    for path in sorted(Path(args.results).glob("*.jsonl")):
        name = path.stem
        recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for rec in recs:
            item = by_id[(rec["set"], rec["id"])]
            events, cpu_s, q, peak = streaming_timeline(rec, item["duration_ms"])
            engines.setdefault(name, []).append((item, events, cpu_s, q, peak if item["duration_ms"] >= 30_000 else None))
    for path in sorted(Path(args.results).glob("whisper-*.tsv")):
        model = path.stem[len("whisper-"):]
        cost = models["whisper"][model]["cost"]
        passes = {}
        with open(path, encoding="utf-8", errors="replace", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                passes.setdefault(row["item"], {}).setdefault(int(row["seg"]), []).append(
                    (int(row["point_ms"]), float(row["ms"]), int(row["n_tokens"]), row["text"]))
        for mode in ("raw", "la2"):
            name = f"whisper-{model}-{'npu' if cost == 'npu' else 'cpu1'}-{mode}"
            for item in items:
                key = f"{item['set']}/{item['id']}"
                if key not in passes:
                    continue
                events, busy_s, q, dropped = whisper_timeline(passes[key], item["segments"], mode, cost)
                if not events:
                    events = [(item["duration_ms"], "")]
                loops.setdefault(name, {}).setdefault(item["set"], 0)
                loops[name][item["set"]] += dropped
                engines.setdefault(name, []).append((item, events, busy_s if cost == "measured" else None, q, None))

    summary, finals = [], []
    for name, runs in sorted(engines.items()):
        for set_name in ("clips", "mixed", "continuous"):
            sel = [r for r in runs if r[0]["set"] == set_name]
            if not sel:
                continue
            agg = {"es": {"S": 0, "D": 0, "I": 0, "N": 0}, "ca": {"S": 0, "D": 0, "I": 0, "N": 0}}
            vs_large = {"es": [0, 0], "ca": [0, 0]}
            appear, settle, firsts, rev, fw, cpu, audio, queue = [], [], [], 0, 0, 0.0, 0.0, 0.0
            peaks = [r[4] for r in sel if r[4] is not None]
            for item, events, cpu_s, q, _ in sel:
                lang = item["utterances"][0]["language"] if len(item["utterances"]) == 1 else "es"
                s = score_item(item, events, lang)
                for u, p in zip(item["utterances"], s["per"]):
                    for k in "SDIN":
                        agg[u["language"]][k] += p[k]
                appear += s["appear"]
                settle += s["settle"]
                rev += s["revisions"]
                fw += s["final_words"]
                if s["first"] is not None:
                    firsts.append(s["first"])
                if cpu_s is not None:
                    cpu += cpu_s
                audio += item["duration_ms"] / 1000
                queue = max(queue, q)
                if set_name == "clips" and item["id"] in large:
                    hw, lw = words(s["final"], lang), words(large[item["id"]], lang)
                    _, ops = align(hw, lw)
                    vs_large[lang][0] += sum(1 for o in ops if o[0] != "C")
                    vs_large[lang][1] += len(lw)
                    finals.append({"engine": name, "id": item["id"], "language": lang, "draft": s["final"]})
            wer = lambda a: round(100 * (a["S"] + a["D"] + a["I"]) / a["N"], 1) if a["N"] else None
            row = {
                "engine": name, "set": set_name,
                "wer_es": wer(agg["es"]), "wer_ca": wer(agg["ca"]),
                "vs_large_es": round(100 * vs_large["es"][0] / vs_large["es"][1], 1) if vs_large["es"][1] else None,
                "vs_large_ca": round(100 * vs_large["ca"][0] / vs_large["ca"][1], 1) if vs_large["ca"][1] else None,
                "appear_p50_s": round(pct(appear, 50) / 1000, 2) if appear else None,
                "appear_p90_s": round(pct(appear, 90) / 1000, 2) if appear else None,
                "settle_p50_s": round(pct(settle, 50) / 1000, 2) if settle else None,
                "settle_p90_s": round(pct(settle, 90) / 1000, 2) if settle else None,
                "first_words_p50_s": round(pct(firsts, 50) / 1000, 2) if firsts else None,
                "revisions_per_100": round(100 * rev / fw, 1) if fw else None,
                "cpu_s_per_audio_s": round(cpu / audio, 3) if any(r[2] is not None for r in sel) else None,
                "max_queue_s": round(queue / 1000, 1),
                "peak_30s_cpu_s_per_s": round(max(peaks), 3) if peaks else None,
                "loop_passes_dropped": loops.get(name, {}).get(set_name),
                "words_matched": len(appear),
            }
            summary.append(row)

    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    with open(out / "drafts-clips.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["engine", "id", "language", "draft"], delimiter="\t")
        w.writeheader()
        w.writerows(finals)
    lines = ["# Draft-text bake-off (kngfsh-translate D127)", "",
             "GitHub runner (x86), one core per streaming engine; see eval/stream/README.md for how each figure is made.",
             "Lags are in seconds after a word was said (estimated); WER in %; CPU in seconds per second of audio.", ""]
    for set_name in ("clips", "mixed", "continuous"):
        rows = [r for r in summary if r["set"] == set_name]
        if not rows:
            continue
        lines += [f"## {set_name}", "",
                  "| engine | WER es | WER ca | vs large-v3 es | vs large-v3 ca | appear p50 / p90 | settle p50 / p90 | first words p50 | revisions /100 | CPU s/s (peak 30 s) | max queue s | loops dropped |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        f = lambda v: "" if v is None else str(v)
        for r in rows:
            lines.append(f"| {r['engine']} | {f(r['wer_es'])} | {f(r['wer_ca'])} | {f(r['vs_large_es'])} | {f(r['vs_large_ca'])} | "
                         f"{f(r['appear_p50_s'])} / {f(r['appear_p90_s'])} | {f(r['settle_p50_s'])} / {f(r['settle_p90_s'])} | "
                         f"{f(r['first_words_p50_s'])} | {f(r['revisions_per_100'])} | {f(r['cpu_s_per_audio_s'])}"
                         f"{' (' + str(r['peak_30s_cpu_s_per_s']) + ')' if r['peak_30s_cpu_s_per_s'] is not None else ''} | "
                         f"{f(r['max_queue_s'])} | {f(r['loop_passes_dropped'])} |")
        lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
