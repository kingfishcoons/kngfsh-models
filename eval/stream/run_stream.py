#!/usr/bin/env python3
"""Streams every item of the bake-off sets through one streaming engine and records what it would show, and when.

    python3 eval/stream/run_stream.py --engine NAME --models eval/stream/models.json --model-dir DIR \
        --items sets/items.json --out results/NAME.jsonl

Audio goes in as 100 ms chunks, as an Android AudioRecord buffer would deliver it. For every chunk the wall time and CPU
time the engine spent on it are recorded; whenever the shown text changes, the chunk index and the text. Run the
script pinned to one core (`taskset -c 0`) so the times say what one core of the runner can do; the scorer turns them
into display times with a queue (a chunk is processed once it has arrived and the previous one is done).
"""
import argparse
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engines  # noqa: E402

CHUNK = 1600  # 100 ms at 16 kHz


def load(path):
    with wave.open(str(path)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2, path
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--engine", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--items", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sets", default="clips,mixed,continuous")
    args = parser.parse_args()
    spec = dict(json.loads(Path(args.models).read_text())["engines"][args.engine])
    spec["dir"] = str(Path(args.model_dir) / spec.get("subdir", ""))
    wanted = set(args.sets.split(","))
    items = [i for i in json.loads(Path(args.items).read_text()) if i["set"] in wanted]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    total_audio = total_cpu = 0.0
    with open(args.out, "w", encoding="utf-8") as out:
        for item in items:
            audio = load(item["wav"])
            t0 = time.perf_counter()
            engine = engines.make(spec)
            load_ms = (time.perf_counter() - t0) * 1000
            wall, cpu, changes, last = [], [], [], None
            cuts = sorted(s["end_ms"] for s in item["segments"]) if spec.get("cut") == "segments" else []
            for k in range(0, len(audio), CHUNK):
                w0, c0 = time.perf_counter(), time.process_time()
                text = engine.feed(audio[k:k + CHUNK])
                while cuts and (k + CHUNK) * 1000 // 16000 >= cuts[0]:
                    cuts.pop(0)
                    engine.cut()
                    text = engine._text() if hasattr(engine, "_text") else text
                wall.append(round((time.perf_counter() - w0) * 1000, 2))
                cpu.append(round((time.process_time() - c0) * 1000, 2))
                if text != last:
                    changes.append([len(wall) - 1, text])
                    last = text
            w0, c0 = time.perf_counter(), time.process_time()
            final = engine.finish()
            finish_wall, finish_cpu = (time.perf_counter() - w0) * 1000, (time.process_time() - c0) * 1000
            out.write(json.dumps({
                "set": item["set"], "id": item["id"], "engine": args.engine, "chunk_ms": CHUNK * 1000 // 16000,
                "load_ms": round(load_ms, 1), "wall_ms": wall, "cpu_ms": cpu, "changes": changes, "final": final,
                "finish_wall_ms": round(finish_wall, 2), "finish_cpu_ms": round(finish_cpu, 2),
            }, ensure_ascii=False) + "\n")
            out.flush()
            total_audio += len(audio) / 16000
            total_cpu += (sum(cpu) + finish_cpu) / 1000
            print(f"{item['set']}/{item['id']}: {len(changes)} changes, cpu {sum(cpu) / 1000:.1f} s for "
                  f"{len(audio) / 16000:.1f} s", file=sys.stderr)
    print(f"{args.engine}: {total_audio:.0f} s of audio, {total_cpu:.1f} s CPU ({total_cpu / max(total_audio, 1):.3f} s a second)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
