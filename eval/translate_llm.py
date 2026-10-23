#!/usr/bin/env python3
"""Translates the reference pack's transcripts and the recognized text of the app's phone sessions with a small
translation LLM through llama.cpp, so it can be compared with ML Kit on exactly what the app translated.

Two inputs, two outputs (both read by the same scorers as the app's own files):

- the 60 transcripts of `reference/manifest.json` -> a CSV in the format of the app's Translation quality export
  (`eval/score_translations.py`, `eval/compare_translations.py`);
- every shown line of the session exports in `eval/sessions/` (the text the recognizer heard, in the language the app
  decoded it in) -> a copy of each session export with the `english` column replaced (`eval/score_session.py`).

The model runs in `llama-server` with greedy decoding; each prompt is the model's own documented format, written out
here rather than left to a chat template so that it is visible and fixed:

- SalamandraTA (BSC-LT, Apache-2.0): ChatML, "Translate the following text from {source} into {target}.\\n{source}:
  {text} \\n{target}:" (model card).
- TranslateGemma (Google, Gemma terms): the prompt its chat template renders for a text message (model card's
  `chat_template.jinja`), target English (en-US).

Time per sentence is the runner's CPU (4 cores), not a phone's; it is only a rough scale.

Run by .github/workflows/llm-mt.yml.
"""
import argparse
import csv
import glob
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

NAMES = {"es": "Spanish", "ca": "Catalan"}
COLUMNS = [
    "id", "language", "fleurs_id", "source", "reference", "hypothesis", "status", "error", "translate_ms",
    "acronyms_lost", "engine", "pack", "device", "soc", "android", "app", "at_epoch_ms",
]
GUARDED = "=+-@\t\r"


def guard(cell):
    return "'" + cell if cell and cell[0] in GUARDED else cell


def unguard(cell):
    return cell[1:] if len(cell) > 1 and cell[0] == "'" and cell[1] in GUARDED else cell


def prompt_for(style, language, text):
    source = NAMES[language]
    if style == "salamandra":
        body = f"Translate the following text from {source} into English.\n{source}: {text} \nEnglish:"
        return f"<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n", ["<|im_end|>"]
    if style == "gemma":
        body = (
            f"You are a professional {source} ({language}) to English (en-US) translator. Your goal is to accurately "
            f"convey the meaning and nuances of the original {source} text while adhering to English grammar, "
            f"vocabulary, and cultural sensitivities.\nProduce only the English translation, without any additional "
            f"explanations or commentary. Please translate the following {source} text into English:\n\n\n{text.strip()}"
        )
        return f"<start_of_turn>user\n{body}<end_of_turn>\n<start_of_turn>model\n", ["<end_of_turn>"]
    raise SystemExit(f"unknown prompt style {style}")


class Server:
    def __init__(self, binary, model, threads, port):
        self.url = f"http://127.0.0.1:{port}"
        self.process = subprocess.Popen(
            [binary, "-m", model, "-c", "4096", "-t", str(threads), "--port", str(port)],
            stdout=subprocess.DEVNULL, stderr=sys.stderr,
        )
        for _ in range(600):
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=2) as response:
                    if response.status == 200:
                        return
            except OSError:
                pass
            if self.process.poll() is not None:
                raise SystemExit("llama-server exited while loading the model")
            time.sleep(1)
        raise SystemExit("llama-server did not become ready")

    def complete(self, prompt, stop):
        payload = {"prompt": prompt, "n_predict": 300, "temperature": 0.0, "top_k": 1, "stop": stop,
                   "cache_prompt": False}
        request = urllib.request.Request(self.url + "/completion", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.load(response)["content"].strip()

    def close(self):
        self.process.terminate()
        self.process.wait(timeout=30)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--server", required=True, help="path to llama-server")
    parser.add_argument("--model", required=True, help="the GGUF file")
    parser.add_argument("--engine", required=True, help="label for the engine column, e.g. salamandrata-2b-q4")
    parser.add_argument("--style", required=True, choices=["salamandra", "gemma"])
    parser.add_argument("--manifest", default="reference/manifest.json")
    parser.add_argument("--sessions", default="eval/sessions")
    parser.add_argument("--out", required=True, help="output folder")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    out = Path(args.out)
    (out / "sessions").mkdir(parents=True, exist_ok=True)

    server = Server(args.server, args.model, args.threads, args.port)
    cache, timings = {}, []

    def translate(language, text):
        key = (language, text)
        if key not in cache:
            prompt, stop = prompt_for(args.style, language, text)
            begin = time.perf_counter()
            cache[key] = server.complete(prompt, stop)
            elapsed = (time.perf_counter() - begin) * 1000
            timings.append(elapsed)
            print(f"[{language}] {elapsed:6.0f} ms  {text[:60]!r} -> {cache[key][:60]!r}", flush=True)
        return cache[key]

    try:
        started = int(time.time() * 1000)
        device = f"GitHub Actions {platform.system()} {platform.machine()}, {os.cpu_count()} CPUs, llama.cpp"
        with open(args.manifest, encoding="utf-8") as handle:
            clips = json.load(handle)["clips"]
        rows = []
        for clip in clips:
            begin = time.perf_counter()
            hypothesis = translate(clip["language"], clip["transcript"])
            rows.append({
                "id": clip["id"], "language": clip["language"], "fleurs_id": clip["fleurs_id"],
                "source": guard(clip["transcript"]), "reference": guard(clip["english_reference"]),
                "hypothesis": guard(hypothesis), "status": "ok", "error": "",
                "translate_ms": f"{(time.perf_counter() - begin) * 1000:.1f}", "acronyms_lost": 0,
                "engine": args.engine, "pack": "kngfsh-models reference/manifest.json", "device": device,
                "soc": platform.processor() or platform.machine(), "android": "", "app": "eval/translate_llm.py",
                "at_epoch_ms": started,
            })
        with open(out / f"{args.engine}.csv", "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\r\n")
            writer.writeheader()
            writer.writerows(rows)

        for path in sorted(glob.glob(os.path.join(args.sessions, "kngfsh-*.csv"))):
            with open(path, encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                fields, lines = reader.fieldnames, list(reader)
            for line in lines:
                source = unguard(line["source"])
                if line["status"] == "DONE" and source.strip() and line["language"] in NAMES:
                    line["english"] = guard(translate(line["language"], source))
            with open(out / "sessions" / Path(path).name, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(lines)
    finally:
        server.close()

    timings.sort()
    if timings:
        print(f"{args.engine}: {len(timings)} distinct texts, median {timings[len(timings) // 2]:.0f} ms, "
              f"90th percentile {timings[int(len(timings) * 0.9)]:.0f} ms per text on this runner")
        (out / f"{args.engine}-timing.txt").write_text(
            f"{args.engine}: {len(timings)} distinct texts; per text on the runner ({device}): median "
            f"{timings[len(timings) // 2]:.0f} ms, 90th percentile {timings[int(len(timings) * 0.9)]:.0f} ms, "
            f"max {timings[-1]:.0f} ms\n")


if __name__ == "__main__":
    main()
