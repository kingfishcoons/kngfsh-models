# eval/

Tools that measure translators on the reference pack (`reference/`). They produce and read the CSV that the
kngfsh-translate app exports from Diagnostics > Translation quality, so the app's translator (ML Kit, on a phone) and
a candidate (Opus-MT, on a runner) are scored the same way, on the same sentences.

| File | What it does |
|---|---|
| `translate_opus_mt.py` | Translates the pack's transcripts with Helsinki-NLP `opus-mt-es-en` and `opus-mt-ca-en` (full precision, beam 4, one sentence at a time) and writes the app's CSV format. |
| `score_translations.py` | BLEU, chrF and chrF++ per direction, a bootstrap interval for chrF, and the lowest-scoring sentences. |
| `compare_translations.py` | Two CSVs of the same sentences: scores side by side, the chrF difference with a paired bootstrap, sentence-by-sentence wins, and the biggest gains and losses. It stops if the two files do not hold the same sources and references. |
| `results/` | `mlkit-oneplus13-*.csv` is the export from the OnePlus 13; the `opus-mt-fp32*` and `compare-*` files are written by `.github/workflows/opus-mt.yml`. |

The workflow runs on a GitHub runner because the app's cloud sessions cannot download models. Only the PyTorch
full-precision model is measured: it is the ceiling for Opus-MT, and an int8 ONNX export (what the app would ship)
can only match or lose to it, so it is worth measuring only if the ceiling is worth acting on.

Scores use sacreBLEU's 0 to 100 scale (the plan's "chrF about 0.60" is 60 here) with one reference per sentence. The
sentences are FLEURS text (CC-BY-4.0), not meeting speech, and the Catalan clips are Catalonia Catalan, not Valencian;
see `reference/README.md`. The app repository's `docs/DIVERGENCES.md` (D89, D91, D92) has the reading of the numbers.

Opus-MT models: https://huggingface.co/Helsinki-NLP (CC-BY-4.0).

## Translation LLMs against ML Kit (`llm-mt.yml`, 2026-10-23)

| File | What it does |
|---|---|
| `translate_llm.py` | Runs a GGUF translation model in llama.cpp's server (greedy, the model's own prompt format) over the pack's 60 transcripts and over every shown line of the phone sessions in `sessions/`; writes the app's translations CSV and a copy of each session export with the model's English. |
| `sessions/` | Eight table exports of the app's phone sessions on the reference tracks (OnePlus 13, 2026-10-16 and 2026-10-23), cut to the columns the scorers read: the recognized text, the language the app decoded it in, and ML Kit's English as shown. The audio was the pack's FLEURS clips played from a laptop. |
| `score_session.py` | A copy of the app repository's `tools/score_session.py`: matches session lines to pack clips and scores the English shown, per clip and for the whole session. |
| `side_by_side.py` | ML Kit's English and each model's English next to each other, for reading. |
| `results/llm/` | Written by the workflow: each model's CSV, scores, the comparison with ML Kit, the session scores and `side-by-side.txt`. |

Models: BSC-LT `salamandraTA-2B-instruct-GGUF` (Q4, Apache-2.0) and Google's TranslateGemma 4B (`mradermacher/translategemma-4b-it-GGUF`, Q4_K_M, Gemma terms), both at the size a phone could hold. The runner's time per sentence is a 4-core server CPU, not a phone. The app repository's `docs/DIVERGENCES.md` D101 has the reading of the numbers.
