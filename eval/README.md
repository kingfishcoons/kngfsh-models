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
