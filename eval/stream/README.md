# Draft-text bake-off

Which offline streaming recognizer could put the Castellano or Valencian words on the kngfsh-translate screen within a
second or so of being said, while the app's large-v3 still writes the final line? kngfsh-translate's DIVERGENCES D125
surveys the options; D126 records what this measures. The workflow is `.github/workflows/stream-draft.yml`.

| File | What it does |
|---|---|
| `sets.py` | The three streams (the 60 reference clips, the mixed track rebuilt from `reference/tracks/kngfsh-ref-mixed.csv`, the continuous track), each utterance's detected speech, and the app's 8 s segments (Silero VAD v6.2.0, `reference/build_tracks.py`'s segmenter) |
| `models.json` | The engines and their files, pinned by sha256 where a hash was known before the run |
| `fetch_models.py` | Downloads one engine and writes a lock file of every file's size and sha256 |
| `engines.py` | Moonshine (moonshine-voice), sherpa-onnx Zipformer and Vosk behind one feed/finish interface |
| `run_stream.py` | Streams every item through one engine in 100 ms chunks; per chunk the wall and CPU time, and every change of the shown text |
| `../asr/asr_host.cpp redecode` | Whisper decoding each 8 s segment again as it grows, once a second, the app's way |
| `score_stream.py` | Display times (a one-core queue, or the NPU cost model for turbo), then lag, revisions, word error and CPU; `eval/results/stream/report.md` |

Caveats: runner cores are x86, not the phone's; the Whisper turbo drafter's timing is Qualcomm's published NPU figure,
not a measurement; when each reference word was said is estimated by spreading an utterance's words evenly over its
detected speech; the clips are read speech (FLEURS), not a meeting, and some engines may have seen FLEURS in training.
