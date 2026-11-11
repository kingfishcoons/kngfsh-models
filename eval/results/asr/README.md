# Recognizer comparison (asr-compare.yml)

Written by `.github/workflows/asr-compare.yml` (kngfsh-translate DIVERGENCES D104): `report.md` (word error rate per set
and model, speed on the runner, punctuation, confidence), `side-by-side.md` (each clip's reference and every model's
text, for reading), `clips-by-model.csv`, the decode outputs (`<model>.tsv`, `.json`, `.log`), the clip list
(`clips.tsv`) and the gate's answers (`gate.tsv`).

## Attribution and licences

- **fleurs-ca, fleurs-es:** FLEURS (`google/fleurs`, CC-BY-4.0), the clips of `reference/` in this repository.
- **corts:** Corts Valencianes Speech Corpus, `projecte-aina/corts_valencianes_asr_a` revision `bbd2705`, clean test
  split, short segments; **CC-BY-NC-4.0**. Projecte Aina (Generalitat de Catalunya) / Barcelona Supercomputing Center,
  from the plenary sessions of the Corts Valencianes. Its card asks users not to try to identify the speakers.
- **codeswitch:** BSC's Catalan-Spanish code-switching ASR test set, `BSC-LT/BSCs_Code_Switching_CA-ES_ASR_Test`
  revision `a8618b9`; **CC-BY-NC-4.0**. Language Technologies Laboratory, Barcelona Supercomputing Center.
- **Changes made:** a seeded selection of clips of 3 to 15 s (120 per set, half from each source for codeswitch),
  resampled to 16 kHz mono where needed; references copied unchanged. The audio of the CC-BY-NC sets is not committed.
- **Models:** OpenAI Whisper large-v3 and large-v3-turbo (MIT; ggml files from `ggerganov/whisper.cpp`), and the
  Apache-2.0 fine-tunes `BSC-LT/whisper-3cat-cv21-valencian`, `projecte-aina/whisper-large-v3-tiny-caesar` and
  `projecte-aina/whisper-large-v3-ca-3catparla`, converted to ggml and quantized to q8_0 by the workflow.

The CC-BY-NC material is used here for non-commercial evaluation only.
