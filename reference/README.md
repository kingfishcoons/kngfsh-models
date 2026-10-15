# Reference clip pack

Clips for comparing speech recognizers on Castellano and Valencian, used by the kngfsh-translate app (its Diagnostics clips come from here) and by the eval harness later.

## Contents

| Path | What it is |
|---|---|
| `audio/es-<id>.wav`, `audio/ca-<id>.wav` | 30 utterances per language, 4 to 15 s, 16 kHz mono PCM16 WAV |
| `manifest.json` | Each clip's id, file, language, duration, raw and normalized transcript, and licence; the build's source revision |
| `transcripts-es.tsv`, `transcripts-ca.tsv` | File, duration and raw transcript, one line per clip |
| `script/es-castellano.txt`, `script/ca-valencia.txt` | A school-themed script of about 2 to 3 minutes, to be read aloud and recorded on the phones |

The audio and transcripts are written by `build_pack.py`, run by `.github/workflows/reference.yml` (manual `workflow_dispatch`, on the branch that holds the workflow). Rebuilding at the same dataset revision picks the same clips.

## Attribution

- **Speech and transcripts:** FLEURS, *Few-shot Learning Evaluation of Universal Representations of Speech*, from the dataset `google/fleurs` on Hugging Face, revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`, test split. Licence **CC-BY-4.0**: https://creativecommons.org/licenses/by/4.0/ . Check the dataset card for the current citation.
- **Changes made:** a seeded selection of 30 utterances per language within the duration window; resampling to 16 kHz mono PCM16 where the original was not already in that format; the transcripts are copied unchanged.

## Languages, and what they are not

- **`es`** is FLEURS config `es_419`, **Latin American Spanish**. FLEURS has no Castellano (peninsular) set, and no public European Spanish benchmark was found. Treat these clips as a proxy for Castellano, not a measure of it; the recordings of Ryan's script are the Castellano check.
- **`ca`** is FLEURS config `ca_es`, **Catalan** as read in Catalonia. It is not a Valencian-specific set. Valencian is checked with the Valencian script recordings.

## The school script

`script/es-castellano.txt` and `script/ca-valencia.txt` are parent-meeting texts of about 380 words each, written for the app's acceptance tests. They contain no real names or real pupils. Recordings made from them are not in this repository.
