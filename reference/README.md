# Reference clip pack

Clips for comparing speech recognizers on Castellano and Valencian, used by the kngfsh-translate app (its Diagnostics clips come from here) and by the eval harness later.

## Contents

| Path | What it is |
|---|---|
| `audio/es-<id>.wav`, `audio/ca-<id>.wav` | 30 utterances per language, 4 to 15 s, 16 kHz mono PCM16 WAV; `<id>` is the FLEURS id (see below) |
| `manifest.json` | Each clip's id, file, language, duration, raw and normalized transcript, `english_reference`, `english_source`, and licence; the build's source revision and English config |
| `transcripts-es.tsv`, `transcripts-ca.tsv` | File, duration, raw transcript and `english` (the English reference), one line per clip; `null` where a clip has no English |
| `script/es-castellano.txt`, `script/ca-valencia.txt` | A school-themed script of about 2 to 3 minutes, to be read aloud and recorded on the phones |

**Clip ids.** FLEURS gives the same id to every recording of a sentence, and a sentence can be read by more than one speaker. The first recording keeps the plain id (`es-1780`); later ones get a letter (`es-1780-b`, `-c`, ...). `build_pack.py` stops with an error if two clips would share an id or a file, if `audio/` and the manifest list different files, if a file does not match its manifest row (format and length), or if an id has two different English texts. It also stops if an English field is missing or empty (`null` is allowed).

**English references.** Each clip's `english_reference` is the FLEURS `en_us` transcription of the same id, read from the `en_us` train, dev and test TSVs at the pinned revision (the same sentence in English; text copied unchanged). `english_source` says where it came from, e.g. `google/fleurs en_us, id 1664, test`. An id with no English in any split is `null`; the build prints those ids and does not fail. Two clips that share a FLEURS id (a second recording, `-b`) carry the same English. Five ids in this pack are repeated (`es-1780`/`-b`, `es-1756`/`ca-1756`, `es-1867`/`ca-1867`, `es-1986`/`ca-1986`, `ca-1992`/`-b`); FLEURS gives the same id across languages, so the Castellano/Valencian and English texts for those ids are parallel.

The audio and transcripts are written by `build_pack.py`, run by `.github/workflows/reference.yml` (manual `workflow_dispatch`, on the branch that holds the workflow). Rebuilding at the same dataset revision picks the same clips.

## Attribution

- **Speech and transcripts:** FLEURS, *Few-shot Learning Evaluation of Universal Representations of Speech*, from the dataset `google/fleurs` on Hugging Face, revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`, test split. Licence **CC-BY-4.0**: https://creativecommons.org/licenses/by/4.0/ . Check the dataset card for the current citation.
- **English references:** the same dataset and revision, config **`en_us`** (train, dev and test TSVs; the English transcription of each id). Licence **CC-BY-4.0**, the same as the speech and transcripts.
- **Changes made:** a seeded selection of 30 utterances per language within the duration window; resampling to 16 kHz mono PCM16 where the original was not already in that format; the transcripts and the English references are copied unchanged. The English fields were added after the audio was built, and the audio is byte-identical to the earlier build.

## Languages, and what they are not

- **`es`** is FLEURS config `es_419`, **Latin American Spanish**. FLEURS has no Castellano (peninsular) set, and no public European Spanish benchmark was found. Treat these clips as a proxy for Castellano, not a measure of it; the recordings of Ryan's script are the Castellano check.
- **`ca`** is FLEURS config `ca_es`, **Catalan** as read in Catalonia. It is not a Valencian-specific set. Valencian is checked with the Valencian script recordings.

## The school script

`script/es-castellano.txt` and `script/ca-valencia.txt` are parent-meeting texts of about 380 words each, written for the app's acceptance tests. They contain no real names or real pupils. Recordings made from them are not in this repository.
