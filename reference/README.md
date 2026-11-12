# Reference clip pack

Clips for comparing speech recognizers on Castellano and Valencian, used by the kngfsh-translate app (its Diagnostics clips come from here) and by the eval harness later.

## Contents

| Path | What it is |
|---|---|
| `audio/es-<id>.wav`, `audio/ca-<id>.wav` | 30 utterances per language, 4 to 15 s, 16 kHz mono PCM16 WAV; `<id>` is the FLEURS id (see below) |
| `manifest.json` | Each clip's id, file, language, duration, raw and normalized transcript, `english_reference`, `english_source`, and licence; the build's source revision and English config |
| `transcripts-es.tsv`, `transcripts-ca.tsv` | File, duration, raw transcript and `english` (the English reference), one line per clip; `null` where a clip has no English |
| `tracks/kngfsh-ref-continuous.wav` | The continuous-speech phone-test track (4 min 30 s), built from 30 of these clips by `build_tracks.py` (below) |
| `tracks/kngfsh-ref-continuous.csv`, `tracks/kngfsh-ref-continuous-segments.csv` | Where each clip sits in the track; the segments the app's detector and segmenter cut it into |
| `script/es-castellano.txt`, `script/ca-valencia.txt` | A school-themed script of about 2 to 3 minutes, to be read aloud and recorded on the phones |

**Clip ids.** FLEURS gives the same id to every recording of a sentence, and a sentence can be read by more than one speaker. The first recording keeps the plain id (`es-1780`); later ones get a letter (`es-1780-b`, `-c`, ...). `build_pack.py` stops with an error if two clips would share an id or a file, if `audio/` and the manifest list different files, if a file does not match its manifest row (format and length), or if an id has two different English texts. It also stops if an English field is missing or empty (`null` is allowed).

**English references.** Each clip's `english_reference` is the FLEURS `en_us` transcription of the same id, read from the `en_us` train, dev and test TSVs at the pinned revision (the same sentence in English; text copied unchanged). `english_source` says where it came from, e.g. `google/fleurs en_us, id 1664, test`. An id with no English in any split is `null`; the build prints those ids and does not fail. Two clips that share a FLEURS id (a second recording, `-b`) carry the same English. Five ids in this pack are repeated (`es-1780`/`-b`, `es-1756`/`ca-1756`, `es-1867`/`ca-1867`, `es-1986`/`ca-1986`, `ca-1992`/`-b`); FLEURS gives the same id across languages, so the Castellano/Valencian and English texts for those ids are parallel.

The audio and transcripts are written by `build_pack.py`, run by `.github/workflows/reference.yml` (manual `workflow_dispatch`, on the branch that holds the workflow). Rebuilding at the same dataset revision picks the same clips.

## The continuous-speech track

The other phone-test tracks (kngfsh-translate DIVERGENCES D81) leave 2.5 s between clips, so each clip closes its own
segment and the app's hard cap is almost never reached. In a real meeting half the segments ended at the cap (D105).
`tracks/kngfsh-ref-continuous.wav` tests the cap: the mixed track's 30 clips (the first 15 `es` and the first 15 `ca` in
id order, without `es-1780` and `ca-1992`), each trimmed to its speech plus 0.1 s, joined 0.4 s apart:

    4 s silence | 15 es clips (2 min 17 s) | 4 s silence | 15 ca clips (2 min 3 s) | 3 s silence

Play it from a laptop to the phone as the other tracks are played, in Auto, and score the session export with
`eval/score_session.py <export.csv> --track continuous`, which scores each language's monologue as one text (most of
its sentences are cut in two, so clip by clip scores mean little) and counts the line flags (`HARD_CAP`,
`LOOP_RETRIED`, `MERGED`).

What the app should do with it, on the clean signal (`tracks/kngfsh-ref-continuous-segments.csv`; Silero VAD v6.2.0
with the app's thresholds and the rules of the app's `Segmenter`, not a phone):

| Segmenter | Segments | At the hard cap | Cut inside a sentence | Median length |
|---|---|---|---|---|
| normal, 8 s cap (the app since build 53) | 33 | 30 | 30 | 8.0 s |
| normal, 12 s cap (before build 53) | 23 | 14 | 16 | 12.0 s |
| live mode, 6 s cap | 48 | 29 | 43 | 6.0 s |

Through a speaker and a room the detector will see more pauses or fewer, so the cuts on the phone will move.

Rebuild with `python3 reference/build_tracks.py --vad silero_vad.onnx` (the onnx file from the silero-vad 6.2.0
wheel; the script checks its hash). The same pack and model give the same bytes.

## Attribution

- **Speech and transcripts:** FLEURS, *Few-shot Learning Evaluation of Universal Representations of Speech*, from the dataset `google/fleurs` on Hugging Face, revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`, test split. Licence **CC-BY-4.0**: https://creativecommons.org/licenses/by/4.0/ . Check the dataset card for the current citation.
- **English references:** the same dataset and revision, config **`en_us`** (train, dev and test TSVs; the English transcription of each id). Licence **CC-BY-4.0**, the same as the speech and transcripts.
- **The continuous track** is an adaptation of 30 of these FLEURS clips (same licence, CC-BY-4.0): each clip
  trimmed to its speech and the clips joined with short silences. The transcripts and English references in its CSV
  are copied unchanged.
- **Changes made:** a seeded selection of 30 utterances per language within the duration window; resampling to 16 kHz mono PCM16 where the original was not already in that format; the transcripts and the English references are copied unchanged. The English fields were added after the audio was built, and the audio is byte-identical to the earlier build.

## Languages, and what they are not

- **`es`** is FLEURS config `es_419`, **Latin American Spanish**. FLEURS has no Castellano (peninsular) set, and no public European Spanish benchmark was found. Treat these clips as a proxy for Castellano, not a measure of it; the recordings of Ryan's script are the Castellano check.
- **`ca`** is FLEURS config `ca_es`, **Catalan** as read in Catalonia. It is not a Valencian-specific set. Valencian is checked with the Valencian script recordings.

## The school script

`script/es-castellano.txt` and `script/ca-valencia.txt` are parent-meeting texts of about 380 words each, written for the app's acceptance tests. They contain no real names or real pupils. Recordings made from them are not in this repository.
