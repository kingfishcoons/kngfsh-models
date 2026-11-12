# Recognizer comparison: Valencian, Catalan, Spanish and code-switching (kngfsh-translate D104)

Word error rate after normalizing both sides (lower case, digits spelled out, punctuation removed). 'bad' = clips with a word error rate of 30% or more. Speed is on the GitHub runner's CPU (4 cores, same threads for every model), as a multiple of the audio's length: a ratio between models, not the phone's time. 'punct' = share of clips whose text has any punctuation.

- **turbo-q8_0**: OpenAI large-v3-turbo, q8_0: the app's recognizer (ggerganov/whisper.cpp@5359861)
- **3catparla-q8_0**: projecte-aina/whisper-large-v3-ca-3catparla@6e07d3a (large-v3 fine-tuned on 710 h of Catalan TV, Apache-2.0), converted and quantized here to q8_0
- **caesar-q8_0**: projecte-aina/whisper-large-v3-tiny-caesar@cf087e1 (large-v3 fine-tuned for Catalan-Spanish code-switching, Apache-2.0), converted and quantized here to q8_0
- **large-v3-q8_0**: OpenAI large-v3, quantized here to q8_0: the base of the three fine-tunes (ggerganov/whisper.cpp@5359861)
- **valencian-q8_0**: BSC-LT/whisper-3cat-cv21-valencian@2cc599f (large-v3 fine-tuned on Corts Valencianes and Common Voice 21, Apache-2.0), converted and quantized here to q8_0

## codeswitch (120 clips, 19.5 min)

| model | WER | bad clips | median clip WER | speed (x real time) | punct | token_p median | flagged < 0.85 (of them bad) |
|---|---|---|---|---|---|---|---|
| turbo-q8_0 | 83.1% | 82 (68%) | 44.0% | 0.84 | 98% | 0.866 | 47 (43) |
| 3catparla-q8_0 | 50.2% | 54 (45%) | 27.1% | 1.17 | 0% | 0.880 | 46 (37) |
| caesar-q8_0 | 46.5% | 60 (50%) | 29.3% | 1.56 | 70% | 0.872 | 50 (39) |
| large-v3-q8_0 | 49.4% | 75 (62%) | 38.2% | 1.31 | 94% | 0.888 | 40 (37) |
| valencian-q8_0 | 62.4% | 58 (48%) | 29.3% | 1.33 | 2% | 0.877 | 43 (35) |

3catparla-q8_0 against turbo-q8_0: 57 clips better by 10 points or more, 6 worse, 57 within 10 points.

caesar-q8_0 against turbo-q8_0: 51 clips better by 10 points or more, 12 worse, 57 within 10 points.

large-v3-q8_0 against turbo-q8_0: 33 clips better by 10 points or more, 12 worse, 75 within 10 points.

valencian-q8_0 against turbo-q8_0: 52 clips better by 10 points or more, 18 worse, 50 within 10 points.

## corts (120 clips, 15.0 min)

| model | WER | bad clips | median clip WER | speed (x real time) | punct | token_p median | flagged < 0.85 (of them bad) |
|---|---|---|---|---|---|---|---|
| turbo-q8_0 | 112.5% | 70 (58%) | 37.2% | 1.01 | 99% | 0.835 | 67 (51) |
| 3catparla-q8_0 | 100.9% | 38 (32%) | 19.0% | 1.39 | 2% | 0.903 | 40 (31) |
| caesar-q8_0 | 73.8% | 43 (36%) | 20.9% | 1.81 | 78% | 0.865 | 54 (37) |
| large-v3-q8_0 | 60.3% | 54 (45%) | 24.5% | 1.46 | 89% | 0.860 | 56 (45) |
| valencian-q8_0 | 93.0% | 37 (31%) | 14.1% | 1.56 | 0% | 0.934 | 35 (30) |

3catparla-q8_0 against turbo-q8_0: 60 clips better by 10 points or more, 23 worse, 37 within 10 points.

caesar-q8_0 against turbo-q8_0: 55 clips better by 10 points or more, 19 worse, 46 within 10 points.

large-v3-q8_0 against turbo-q8_0: 49 clips better by 10 points or more, 23 worse, 48 within 10 points.

valencian-q8_0 against turbo-q8_0: 68 clips better by 10 points or more, 19 worse, 33 within 10 points.

## fleurs-ca (30 clips, 5.1 min)

| model | WER | bad clips | median clip WER | speed (x real time) | punct | token_p median | flagged < 0.85 (of them bad) |
|---|---|---|---|---|---|---|---|
| turbo-q8_0 | 45.5% | 8 (27%) | 11.8% | 0.79 | 100% | 0.930 | 7 (5) |
| 3catparla-q8_0 | 7.8% | 1 (3%) | 5.3% | 1.04 | 0% | 0.960 | 1 (1) |
| caesar-q8_0 | 9.9% | 3 (10%) | 8.7% | 1.42 | 33% | 0.940 | 4 (2) |
| large-v3-q8_0 | 9.9% | 3 (10%) | 4.8% | 1.20 | 97% | 0.954 | 4 (3) |
| valencian-q8_0 | 31.8% | 5 (17%) | 6.6% | 1.22 | 0% | 0.937 | 6 (4) |

3catparla-q8_0 against turbo-q8_0: 10 clips better by 10 points or more, 2 worse, 18 within 10 points.

caesar-q8_0 against turbo-q8_0: 11 clips better by 10 points or more, 3 worse, 16 within 10 points.

large-v3-q8_0 against turbo-q8_0: 10 clips better by 10 points or more, 1 worse, 19 within 10 points.

valencian-q8_0 against turbo-q8_0: 10 clips better by 10 points or more, 2 worse, 18 within 10 points.

## fleurs-es (30 clips, 5.7 min)

| model | WER | bad clips | median clip WER | speed (x real time) | punct | token_p median | flagged < 0.85 (of them bad) |
|---|---|---|---|---|---|---|---|
| turbo-q8_0 | 15.3% | 5 (17%) | 4.5% | 0.74 | 97% | 0.954 | 5 (2) |
| 3catparla-q8_0 | 4.0% | 0 (0%) | 1.6% | 0.99 | 23% | 0.963 | 1 (0) |
| caesar-q8_0 | 4.5% | 0 (0%) | 1.8% | 1.34 | 17% | 0.980 | 0 (0) |
| large-v3-q8_0 | 2.9% | 0 (0%) | 0.0% | 1.13 | 100% | 0.981 | 1 (0) |
| valencian-q8_0 | 4.0% | 0 (0%) | 3.3% | 1.12 | 0% | 0.971 | 1 (0) |

3catparla-q8_0 against turbo-q8_0: 5 clips better by 10 points or more, 2 worse, 23 within 10 points.

caesar-q8_0 against turbo-q8_0: 5 clips better by 10 points or more, 2 worse, 23 within 10 points.

large-v3-q8_0 against turbo-q8_0: 5 clips better by 10 points or more, 2 worse, 23 within 10 points.

valencian-q8_0 against turbo-q8_0: 7 clips better by 10 points or more, 3 worse, 20 within 10 points.

