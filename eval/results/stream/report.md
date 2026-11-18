# Draft-text bake-off (kngfsh-translate D127)

GitHub runner (x86), one core per streaming engine; see eval/stream/README.md for how each figure is made.
Lags are in seconds after a word was said (estimated); WER in %; CPU in seconds per second of audio.

## clips

| engine | WER es | WER ca | vs large-v3 es | vs large-v3 ca | appear p50 / p90 | settle p50 / p90 | first words p50 | revisions /100 | CPU s/s (peak 30 s) | max queue s | loops dropped |
|---|---|---|---|---|---|---|---|---|---|---|---|
| moonshine-small-es | 7.6 | 84.0 | 6.6 | 84.9 | 0.18 / 1.03 | 0.89 / 3.52 | 0.62 | 145.4 | 0.272 | 0.8 |  |
| moonshine-small-es-seg | 9.3 | 85.6 | 8.3 | 87.1 | 0.2 / 1.05 | 0.94 / 3.34 | 0.64 | 138.0 | 0.372 | 0.9 |  |
| moonshine-tiny-es | 8.5 | 88.4 | 7.9 | 89.1 | 0.36 / 1.08 | 0.74 / 3.31 | 1.0 | 110.0 | 0.062 | 0.1 |  |
| moonshine-tiny-es-seg | 11.2 | 88.5 | 10.6 | 89.4 | 0.34 / 1.07 | 0.69 / 3.01 | 1.01 | 103.4 | 0.075 | 0.2 |  |
| vosk-small-ca | 80.7 | 15.1 | 81.0 | 19.6 | 0.32 / 0.95 | 0.69 / 6.22 | 0.72 | 77.4 | 0.055 | 0.1 |  |
| vosk-small-es | 13.4 | 85.6 | 12.2 | 86.9 | 0.11 / 0.73 | 0.54 / 5.54 | 0.54 | 75.6 | 0.051 | 0.1 |  |
| whisper-base-q5_1-cpu1-la2 | 13.3 | 35.6 | 11.8 | 36.6 | 2.78 / 5.01 | 3.32 / 6.26 | 2.9 | 24.2 | 0.794 | 3.0 | 6 |
| whisper-base-q5_1-cpu1-raw | 13.3 | 35.6 | 11.8 | 36.6 | 1.06 / 1.9 | 2.15 / 5.13 | 1.86 | 120.8 | 0.794 | 3.0 | 6 |
| whisper-turbo-q8_0-npu-la2 | 8.1 | 14.4 | 7.7 | 15.9 | 1.94 / 3.86 | 2.02 / 4.41 | 3.39 | 5.6 |  | 2.9 | 65 |
| whisper-turbo-q8_0-npu-raw | 8.1 | 14.4 | 7.7 | 15.9 | 0.54 / 1.51 | 0.78 / 2.31 | 1.35 | 35.2 |  | 2.9 | 65 |
| zipformer-kroko-es | 3.6 | 82.3 | 2.7 | 83.6 | 0.77 / 1.52 | 0.87 / 1.61 | 1.47 | 8.9 | 0.034 | 0.1 |  |
| zipformer-surus-es | 14.5 | 87.0 | 12.6 | 88.0 | 0.09 / 0.62 | 0.26 / 0.74 | 0.52 | 47.0 | 0.035 | 0.0 |  |

## mixed

| engine | WER es | WER ca | vs large-v3 es | vs large-v3 ca | appear p50 / p90 | settle p50 / p90 | first words p50 | revisions /100 | CPU s/s (peak 30 s) | max queue s | loops dropped |
|---|---|---|---|---|---|---|---|---|---|---|---|
| moonshine-small-es | 4.9 | 81.3 |  |  | 0.74 / 17.68 | 1.78 / 20.58 |  | 153.1 | 0.814 (1.354) | 31.5 |  |
| moonshine-small-es-seg | 7.3 | 86.5 |  |  | 0.31 / 1.02 | 0.89 / 3.62 |  | 120.5 | 0.305 (0.39) | 1.5 |  |
| moonshine-tiny-es | 7.3 | 85.3 |  |  | 0.6 / 1.66 | 1.15 / 4.7 |  | 125.6 | 0.567 (1.074) | 3.8 |  |
| moonshine-tiny-es-seg | 9.2 | 86.2 |  |  | 0.37 / 1.13 | 0.74 / 3.54 |  | 113.4 | 0.064 (0.093) | 0.3 |  |
| vosk-small-ca | 82.1 | 16.9 |  |  | 0.34 / 0.94 | 0.78 / 6.02 |  | 80.7 | 0.031 (0.04) | 0.1 |  |
| vosk-small-es | 10.3 | 83.7 |  |  | 0.07 / 0.63 | 0.56 / 5.82 |  | 81.1 | 0.04 (0.051) | 0.1 |  |
| whisper-base-q5_1-cpu1-la2 | 13.3 | 43.3 |  |  | 2.71 / 4.76 | 3.38 / 6.58 |  | 29.1 | 0.638 | 2.8 | 3 |
| whisper-base-q5_1-cpu1-raw | 13.3 | 43.3 |  |  | 1.05 / 1.88 | 2.24 / 5.45 |  | 121.0 | 0.638 | 2.8 | 3 |
| whisper-turbo-q8_0-npu-la2 | 11.4 | 16.6 |  |  | 1.96 / 4.1 | 2.09 / 5.18 |  | 8.6 |  | 2.6 | 36 |
| whisper-turbo-q8_0-npu-raw | 11.4 | 16.6 |  |  | 0.55 / 1.7 | 0.78 / 2.29 |  | 28.9 |  | 2.6 | 36 |
| zipformer-kroko-es | 1.9 | 81.0 |  |  | 0.77 / 1.45 | 0.88 / 1.6 |  | 8.3 | 0.034 (0.035) | 0.1 |  |
| zipformer-surus-es | 23.6 | 86.8 |  |  | 0.1 / 0.68 | 0.23 / 0.77 |  | 47.4 | 0.033 (0.035) | 0.0 |  |

## continuous

| engine | WER es | WER ca | vs large-v3 es | vs large-v3 ca | appear p50 / p90 | settle p50 / p90 | first words p50 | revisions /100 | CPU s/s (peak 30 s) | max queue s | loops dropped |
|---|---|---|---|---|---|---|---|---|---|---|---|
| moonshine-small-es | 3.5 | 83.1 |  |  | 0.59 / 21.77 | 1.15 / 24.1 |  | 163.7 | 0.909 (1.494) | 35.2 |  |
| moonshine-small-es-seg | 6.8 | 87.1 |  |  | 0.01 / 1.05 | 0.93 / 2.99 |  | 147.3 | 0.441 (0.502) | 0.9 |  |
| moonshine-tiny-es | 4.9 | 88.3 |  |  | 0.57 / 1.89 | 0.95 / 5.02 |  | 125.0 | 0.6 (1.093) | 3.5 |  |
| moonshine-tiny-es-seg | 8.7 | 85.3 |  |  | 0.27 / 1.13 | 0.66 / 2.13 |  | 82.9 | 0.088 (0.104) | 0.1 |  |
| vosk-small-ca | 83.7 | 18.7 |  |  | 0.29 / 0.98 | 0.76 / 11.72 |  | 79.7 | 0.04 (0.058) | 0.0 |  |
| vosk-small-es | 14.6 | 86.8 |  |  | 0.12 / 1.0 | 1.03 / 13.19 |  | 90.8 | 0.055 (0.094) | 0.1 |  |
| whisper-base-q5_1-cpu1-la2 | 14.9 | 44.8 |  |  | 2.83 / 5.35 | 3.42 / 6.87 |  | 28.9 | 0.939 | 3.4 | 7 |
| whisper-base-q5_1-cpu1-raw | 14.9 | 44.8 |  |  | 1.2 / 2.23 | 2.6 / 6.3 |  | 136.6 | 0.939 | 3.4 | 7 |
| whisper-turbo-q8_0-npu-la2 | 3.0 | 23.3 |  |  | 2.17 / 5.72 | 2.63 / 6.91 |  | 14.3 |  | 3.6 | 35 |
| whisper-turbo-q8_0-npu-raw | 3.0 | 23.3 |  |  | 0.62 / 2.21 | 1.21 / 5.64 |  | 75.5 |  | 3.6 | 35 |
| zipformer-kroko-es | 2.2 | 78.5 |  |  | 0.75 / 1.54 | 0.84 / 1.59 |  | 7.1 | 0.034 (0.035) | 0.0 |  |
| zipformer-surus-es | 20.1 | 90.2 |  |  | -0.03 / 0.61 | 0.15 / 0.74 |  | 55.8 | 0.034 (0.035) | 0.0 |  |
