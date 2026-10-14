# kngfsh-models

Public companion repository for the kngfsh-translate Android app, which listens to a meeting in Castellano or Valencian and shows and speaks American English, fully offline.

This repository will hold, as the app grows:

- `registry.json`: the public description of the models the app can download (files, URLs, SHA-256, sizes, evaluation results).
- `reference/`: a small pack of reference audio clips with transcripts, used to compare speech recognizers.
- `eval/`: the harness that measures word error rate and speed for candidate models.
- `.github/workflows/`: model conversion, the monthly scout that proposes new models by pull request, and publishing of the registry.

It is empty apart from this file until those pieces are built. Models themselves are not stored in git; they are published as release assets or fetched from their original sources.
