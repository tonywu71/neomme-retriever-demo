---
title: NeoMME Document Retrieval
emoji: 📄
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.22.0
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Late-interaction visual document retrieval (NeoMME 256M)
startup_duration_timeout: 1h
---

# NeoMME retrieval demo

Gradio Space on ZeroGPU that indexes PDFs and images with NeoMME, then answers questions about them by
late-interaction retrieval.

It runs on **`transformers` alone** (`NeoMMEForRetrieval` + `NeoMMEProcessor`); the research `neomme` package is
deliberately not a dependency, so it doubles as a check that the published model is usable the way anyone else
would use it.

## What it does

1. Rasterize the uploads (PDFs via `pypdfium2`), then one forward per `NEOMME_PAGE_BATCH` pages, keeping
   `multivector_embeddings` trimmed to each page's real tokens. A visitor with no PDF to hand can click one button
   to drop the shipped sample into the uploader, and the example-query buttons fill the box with questions that
   document answers. Indexing makes the corpus match the file list exactly, so clearing the uploader and
   re-indexing is how you empty it.
2. Score a query against those grids with exact MaxSim (`processor.score_retrieval`), in-process, no index server.
3. Answer from the top pages. The default model runs **on the Space**, so no key is needed; OpenAI / Anthropic /
   Gemini stay in the picker for stronger answers on dense pages, using the visitor's key for that request only.

## Where this code lives

GitHub is the source of truth; the Space is a mirror of it, pushed by `.github/workflows/sync-to-hub.yml`. That
workflow is **manual-dispatch only** for now, so a push to `main` does not touch the Space until the `push:`
trigger is uncommented. Until then, mirror by hand:

```bash
hf upload tonywu71/<space-name> . --repo-type space --exclude "**/__pycache__/**" --exclude ".git/**"
```

The sync is one-way. Anything edited in the Space's web UI or by hot-reload is overwritten on the next push, so
fixes have to go back into this repo.

## Space settings

Hardware and visibility are set on the Space, never from this repo (`hardware:` in the frontmatter above is
silently ignored):

| Setting | Value | Why |
| --- | --- | --- |
| Hardware | `zero-a10g` (ZeroGPU) | the app is written for it: `import spaces` before torch, `@spaces.GPU` on encode / score / generate, module-scope `.to("cuda")` |
| Secret `HF_TOKEN` | a read token | `NEOMME_RELEASE` is a private repo, so the Space cannot download the weights without it |
| Variable `NEOMME_RELEASE` | see the table below | which checkpoint to serve |

## Run locally

Use a **separate venv**: these requirements leave `torch` unpinned and would fight the research repo's pins.

```bash
uv venv --python 3.12 .venv-space
VIRTUAL_ENV=.venv-space uv pip install -r requirements-local.txt

.venv-space/bin/python app.py          # http://127.0.0.1:7860
```

`requirements.txt` is the Space build spec and omits `gradio` and `spaces`, because the Spaces runtime preinstalls
both and pinning them breaks the ZeroGPU hijack. `requirements-local.txt` adds them back for local runs.

On Apple silicon it picks mps and bfloat16: 1.43GB resident (0.53 retriever + 0.90 answer VLM), ~1.5GB of weights
downloaded on first launch. Only cpu falls back to float32, because it emulates bf16 rather than running it.

| Variable | Default | Effect |
| --- | --- | --- |
| `NEOMME_RELEASE` | `Hcompany/neomme-250M-retrieval-dev-transformers-v0.3` | any repo the port can load |
| `NEOMME_VLM_LOCAL` | `LiquidAI/LFM2.5-VL-450M` | empty disables local generation, and with it the torchvision requirement |
| `NEOMME_VLM_MAX_NEW_TOKENS` | 512 | answer length; the main lever on generation time |
| `NEOMME_MAX_SIDE` | 2048 (the ViDoRe eval cap) | page resolution, trading quality for speed |
| `NEOMME_PAGE_BATCH` | 4 | pages per forward, bounding native-resolution memory |
| `NEOMME_GPU_DURATION` | 120 | ZeroGPU budget (s) declared for indexing |
| `NEOMME_VLM_GPU_DURATION` | 60 | ZeroGPU budget (s) declared for generation |
| `PORT` | 7860 | server port, local runs only |

ZeroGPU checks a visitor's remaining quota against the **declared** duration, not the actual runtime, so the two
`*_GPU_DURATION` values should be trimmed to the realistic worst case once the Space logs show real timings. A
smaller declared duration also ranks higher in the GPU queue.

### Port already in use

```bash
PORT=7861 .venv-space/bin/python app.py    # or: lsof -nP -iTCP:7860 -sTCP:LISTEN  then  kill <PID>
```

Gradio's `GRADIO_SERVER_PORT`, which its error message suggests, has no effect: `app.py` passes `server_port`
explicitly and that wins.

### `operator torchvision::nms does not exist`

The torchvision wheel does not match the installed torch (2.8 -> 0.23, 2.9 -> 0.24, 2.10 -> 0.25, 2.11 -> 0.26).
Reinstall the matching one. `vlm.py` catches this at import and falls back to the key-only providers, so the
symptom is "no local answers" rather than a crash.

## Smoke test

```bash
.venv-space/bin/python smoke_test.py    # tiny random model, 3 images, one query, MaxSim ranking
```

No checkpoint and no Hub access: it builds a small `NeoMMEForRetrieval` and a word-level tokenizer, then runs the
app's exact encode-and-score path.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | Gradio UI + ZeroGPU: upload → index → query → gallery → answer |
| `vlm.py` | answer generation: the local VLM plus the bring-your-own-key providers |
| `theme.py` | Gradio theme + CSS porting the `how-it-works.html` editorial look (light/dark) |
| `smoke_test.py` | offline end-to-end check |
| `samples/` | the document the sample button adds to the uploader (`_SAMPLE_PDF` in `app.py`) |
| `assets/` | the glyph `app.py` inlines into the hero bar as a data URI |
