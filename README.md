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

Upload PDFs or images, index them with NeoMME, ask a question, get an answer from the top-ranked pages.

Pages are rasterized (`pypdfium2`), encoded one batch at a time into a grid of token vectors per page, and scored
against the query with exact MaxSim (`processor.score_retrieval`). No index server. The answer is written by a
small VLM running on the Space, so no key is needed; OpenAI / Anthropic / Gemini stay in the picker for stronger
answers on dense pages, using the visitor's key for that request only.

It runs on `transformers` alone (`NeoMMEForRetrieval` + `NeoMMEProcessor`). The research `neomme` package is
deliberately not a dependency, so this doubles as a check that the published model is usable the way anyone else
would use it.

## Space settings

These live on the Space, not in git, and a rebuild from scratch will not restore them.

| Setting | Value | Why |
| --- | --- | --- |
| Hardware | `zero-a10g` | the app is written for ZeroGPU: `import spaces` before torch, `@spaces.GPU` on encode / score / generate, module-scope `.to("cuda")` |
| Secret `HF_TOKEN` | read token for the `Hcompany` org | `NEOMME_RELEASE` is a private repo |
| Variable `NEOMME_RELEASE` | `Hcompany/neomme-250M-retrieval-dev-transformers-v0.3` | which checkpoint to serve |
| Variable `GRADIO_SSR_MODE` | `false` | **required for correct styling**, see below |

`hardware:` in the frontmatter above is silently ignored; hardware is set at creation or in Space settings.

### Why SSR is off

With SSR on, Gradio writes the `css=` string into `<head>` at render time, ahead of its own component
stylesheets. Equal specificity, later wins, so Gradio's rules override this app's: the hero glyph loses
`height: 2.2rem` and renders at the webp's natural size, and the layout loses its `max-width`. With SSR off the
CSS ships inside `gradio_config` and is injected after the bundle, which is also what happens locally.

Turning SSR back on means making the CSS immune to load order first, for example by prefixing selectors with
`.gradio-container`.

## Deploying

GitHub is the source of truth. `.github/workflows/sync-to-hub.yml` mirrors to the Space but is
**manual-dispatch only** until the `push:` trigger is uncommented. Until then:

```bash
hf upload tonywu71/neomme-retriever-demo . --repo-type space --exclude "**/__pycache__/**" --exclude ".git/**" --exclude ".github/**"
```

The sync is one-way, so anything edited in the Space's web UI or by hot-reload is lost on the next push.

## Running locally

Use a separate venv: these requirements leave `torch` unpinned and would fight the research repo's pins.

```bash
uv venv --python 3.12 .venv-space
VIRTUAL_ENV=.venv-space uv pip install -r requirements-local.txt
.venv-space/bin/python app.py                     # http://127.0.0.1:7860
.venv-space/bin/python smoke_test.py              # offline check: tiny random model, 3 images, one query
```

`requirements.txt` is the Space build spec and omits `gradio` and `spaces`, which the Spaces runtime preinstalls
and which break the ZeroGPU hijack if pinned. `requirements-local.txt` adds them back.

`HF_TOKEN` must be able to read the private checkpoint. On Apple silicon the app picks mps and bfloat16, 1.43GB
resident, ~1.5GB downloaded on first launch. Only cpu falls back to float32.

| Variable | Default | Effect |
| --- | --- | --- |
| `NEOMME_RELEASE` | `Hcompany/neomme-250M-retrieval-dev-transformers-v0.3` | any repo the port can load |
| `NEOMME_VLM_LOCAL` | `LiquidAI/LFM2.5-VL-450M` | empty disables local generation, and the torchvision requirement with it |
| `NEOMME_VLM_MAX_NEW_TOKENS` | 512 | answer length, the main lever on generation time |
| `NEOMME_MAX_SIDE` | 2048 | page resolution, trading quality for speed (the ViDoRe eval cap) |
| `NEOMME_PAGE_BATCH` | 4 | pages per forward, bounding native-resolution memory |
| `NEOMME_GPU_DURATION` | 120 | ZeroGPU seconds declared for indexing |
| `NEOMME_VLM_GPU_DURATION` | 60 | ZeroGPU seconds declared for generation |
| `PORT` | 7860 | local server port; Gradio's `GRADIO_SERVER_PORT` has no effect here |

ZeroGPU checks a visitor's remaining quota against the **declared** duration, not the actual runtime, so trim the
two `*_GPU_DURATION` values once the logs show real timings. Smaller also ranks higher in the GPU queue.

### `operator torchvision::nms does not exist`

The torchvision wheel does not match the installed torch (2.8 to 0.23, 2.9 to 0.24, 2.10 to 0.25, 2.11 to 0.26).
`vlm.py` catches this at import and falls back to the key-only providers, so the symptom is "no local answers"
rather than a crash.

## Files

`app.py` UI, indexing and search. `vlm.py` answer generation. `theme.py` theme and CSS. `smoke_test.py` offline
check. `samples/` the sample-button PDF. `assets/` the glyph inlined into the hero as a data URI.
