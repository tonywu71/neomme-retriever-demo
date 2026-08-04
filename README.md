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

Upload PDFs or images, index them with NeoMME, then ask a question and read an answer taken from the pages that
ranked highest.

## How it works

Every PDF page is converted into an image, because NeoMME reads a page as pixels instead of as extracted text.
The model encodes each page image into a grid of token vectors, a few pages per forward pass, and it scores your
query against those grids with MaxSim. Scoring runs inside the app, so there is no separate index server.

The same forward pass also produces one pooled vector per page, so the app indexes both representations and you
can switch scoring under the query box. MaxSim is the default, and it is the only scoring the published retrieval
numbers cover, so treat the dense option as a comparison rather than a supported mode.

A small vision language model on the Space writes the answer from the top pages, so a visitor needs no API key.
OpenAI, Anthropic and Gemini are also in the provider list, and they answer dense pages better. A visitor's key is
used for that one request and is never stored.

The app depends on `transformers` alone, through `NeoMMEForRetrieval` and `NeoMMEProcessor`. The research `neomme`
package is deliberately not a dependency, so the app also shows that the published model works the way anyone
outside the team would use it.

## Space settings

These settings live on the Space rather than in git, so a rebuild from scratch does not restore them.

| Setting | Value | Why |
| --- | --- | --- |
| Hardware | `zero-a10g` | the app is written for ZeroGPU, with `import spaces` before torch, `@spaces.GPU` on encode, score and generate, and the models placed on cuda at import |
| Secret `HF_TOKEN` | a read token for the `Hcompany` org | the checkpoint in `NEOMME_RELEASE` is private |
| Variable `NEOMME_RELEASE` | `Hcompany/neomme-250M-retrieval-dev-transformers-v0.3` | which checkpoint to serve |
| Variable `GRADIO_SSR_MODE` | `false` | required for correct styling, because Gradio's server side rendering puts the app's CSS in the page before its own component stylesheets, which then override it. With rendering off, the CSS is applied last, as it is locally |

Hardware cannot be set from this repo. The `hardware:` key in the front matter above is ignored, so use the
Space settings page.

## Deploying

GitHub holds the source of truth. Every push to `main` runs `.github/workflows/sync-to-hub.yml`, which copies the
repo to the Space, and you can also start it by hand from the Actions tab. Pushes that only touch `.github` are
skipped, because the action never uploads that directory.

The workflow needs a GitHub Actions secret named `HF_TOKEN`, which is a write token for the `tonywu71` account.
It is not the same token as the Space secret of the same name, which is a read token for the `Hcompany` org.

A code change restarts the Space in about 40 seconds. A change to `requirements.txt` or to the front matter of
this file triggers a full rebuild instead, which takes several minutes because `transformers` installs from git.

To deploy without the workflow, for example while debugging:

```bash
hf upload tonywu71/neomme-retriever-demo . --repo-type space \
  --exclude "**/__pycache__/**" --exclude ".git/**" --exclude ".github/**"
```

The copy only goes from GitHub to the Space, so anything you edit in the Space web UI is lost on the next deploy.

## Running locally

Use a separate virtual environment, because these requirements leave `torch` unpinned and would conflict with the
research repo's pinned versions.

```bash
uv venv --python 3.12 .venv-space
VIRTUAL_ENV=.venv-space uv pip install -r requirements-local.txt
.venv-space/bin/python app.py                     # http://127.0.0.1:7860
.venv-space/bin/python smoke_test.py              # offline check with a tiny random model
```

`requirements.txt` is the build file for the Space, and it leaves out `gradio` and `spaces` because the Spaces
runtime installs both and pinning them breaks ZeroGPU. `requirements-local.txt` adds them back for local runs.

`HF_TOKEN` has to be a token that can read the private checkpoint. On Apple silicon the app selects mps and
bfloat16 and uses 1.43GB of memory, after downloading about 1.5GB of weights on the first launch. Only cpu falls
back to float32.

If the answer model fails to load, the app still starts and offers only the providers that need a key.

### Environment variables

| Variable | Default | Effect |
| --- | --- | --- |
| `NEOMME_RELEASE` | `Hcompany/neomme-250M-retrieval-dev-transformers-v0.3` | any repo the port can load |
| `NEOMME_VLM_LOCAL` | `LiquidAI/LFM2.5-VL-450M` | leave empty to disable local answers, which also removes the need for torchvision |
| `NEOMME_VLM_MAX_NEW_TOKENS` | 512 | how long an answer can be, and the main control on how long generation takes |
| `NEOMME_MAX_SIDE` | 2048 | the longest side a page image is resized to, trading quality for speed. 2048 is the cap used in the ViDoRe evaluation |
| `NEOMME_PAGE_BATCH` | 4 | how many pages go through one forward pass, which bounds memory at full page resolution |
| `NEOMME_GPU_DURATION` | 120 | how many seconds of GPU time indexing asks ZeroGPU for |
| `NEOMME_VLM_GPU_DURATION` | 60 | how many seconds of GPU time answering asks ZeroGPU for |
| `PORT` | 7860 | the local server port. Gradio's own `GRADIO_SERVER_PORT` has no effect, because `app.py` passes the port explicitly |
