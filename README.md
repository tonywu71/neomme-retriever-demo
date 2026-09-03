---
title: NeoMME-Retriever - Demo
emoji: 🔎
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.22.0
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Visual document retrieval and RAG with NeoMME-Retriever
startup_duration_timeout: 1h
---

# NeoMME-Retriever

[![Hugging Face](https://img.shields.io/badge/Model_doc-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/docs/transformers/en/model_doc/neomme)
[![Hugging Face](https://img.shields.io/badge/Collection-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000)](https://hf.co/collections/Hcompany/neomme)
[![arXiv](https://img.shields.io/badge/arXiv-2609.01657-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2609.01657)

Upload PDFs or images, index them with NeoMME, then ask a question and read an answer taken from the pages that
ranked highest.

## How it works

The app converts every PDF page to an image because NeoMME reads pixels, not extracted text. NeoMME turns each page
into token vectors and compares the query with those vectors using MaxSim. The app runs that scoring itself, so it
does not need a separate index server.

Choose NeoMME 260M or 800M before indexing. Changing the model clears the index because vectors from different
models cannot be compared.

The app also stores one pooled vector for each page. You can switch to dense scoring in the retrieval settings, but
MaxSim is the default and the only mode covered by the published retrieval results.

When you ask a question, the app retrieves the highest-ranked pages first. It then sends the question and those page
images to a vision language model, which writes an answer using the retrieved pages as context. The Space includes a
small local model, and you can also use OpenAI, Anthropic, or Gemini. The app uses a visitor's key for one request
and does not store it.

The app uses `NeoMMEForRetrieval` and `NeoMMEProcessor` from `transformers`. It does not depend on the research
`neomme` package.

<!-- Image placeholder: visual RAG flow from user question, to top-k retrieved page images, to the vision language model, to the answer. -->

## Deploying

GitHub holds the source of truth, not HuggingFace Hub. Every push to `main` runs `.github/workflows/sync-to-hub.yml`,
which copies the repo to the Space, and you can also start it by hand from the Actions tab. Pushes that only touch
`.github` are skipped because the action never uploads that directory.

The workflow needs a GitHub Actions secret named `HF_TOKEN`, which is a write token for the `tonywu71` account.
It is not the same token as the Space secret of the same name, which is a read token for the `Hcompany` org.

To deploy without the workflow, for example while debugging:

```bash
hf upload tonywu71/neomme-retriever-demo . --repo-type space \
  --exclude "**/__pycache__/**" --exclude ".git/**" --exclude ".github/**"
```

The copy only goes from GitHub to the Hf Space, so anything you edit in the Space web UI is lost on the next deploy.

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

`HF_TOKEN` has to be a token that can read both private checkpoints. The app loads both sizes once so different
browser sessions can use different models safely. On Apple silicon it selects MPS and bfloat16. Only CPU falls
back to float32.

If the answer model fails to load, the app still starts and offers only the providers that need a key.

<details>
<summary>Environment variables</summary>

| Variable | Default | Effect |
| --- | --- | --- |
| `NEOMME_RELEASE_260M` | `Hcompany/NeoMME-260M-Retriever` | any compatible 260M repo the port can load |
| `NEOMME_RELEASE_800M` | `Hcompany/NeoMME-800M-Retriever` | any compatible 800M repo the port can load |
| `NEOMME_MODEL_SIZE` | `260m` | default model-size selection, either `260m` or `800m` |
| `NEOMME_VLM_LOCAL` | `LiquidAI/LFM2.5-VL-450M` | leave empty to disable local answers, which also removes the need for torchvision |
| `NEOMME_VLM_MAX_NEW_TOKENS` | 512 | how long an answer can be, and the main control on how long generation takes |
| `NEOMME_MAX_SIDE` | 2048 | the longest side a page image is resized to, trading quality for speed. 2048 is the cap used in the ViDoRe evaluation |
| `NEOMME_PAGE_BATCH` | 4 | how many pages go through one forward pass, which bounds memory at full page resolution |
| `NEOMME_GPU_DURATION` | 120 | how many seconds of GPU time indexing asks ZeroGPU for |
| `NEOMME_VLM_GPU_DURATION` | 60 | how many seconds of GPU time answering asks ZeroGPU for |
| `PORT` | 7860 | the local server port. Gradio's own `GRADIO_SERVER_PORT` has no effect, because `app.py` passes the port explicitly |

</details>

<details>
<summary>Space settings</summary>

These settings live on the Space rather than in git, so a rebuild from scratch does not restore them.

| Setting | Value | Why |
| --- | --- | --- |
| Hardware | `zero-a10g` | the app is written for ZeroGPU, with `import spaces` before torch, `@spaces.GPU` on encode, score and generate, and the models placed on cuda at import |
| Secret `HF_TOKEN` | a read token for the `Hcompany` org | both retrieval checkpoints are private |
| Variable `NEOMME_RELEASE_260M` | `Hcompany/NeoMME-260M-Retriever` | optional override for the published 260M checkpoint |
| Variable `NEOMME_RELEASE_800M` | `Hcompany/NeoMME-800M-Retriever` | optional override for the published 800M checkpoint |
| Variable `NEOMME_MODEL_SIZE` | `260m` | which radio option is selected when the app opens |
| Variable `GRADIO_SSR_MODE` | `false` | required for correct styling, because Gradio's server side rendering puts the app's CSS in the page before its own component stylesheets, which then override it. With rendering off, the CSS is applied last, as it is locally |

Hardware cannot be set from this repo. The `hardware:` key in the front matter above is ignored, so use the
Space settings page.

</details>

## Citation

```bibtex
@misc{lac2026neommesingletowermultimodalnativemultilingual,
      title={NeoMME: A Single-Tower Multimodal-Native Multilingual Foundation Encoder for Efficient Fine-Tuning and Inference},
      author={Aurélien Lac and Tony Wu},
      year={2026},
      eprint={2609.01657},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2609.01657},
}
```
