"""NeoMME document-retrieval demo — Gradio SDK Space, ZeroGPU-ready.

Runs on `transformers` alone: `NeoMMEForRetrieval` + `NeoMMEProcessor` from a converted checkpoint. The research
`neomme` package is NOT a dependency, which is the point — this is what a user of the published model does.

Flow: upload PDFs/images -> rasterize (pypdfium2 for PDFs) -> one forward per page batch, keeping both the
per-page multivector grids and the per-page pooled dense vectors in memory. One forward produces both, so
indexing both costs nothing extra. A query is then scored either with exact late-interaction MaxSim
(`sentence_transformers.util.mean_maxsim`, per-token L2-normalized cosine — the objective the checkpoint was
evaluated with)
or with cosine on the pooled vectors, whichever the visitor picks; MaxSim is the default and the only scoring the
published numbers cover. No external index server. The top pages are then sent, as images, to a user-selected VLM
(OpenAI / Anthropic / Gemini) with the user's own API key to synthesize an answer.

ZeroGPU: the GPU-heavy work (encode + score) lives in @spaces.GPU functions, so a GPU is allocated only for the
duration of those calls and released afterwards. The model is placed on cuda at import (ZeroGPU runs a CUDA
emulation outside @spaces.GPU that permits this); off ZeroGPU it auto-selects mps/cpu and the decorator is a no-op.

Env: NEOMME_RELEASE_260M / NEOMME_RELEASE_800M (HF repo ids), NEOMME_MODEL_SIZE (default selection),
NEOMME_MAX_SIDE (px, default 2048 = the ViDoRe eval), NEOMME_PAGE_BATCH, NEOMME_GPU_DURATION.
"""

import base64
import os
import time
from dataclasses import dataclass, field

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # a few ops MPS lacks must fall back to CPU

import gradio as gr
import spaces
import torch
from PIL import Image
from sentence_transformers.util import mean_maxsim
from theme import NEOMME_CSS, build_theme
from torch.nn.utils.rnn import pad_sequence
from transformers import NeoMMEForRetrieval, NeoMMEProcessor
from vlm import LOCAL_PROVIDER, PROVIDERS, generate_answer

_MAX_SIDE = int(os.environ.get("NEOMME_MAX_SIDE", "2048"))  # longest-side px cap; 2048 matches the ViDoRe eval
_PAGE_BATCH = int(os.environ.get("NEOMME_PAGE_BATCH", "4"))  # pages per forward; native resolution is memory-hungry
_GPU_DURATION = int(os.environ.get("NEOMME_GPU_DURATION", "120"))  # per-call ZeroGPU budget (s) for encode
_ZEROGPU = os.environ.get("SPACES_ZERO_GPU") == "true"

# On ZeroGPU, load onto cuda at import (a real GPU exists only inside @spaces.GPU; a CUDA emulation covers this
# module-level placement). Elsewhere, mps/cpu. Both sizes stay loaded so one visitor cannot change the shared
# process out from under another visitor who selected a different size.
_DEVICE = "cuda" if _ZEROGPU else ("mps" if torch.backends.mps.is_available() else "cpu")
# bf16 everywhere but cpu, which has no native bf16 kernels and emulates them: measured ~900x slower on a
# 1024x1024 matmul. On mps bf16 is 6.7x FASTER than fp32 and halves the resident weights.
_DTYPE = torch.float32 if _DEVICE == "cpu" else torch.bfloat16


@dataclass(frozen=True)
class RetrieverSpec:
    """One published NeoMME size and the dense widths used during its Matryoshka training."""

    label: str
    repo: str
    dense_dims: tuple[int, ...]


@dataclass(frozen=True)
class Retriever:
    """A loaded processor and model, kept together so their checkpoints cannot be mixed."""

    spec: RetrieverSpec
    processor: NeoMMEProcessor
    model: NeoMMEForRetrieval


_RETRIEVER_SPECS = {
    "260m": RetrieverSpec(
        label="260M",
        repo=os.environ.get("NEOMME_RELEASE_260M", "Hcompany/NeoMME-260M-Retriever"),
        dense_dims=(128, 256, 512, 1024),
    ),
    "800m": RetrieverSpec(
        label="800M",
        repo=os.environ.get("NEOMME_RELEASE_800M", "Hcompany/NeoMME-800M-Retriever"),
        dense_dims=(128, 256, 512, 1024, 1792),
    ),
}
_DEFAULT_RETRIEVER = os.environ.get("NEOMME_MODEL_SIZE", "260m").lower()
if _DEFAULT_RETRIEVER not in _RETRIEVER_SPECS:
    raise ValueError(f"NEOMME_MODEL_SIZE must be one of {sorted(_RETRIEVER_SPECS)}, got {_DEFAULT_RETRIEVER!r}")


def _load_retriever(spec: RetrieverSpec) -> Retriever:
    processor = NeoMMEProcessor.from_pretrained(spec.repo)
    model, loading_info = NeoMMEForRetrieval.from_pretrained(
        spec.repo, dtype=_DTYPE, output_loading_info=True
    )
    problems = {
        key: values
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if (values := loading_info.get(key))
    }
    if problems:
        raise RuntimeError(
            f"{spec.repo} is not a compatible Transformers-format NeoMME checkpoint: {problems}"
        )
    model = model.to(_DEVICE).eval()
    return Retriever(spec=spec, processor=processor, model=model)


_RETRIEVERS = {key: _load_retriever(spec) for key, spec in _RETRIEVER_SPECS.items()}
_RETRIEVER_CHOICES = [(f"neomme-retriever-{spec.label}", key) for key, spec in _RETRIEVER_SPECS.items()]
_RETRIEVER_LINKS = """
<div class="neo-model-links">
  Model cards:
  <a href="https://huggingface.co/Hcompany/NeoMME-260M-Retriever"
     target="_blank" rel="noopener noreferrer">🤗 Hcompany/NeoMME-260M-Retriever</a>
  <a href="https://huggingface.co/Hcompany/NeoMME-800M-Retriever"
     target="_blank" rel="noopener noreferrer">🤗 Hcompany/NeoMME-800M-Retriever</a>
</div>
"""


# A document to try the demo with. Clicking it puts the file in the UPLOADER rather than straight into the
# corpus, so the uploader stays the only thing that decides what is indexed: multi-select and deselect keep
# working, and there is no such thing as a document you cannot remove.
_SAMPLE_PDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples", "colpali_paper.pdf")

# Queries that actually have an answer in the sample, so the demo shows a hit rather than a shrug. The short label
# keeps the three buttons on one line; clicking one puts the full query in the box.
_EXAMPLE_QUERIES = (
    ("ColPali arch.", "Describe the ColPali architecture."),
    ("speed vs RAG", "How fast is ColPali vs traditional text-based RAG?"),
    ("token pooling", "What is token pooling?"),
)

# How a query is scored against the index. MaxSim is the default: it is what the checkpoint was trained and
# evaluated with, so the dense option is there to compare against, not as an equally supported mode.
_MAXSIM = "MaxSim"
_DENSE = "Dense"
_SCORINGS = (_MAXSIM, _DENSE)


@dataclass
class Corpus:
    """One visitor's indexed pages, carried between their index and search calls.

    Held in a `gr.State`, so every browser session gets its own. Module-level mutable state would be
    shared by every concurrent visitor of a public Space, which means serving one person's uploaded
    documents — and their page images — to the next.
    """

    pages: list[tuple[str, Image.Image]] = field(default_factory=list)
    doc_embeds: list[torch.Tensor] = field(default_factory=list)  # per-page multivector grids, on the CPU
    dense_embeds: torch.Tensor = field(default_factory=lambda: torch.empty(0))  # (n_pages, dense_dim), on the CPU
    retriever: str | None = None


def _pdf_to_images(path: str, dpi: int = 150) -> list[Image.Image]:
    """Rasterize every page of a PDF. pypdfium2 renders at 72 dpi natively, hence the scale."""
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(path)
    try:
        return [document[i].render(scale=dpi / 72.0).to_pil().convert("RGB") for i in range(len(document))]
    finally:
        document.close()


def _pages_from_files(paths: list[str]) -> list[tuple[str, Image.Image]]:
    """Expand uploaded files into (label, page image) pairs (PDFs -> one entry per page)."""
    pages: list[tuple[str, Image.Image]] = []
    for path in paths:
        if path.lower().endswith(".pdf"):
            for i, img in enumerate(_pdf_to_images(path)):
                pages.append((f"{os.path.basename(path)}#p{i + 1}", img))
        else:
            pages.append((os.path.basename(path), Image.open(path).convert("RGB")))
    return pages


@spaces.GPU(duration=_GPU_DURATION)
def _encode_documents(images: list[Image.Image], retriever: str) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Page images -> one (n_tokens, embedding_dim) grid per page plus one pooled vector per page, on the CPU.

    Both come out of the same forward, so keeping both costs one extra vector per page.

    The grids are trimmed to each page's real tokens: the head zeroes padding rows, and a ragged list keeps a
    long page from padding every short one across batches; scoring re-pads them behind a mask.
    """
    bundle = _RETRIEVERS[retriever]
    grids: list[torch.Tensor] = []
    pooled: list[torch.Tensor] = []
    for start in range(0, len(images), _PAGE_BATCH):
        batch = bundle.processor(
            images=images[start : start + _PAGE_BATCH], max_side=_MAX_SIDE, padding="longest", return_tensors="pt"
        ).to(_DEVICE)
        with torch.no_grad():
            output = bundle.model(**batch)
        embeddings = output.embeddings.float().cpu()
        pooled.append(output.dense_embeddings.float().cpu())
        lengths = batch["attention_mask"].sum(dim=-1).tolist()
        grids.extend(embeddings[row, :length] for row, length in enumerate(lengths))
    return grids, torch.cat(pooled)


@spaces.GPU
def _score(query: str, corpus: Corpus, scoring: str) -> list[float]:
    """Encode the query and score every indexed page with the scoring the visitor picked.

    Both heads L2-normalize their output, so the dense dot product below is already a cosine.
    """
    if corpus.retriever is None:
        raise ValueError("The corpus has no model size. Index the documents again.")
    bundle = _RETRIEVERS[corpus.retriever]
    messages = [[{"role": "user", "content": query}]]
    inputs = bundle.processor.apply_chat_template(
        messages, task="query", tokenize=True, return_dict=True, return_tensors="pt"
    ).to(_DEVICE)
    with torch.no_grad():
        output = bundle.model(**inputs)
    if scoring == _DENSE:
        query_dense = output.dense_embeddings.float().cpu()
        return (query_dense @ corpus.dense_embeds.T)[0].tolist()
    query_grid = output.embeddings.float().cpu()
    doc_grids = pad_sequence(corpus.doc_embeds, batch_first=True)
    doc_mask = pad_sequence(
        [torch.ones(grid.shape[0], dtype=torch.bool) for grid in corpus.doc_embeds], batch_first=True
    )
    return mean_maxsim(query_grid, doc_grids, a_mask=inputs["attention_mask"].cpu(), b_mask=doc_mask)[0].tolist()


def _index(pages: list[tuple[str, Image.Image]], retriever: str) -> tuple[Corpus, str]:
    """Encode `pages` into a fresh corpus for this session, and the status line that reports it."""
    grids, pooled = _encode_documents([image for _, image in pages], retriever)
    corpus = Corpus(pages=pages, doc_embeds=grids, dense_embeds=pooled, retriever=retriever)
    label = _RETRIEVER_SPECS[retriever].label
    return corpus, f"Indexed {len(pages)} pages with NeoMME {label}. Ready to retrieve."


def add_sample(files) -> list[str]:
    """Append the shipped sample to the uploader's list, leaving anything already there in place.

    `gr.File` hands us `NamedString`, a `str` whose value is the path, and takes plain paths back.
    """
    paths = [str(file) for file in (files or [])]
    if _SAMPLE_PDF not in paths:
        paths.append(_SAMPLE_PDF)
    return paths


def invalidate_index(files, retriever: str):
    """Clear embeddings and results whenever the selected files or retriever change."""
    label = _RETRIEVER_SPECS[retriever].label
    if files:
        status = f"Files changed. Index them with NeoMME {label} before retrieving."
    else:
        status = "No documents indexed yet."
    return (
        Corpus(),
        [],
        status,
        gr.update(variant="primary"),
        gr.update(interactive=False, variant="secondary"),
        [],
        "Index a document to see ranked pages.",
        gr.update(interactive=False),
        "",
        "",
    )


def build_index(files, retriever: str, progress=gr.Progress()):
    """Make the corpus match the file list exactly and update the active workflow state.

    The alternative, leaving the previous corpus in place, is worse than it sounds: the pages panel and the
    file list would disagree about what a search is actually searching.
    """
    if not files:
        raise gr.Error("Upload a PDF or image before indexing.")
    try:
        progress(0.05, desc="Preparing pages")
        pages = _pages_from_files([str(file) for file in files])
        progress(0.2, desc=f"Encoding {len(pages)} pages")
        corpus, status = _index(pages, retriever)
        progress(1.0, desc="Index ready")
    except Exception as error:
        print(f"[index] {type(error).__name__}: {error}")
        return (
            Corpus(),
            [],
            "Indexing failed. Check the uploaded files and try again.",
            gr.update(variant="primary"),
            gr.update(interactive=False, variant="secondary"),
            [],
            "No ranked pages yet.",
            gr.update(interactive=False),
            "",
            "",
        )
    return (
        corpus,
        [],
        status,
        gr.update(variant="secondary"),
        gr.update(interactive=True, variant="primary"),
        [],
        "Enter a query, then retrieve pages.",
        gr.update(interactive=False),
        "",
        "",
    )


def clear_results(corpus: Corpus):
    """Keep the index but clear results that no longer match the retrieval settings."""
    status = (
        "Retrieval settings changed. Retrieve again."
        if corpus and corpus.doc_embeds
        else "Index a document to see ranked pages."
    )
    return [], [], status, gr.update(interactive=False), "", ""


def _retrieve(query: str, top_k: int, scoring: str, corpus: Corpus) -> list[tuple[str, Image.Image]]:
    scores = _score(query, corpus, scoring)
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[: int(top_k)]
    return [(f"{corpus.pages[i][0]}  ({scores[i]:.3f})", corpus.pages[i][1]) for i in order]


_BIBTEX = """@misc{lac2026neommesingletowermultimodalnativemultilingual,
      title={NeoMME: A Single-Tower Multimodal-Native Multilingual Foundation Encoder for Efficient Fine-Tuning and Inference},
      author={Aurélien Lac and Tony Wu},
      year={2026},
      eprint={2609.01657},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2609.01657},
}"""

_NO_KEY_NOTE = (
    "The ranked pages are ready. This provider needs a key. Switch to the Local VLM or enter the provider key."
)


def retrieve_pages(
    query: str,
    top_k: int,
    scoring: str,
    retriever: str,
    corpus: Corpus,
):
    if not corpus or not corpus.doc_embeds:
        raise gr.Error("Index the documents before retrieving.")
    if corpus.retriever != retriever:
        raise gr.Error("The model size changed. Index the documents again.")
    if not query.strip():
        raise gr.Error("Enter a query before retrieving.")
    started = time.perf_counter()
    ranked = _retrieve(query, top_k, scoring, corpus)
    gallery = [(image, label) for label, image in ranked]
    elapsed = time.perf_counter() - started
    label = _RETRIEVER_SPECS[retriever].label
    status = f"Retrieved {len(ranked)} pages with NeoMME {label}, {scoring}, in {elapsed:.2f}s."
    return ranked, gallery, status, gr.update(interactive=True), "", ""


def generate_answer_from_pages(
    query: str,
    provider: str,
    api_key: str,
    model: str,
    ranked: list[tuple[str, Image.Image]],
):
    if not ranked:
        raise gr.Error("Retrieve pages before generating an answer.")
    if PROVIDERS[provider].needs_key and not api_key.strip():  # skip cleanly rather than erroring
        return _NO_KEY_NOTE, _NO_KEY_NOTE
    try:
        answer = generate_answer(provider, api_key, model, query, ranked)
    except Exception as error:  # surface provider/auth errors in the answer box, keep the pages visible
        print(f"[answer] {type(error).__name__}: {error}")
        raise gr.Error("Answer generation failed. Check the provider settings and try again.") from error
    return answer, answer  # same text feeds the rendered-markdown and raw views


def _on_provider_change(provider: str):
    spec = PROVIDERS[provider]
    return (
        gr.update(placeholder=spec.key_hint, interactive=spec.needs_key, value="" if not spec.needs_key else None),
        gr.update(value=spec.default_model, placeholder=spec.default_model),
    )


def _glyph_img() -> str:
    """Inline the glyph as a data URI so it renders without relying on Gradio static-file serving."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "neomme_glyph.webp")
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as handle:
        data = base64.b64encode(handle.read()).decode("ascii")
    return f'<img class="neo-glyph" src="data:image/webp;base64,{data}" alt="NeoMME logo"/>'


_HERO = f"""
<div class="neo-hero-bar">
  {_glyph_img()}<h1>NeoMME-Retriever</h1>
</div>
"""

_BADGES = """
<div class="neo-badges">
  <a href="https://arxiv.org/abs/2609.01657" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/arXiv-2609.01657-b31b1b.svg?style=for-the-badge" alt="arXiv: 2609.01657">
  </a>
  <a href="https://hf.co/collections/Hcompany/neomme" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/NeoMME_Collection-FFD21E?style=for-the-badge&amp;logo=huggingface&amp;logoColor=000" alt="NeoMME Collection on Hugging Face">
  </a>
</div>
"""

_ABOUT = """
<p>Upload PDFs or images, retrieve the pages that answer your query (<span class="neo-emphasis">retrieval</span>),
then generate an answer from those pages using a VLM (<span class="neo-emphasis">visual RAG</span>).</p>
"""

_INTRO = f"""
<div class="neo-intro">
  {_BADGES}
  <div class="neo-about">{_ABOUT}</div>
</div>
"""


def _demo():
    with gr.Blocks(title="NeoMME-Retriever") as demo:  # theme/css passed at launch (Gradio 6)
        gr.HTML(_HERO)
        gr.HTML(_INTRO)

        with gr.Accordion("⚙️ Retrieval settings", open=False, elem_classes="neo-settings"):
            with gr.Row():
                with gr.Column(scale=2, min_width=260):
                    retriever = gr.Radio(
                        choices=_RETRIEVER_CHOICES,
                        value=_DEFAULT_RETRIEVER,
                        label="Retriever",
                        elem_classes="neo-retriever",
                    )
                    gr.HTML(_RETRIEVER_LINKS)
                top_k = gr.Slider(1, 10, value=3, step=1, label="Pages to retrieve", scale=1)
                scoring = gr.Radio(list(_SCORINGS), value=_MAXSIM, label="Scoring", scale=1)

        with gr.Row(equal_height=True, elem_classes="neo-workspace"):
            with gr.Column(scale=2, min_width=300, elem_classes="neo-index-panel"):
                gr.Markdown("## 1. Build the index", elem_classes="neo-step")
                files = gr.File(
                    file_count="multiple",
                    file_types=[".pdf", "image"],
                    label="PDFs / images",
                    height=220,
                    elem_classes="neo-upload",
                )
                gr.Markdown(
                    "No document ready? Try the sample paper.",
                    elem_classes="neo-hint",
                    visible=os.path.isfile(_SAMPLE_PDF),
                )
                sample_btn = gr.Button(
                    "Example: add the ColPali paper 📄 (10 pages)",
                    size="sm",
                    variant="secondary",
                    visible=os.path.isfile(_SAMPLE_PDF),
                )
                build_btn = gr.Button("Index documents", variant="primary", size="sm")
                status = gr.Markdown(
                    "No documents indexed yet.", elem_classes="neo-status", elem_id="neo-index-status"
                )
            with gr.Column(scale=3, min_width=340, elem_classes="neo-query-panel"):
                gr.Markdown("## 2. Search the corpus", elem_classes="neo-step")
                query = gr.Textbox(
                    label="Query", placeholder="What does the report say about …?", lines=5, elem_classes="neo-query"
                )
                gr.Markdown("Examples queries about the ColPali paper:", elem_classes="neo-hint")
                with gr.Row(elem_classes="neo-examples"):
                    for label, example in _EXAMPLE_QUERIES:
                        # default argument, or every button would close over the last loop value
                        gr.Button(label, size="sm", variant="secondary").click(lambda text=example: text, None, query)
                retrieve_btn = gr.Button("Retrieve pages", variant="secondary", interactive=False)
                retrieval_status = gr.Markdown(
                    "Index a document to see ranked pages.",
                    elem_classes=["neo-status", "neo-run-meta"],
                    elem_id="neo-retrieval-status",
                )

        gr.Markdown("## 3. Ranked pages", elem_classes="neo-step neo-results-heading")
        gallery = gr.Gallery(
            label="Ranked pages",
            columns=3,
            object_fit="contain",
            elem_classes="neo-gallery",
        )

        gr.Markdown(
            "## 4. Answer from the ranked pages",
            elem_classes=["neo-step", "neo-answer-heading"],
        )
        with gr.Column(elem_classes="neo-answer-section"):
            gr.Markdown(
                "Generate a grounded answer after checking the ranked pages. The default model runs on this Space.",
                elem_classes="neo-hint",
            )
            with gr.Row(elem_classes="neo-answer-controls"):
                provider = gr.Dropdown(
                    choices=list(PROVIDERS),
                    value=LOCAL_PROVIDER,
                    label="Provider",
                    scale=1,
                    min_width=170,
                )
                model = gr.Textbox(
                    label="Model",
                    value=PROVIDERS[LOCAL_PROVIDER].default_model,
                    placeholder=PROVIDERS[LOCAL_PROVIDER].default_model,
                    max_lines=1,
                    scale=2,
                    min_width=280,
                )
                api_key = gr.Textbox(
                    label="API key",
                    type="password",
                    placeholder=PROVIDERS[LOCAL_PROVIDER].key_hint,
                    interactive=PROVIDERS[LOCAL_PROVIDER].needs_key,
                    scale=1,
                    min_width=210,
                )
            answer_btn = gr.Button(
                "Generate answer from these pages", variant="primary", interactive=False
            )
            with gr.Tabs():
                with gr.Tab("Rendered"):
                    answer_md = gr.Markdown(elem_id="neo-answer")
                with gr.Tab("Raw"):
                    answer_raw = gr.Textbox(
                        label="Raw answer", lines=10, interactive=False, elem_classes="neo-answer"
                    )

        # Collapsed, because nobody needs the BibTeX on arrival and an open block costs the page a screenful.
        with gr.Accordion("Cite", open=False, elem_classes="neo-cite"):
            gr.Code(value=_BIBTEX, language=None, label="BibTeX (click to copy)", wrap_lines=False)

        # Per-session corpus: a Space serves many visitors from one process, so this must never be global.
        corpus = gr.State(Corpus())
        ranked = gr.State([])

        provider.change(_on_provider_change, [provider], [api_key, model])
        invalidation_outputs = [
            corpus,
            ranked,
            status,
            build_btn,
            retrieve_btn,
            gallery,
            retrieval_status,
            answer_btn,
            answer_md,
            answer_raw,
        ]
        files.change(invalidate_index, [files, retriever], invalidation_outputs)
        retriever.change(invalidate_index, [files, retriever], invalidation_outputs)
        build_btn.click(build_index, [files, retriever], invalidation_outputs)

        result_outputs = [ranked, gallery, retrieval_status, answer_btn, answer_md, answer_raw]
        retrieve_inputs = [query, top_k, scoring, retriever, corpus]
        retrieve_btn.click(retrieve_pages, retrieve_inputs, result_outputs)
        query.submit(retrieve_pages, retrieve_inputs, result_outputs)

        stale_result_outputs = [ranked, gallery, retrieval_status, answer_btn, answer_md, answer_raw]
        scoring.change(clear_results, [corpus], stale_result_outputs)
        top_k.release(clear_results, [corpus], stale_result_outputs)

        answer_btn.click(
            generate_answer_from_pages,
            [query, provider, api_key, model, ranked],
            [answer_md, answer_raw],
        )
        sample_btn.click(add_sample, files, files)
    return demo


if __name__ == "__main__":
    _demo().launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        theme=build_theme(),  # Gradio 6 takes theme/css on launch(), not on Blocks()
        css=NEOMME_CSS,
        ssr_mode=False,  # SSR loads our CSS before Gradio's component styles, which then override it
    )
