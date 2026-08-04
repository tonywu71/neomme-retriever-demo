"""NeoMME document-retrieval demo — Gradio SDK Space, ZeroGPU-ready.

Runs on `transformers` alone: `NeoMMEForRetrieval` + `NeoMMEProcessor` from a converted checkpoint. The research
`neomme` package is NOT a dependency, which is the point — this is what a user of the published model does.

Flow: upload PDFs/images -> rasterize (pypdfium2 for PDFs) -> one forward per page batch, keeping both the
per-page multivector grids and the per-page pooled dense vectors in memory. One forward produces both, so
indexing both costs nothing extra. A query is then scored either with exact late-interaction MaxSim
(`processor.score_retrieval`, per-token L2-normalized cosine — the objective the checkpoint was evaluated with)
or with cosine on the pooled vectors, whichever the visitor picks; MaxSim is the default and the only scoring the
published numbers cover. No external index server. The top pages are then sent, as images, to a user-selected VLM
(OpenAI / Anthropic / Gemini) with the user's own API key to synthesize an answer.

ZeroGPU: the GPU-heavy work (encode + score) lives in @spaces.GPU functions, so a GPU is allocated only for the
duration of those calls and released afterwards. The model is placed on cuda at import (ZeroGPU runs a CUDA
emulation outside @spaces.GPU that permits this); off ZeroGPU it auto-selects mps/cpu and the decorator is a no-op.

Env: NEOMME_RELEASE (HF repo id), NEOMME_MAX_SIDE (px, default 2048 = the ViDoRe eval), NEOMME_PAGE_BATCH,
NEOMME_GPU_DURATION.
"""

import base64
import os
from dataclasses import dataclass, field

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # a few ops MPS lacks must fall back to CPU

import gradio as gr
import spaces
import torch
from PIL import Image
from theme import NEOMME_CSS, build_theme
from transformers import NeoMMEForRetrieval, NeoMMEProcessor
from vlm import LOCAL_PROVIDER, PROVIDERS, generate_answer

_REPO = os.environ.get("NEOMME_RELEASE", "Hcompany/neomme-250M-retrieval-dev-transformers-v0.3")
_MAX_SIDE = int(os.environ.get("NEOMME_MAX_SIDE", "2048"))  # longest-side px cap; 2048 matches the ViDoRe eval
_PAGE_BATCH = int(os.environ.get("NEOMME_PAGE_BATCH", "4"))  # pages per forward; native resolution is memory-hungry
_GPU_DURATION = int(os.environ.get("NEOMME_GPU_DURATION", "120"))  # per-call ZeroGPU budget (s) for encode
_ZEROGPU = os.environ.get("SPACES_ZERO_GPU") == "true"

# On ZeroGPU, load onto cuda at import (a real GPU exists only inside @spaces.GPU; a CUDA emulation covers this
# module-level placement). Elsewhere, mps/cpu.
_DEVICE = "cuda" if _ZEROGPU else ("mps" if torch.backends.mps.is_available() else "cpu")
# bf16 everywhere but cpu, which has no native bf16 kernels and emulates them: measured ~900x slower on a
# 1024x1024 matmul. On mps bf16 is 6.7x FASTER than fp32 and halves the resident weights.
_DTYPE = torch.float32 if _DEVICE == "cpu" else torch.bfloat16

processor = NeoMMEProcessor.from_pretrained(_REPO)
model = NeoMMEForRetrieval.from_pretrained(_REPO, dtype=_DTYPE).to(_DEVICE).eval()


# A document to try the demo with. Clicking it puts the file in the UPLOADER rather than straight into the
# corpus, so the uploader stays the only thing that decides what is indexed: multi-select and deselect keep
# working, and there is no such thing as a document you cannot remove.
_SAMPLE_PDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples", "colpali_paper.pdf")

# Queries that actually have an answer in the sample, so the demo shows a hit rather than a shrug. The short label
# keeps the three buttons on one line; clicking one puts the full query in the box.
_EXAMPLE_QUERIES = (
    ("token pooling", "What is token pooling?"),
    ("ColPali arch.", "Describe the ColPali architecture."),
    ("speed vs RAG", "How fast is ColPali vs traditional text-based RAG?"),
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
def _encode_documents(images: list[Image.Image]) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Page images -> one (n_tokens, embedding_dim) grid per page plus one pooled vector per page, on the CPU.

    Both come out of the same forward, so keeping both costs one extra vector per page.

    The grids are trimmed to each page's real tokens: the head zeroes padding rows, but a ragged list is what
    `score_retrieval` wants and it keeps a long page from padding every short one in the batch.
    """
    grids: list[torch.Tensor] = []
    pooled: list[torch.Tensor] = []
    for start in range(0, len(images), _PAGE_BATCH):
        batch = processor(images=images[start : start + _PAGE_BATCH], max_side=_MAX_SIDE).to(_DEVICE)
        with torch.no_grad():
            output = model(**batch)
        embeddings = output.multivector_embeddings.float().cpu()
        pooled.append(output.dense_embeddings.float().cpu())
        lengths = batch["attention_mask"].sum(dim=-1).tolist()
        grids.extend(embeddings[row, :length] for row, length in enumerate(lengths))
    return grids, torch.cat(pooled)


@spaces.GPU
def _score(query: str, corpus: Corpus, scoring: str) -> list[float]:
    """Encode the query and score every indexed page with the scoring the visitor picked.

    Both heads L2-normalize their output, so the dense dot product below is already a cosine.
    """
    inputs = processor(text=[query], text_role="query").to(_DEVICE)
    with torch.no_grad():
        output = model(**inputs)
    if scoring == _DENSE:
        query_dense = output.dense_embeddings.float().cpu()
        return (query_dense @ corpus.dense_embeds.T)[0].tolist()
    query_multivector = output.multivector_embeddings.float().cpu()
    return processor.score_retrieval(query_multivector, corpus.doc_embeds)[0].tolist()


def _index(pages: list[tuple[str, Image.Image]]) -> tuple[Corpus, str]:
    """Encode `pages` into a fresh corpus for this session, and the status line that reports it."""
    grids, pooled = _encode_documents([image for _, image in pages])
    corpus = Corpus(pages=pages, doc_embeds=grids, dense_embeds=pooled)
    return corpus, f"✓ Indexed {len(pages)} pages, late interaction and dense. Ready to search."


def add_sample(files) -> list[str]:
    """Append the shipped sample to the uploader's list, leaving anything already there in place.

    `gr.File` hands us `NamedString`, a `str` whose value is the path, and takes plain paths back.
    """
    paths = [str(file) for file in (files or [])]
    if _SAMPLE_PDF not in paths:
        paths.append(_SAMPLE_PDF)
    return paths


def build_index(files) -> tuple[Corpus, str]:
    """Make the corpus match the file list exactly, so clearing the uploader and re-indexing empties it.

    The alternative, leaving the previous corpus in place, is worse than it sounds: the pages panel and the
    file list would disagree about what a search is actually searching.
    """
    if not files:
        return Corpus(), "*Corpus cleared. Upload PDFs or images, then index them.*"
    return _index(_pages_from_files([str(file) for file in files]))


def _retrieve(query: str, top_k: int, scoring: str, corpus: Corpus) -> list[tuple[str, Image.Image]]:
    scores = _score(query, corpus, scoring)
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[: int(top_k)]
    return [(f"{corpus.pages[i][0]}  ({scores[i]:.3f})", corpus.pages[i][1]) for i in order]


_BIBTEX = """@software{neomme2026,
  author = {Lac, Aurélien and Wu, Tony},
  title  = {{NeoMME}: a vision-tower-free masked-diffusion multimodal document retriever},
  year   = {2026},
  url    = {https://github.com/hcompai/neomme},
}"""

_NO_KEY_NOTE = (
    "*The ranked pages are on the left. This provider needs a key, so either switch back to the Local VLM "
    "or paste a key under **Generate an answer**.*"
)


def search(query: str, top_k: int, scoring: str, provider: str, api_key: str, model: str, corpus: Corpus):
    if not corpus or not corpus.doc_embeds:
        return [], "Build the index first.", "Build the index first."
    if not query.strip():
        return [], "Enter a query.", "Enter a query."
    ranked = _retrieve(query, top_k, scoring, corpus)
    gallery = [(image, label) for label, image in ranked]
    if PROVIDERS[provider].needs_key and not api_key.strip():  # skip cleanly rather than erroring
        return gallery, _NO_KEY_NOTE, _NO_KEY_NOTE
    try:
        answer = generate_answer(provider, api_key, model, query, ranked)
    except Exception as error:  # surface provider/auth errors in the answer box, keep the pages visible
        answer = f"⚠️ {type(error).__name__}: {error}"
    return gallery, answer, answer  # same text feeds the rendered-markdown and raw views


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
  {_glyph_img()}<h1>NeoMME</h1><span class="neo-subtitle">document retrieval</span>
</div>
"""

_ABOUT = """
NeoMME encodes each page as a grid of token vectors and scores a query against them with MaxSim, the
ColBERT-style late-interaction objective. Retrieval reads the *rendered* page, so there is no OCR step.
The retrieved pages are then fed to a vision-language model, which defaults to a small one running on
this Space. Index again whenever you change documents.
"""


def _demo():
    with gr.Blocks(title="NeoMME document retrieval") as demo:  # theme/css passed at launch (Gradio 6)
        gr.HTML(_HERO)
        gr.Markdown(f"### How it works\n\n{_ABOUT}", elem_classes="neo-about")

        # Inputs: Upload | Query | Generate-answer, each column owning the button that acts on it: Index under
        # the uploader, Submit under the provider controls. min_width => clean wrap on a narrow viewport.
        with gr.Row(equal_height=True, elem_classes="neo-controls"):
            with gr.Column(scale=2, min_width=220):
                gr.Markdown("## 1. Upload", elem_classes="neo-step")
                files = gr.File(
                    file_count="multiple",
                    file_types=[".pdf", "image"],
                    label="PDFs / images",
                    height=210,
                    elem_classes="neo-upload",
                )
                gr.Markdown(
                    "*No PDF? Add the sample, then index it.*",
                    elem_classes="neo-hint",
                    visible=os.path.isfile(_SAMPLE_PDF),
                )
                sample_btn = gr.Button(
                    "📄  Add the ColPali paper (10 pages)",
                    size="sm",
                    variant="secondary",
                    visible=os.path.isfile(_SAMPLE_PDF),
                )
                build_btn = gr.Button("Index documents", variant="primary", size="sm")
                status = gr.Markdown("*No documents indexed yet.*", elem_classes="neo-status")
            with gr.Column(scale=3, min_width=240):
                gr.Markdown("## 2. Ask", elem_classes="neo-step")
                query = gr.Textbox(
                    label="Query", placeholder="What does the report say about …?", lines=4, elem_classes="neo-query"
                )
                gr.Markdown("*Click an example to fill the query box.*", elem_classes="neo-hint")
                with gr.Row(elem_classes="neo-examples"):
                    for label, example in _EXAMPLE_QUERIES:
                        # default argument, or every button would close over the last loop value
                        gr.Button(label, size="sm", variant="secondary").click(lambda text=example: text, None, query)
                with gr.Row():  # side by side, so the control band stays as short as column 3
                    top_k = gr.Slider(1, 10, value=3, step=1, label="Pages to retrieve")
                    scoring = gr.Radio(list(_SCORINGS), value=_MAXSIM, label="Scoring")
            with gr.Column(scale=3, min_width=240):
                gr.Markdown("## 3. Generate an answer", elem_classes="neo-step")
                gr.Markdown(
                    "*The default model runs on this Space, so no key is needed.*",
                    elem_classes="neo-hint",
                )
                provider = gr.Dropdown(choices=list(PROVIDERS), value=LOCAL_PROVIDER, label="Provider")
                with gr.Row():  # side by side, so the column ends near the other two
                    model = gr.Textbox(
                        label="Model",
                        value=PROVIDERS[LOCAL_PROVIDER].default_model,
                        placeholder=PROVIDERS[LOCAL_PROVIDER].default_model,
                        max_lines=1,
                    )
                    api_key = gr.Textbox(
                        label="API key",
                        type="password",
                        placeholder=PROVIDERS[LOCAL_PROVIDER].key_hint,
                        interactive=PROVIDERS[LOCAL_PROVIDER].needs_key,
                    )
                search_btn = gr.Button("Submit", variant="primary", size="sm")

        with gr.Row(equal_height=False):
            with gr.Column(scale=3, min_width=320):
                gr.Markdown("## 4. Retrieved pages", elem_classes="neo-step")
                # Height comes from the CSS, which scales it with the window instead of pinning it.
                gallery = gr.Gallery(label="Ranked pages", columns=3, object_fit="contain", elem_classes="neo-gallery")
            with gr.Column(scale=2, min_width=280):
                gr.Markdown("## 5. Answer", elem_classes="neo-step")
                with gr.Tabs():  # markdown-rendered by default; raw for copy/inspection
                    with gr.Tab("Rendered"):
                        answer_md = gr.Markdown(elem_id="neo-answer")
                    with gr.Tab("Raw"):
                        answer_raw = gr.Textbox(
                            label="Raw answer", lines=10, interactive=False, elem_classes="neo-answer"
                        )

        # Collapsed, because nobody needs the BibTeX on arrival and an open block costs the page a screenful.
        with gr.Accordion("Cite", open=False):
            gr.Code(value=_BIBTEX, language=None, label="BibTeX (click to copy)", wrap_lines=False)

        # Per-session corpus: a Space serves many visitors from one process, so this must never be global.
        corpus = gr.State(Corpus())

        provider.change(_on_provider_change, [provider], [api_key, model])
        build_btn.click(build_index, [files], [corpus, status])
        search_btn.click(
            search, [query, top_k, scoring, provider, api_key, model, corpus], [gallery, answer_md, answer_raw]
        )
        sample_btn.click(add_sample, files, files)
    return demo


if __name__ == "__main__":
    _demo().launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        theme=build_theme(),  # Gradio 6 takes theme/css on launch(), not on Blocks()
        css=NEOMME_CSS,
    )
