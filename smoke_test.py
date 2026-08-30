"""Offline end-to-end smoke test for the demo: tiny random model -> encode 3 images -> one query -> a ranking.

Exercises the exact path the app uses (`NeoMMEProcessor` + `NeoMMEForRetrieval` + `mean_maxsim`), so it needs
no checkpoint, no Hub access and no index server. Only `transformers` carrying the NeoMME port plus
`sentence-transformers` for the scoring util. Exits 0 on success.

    python smoke_test.py
"""

import os
import sys
import tempfile

import torch
from PIL import Image
from sentence_transformers.util import mean_maxsim
from tokenizers import Tokenizer, models, pre_tokenizers
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    NeoMMEConfig,
    NeoMMEForRetrieval,
    NeoMMEImageProcessor,
    NeoMMEProcessor,
    PreTrainedTokenizerFast,
)

# The retrieval chat template the published checkpoints ship; copied from the transformers NeoMME processor tests.
_CHAT_TEMPLATE = """
{%- if task is not defined -%}
    {{- raise_exception("NeoMME chat templates require task='query' or task='document'.") -}}
{%- endif -%}
{%- if task not in ['query', 'document'] -%}
    {{- raise_exception("task=" ~ task ~ " is not supported: expected 'query' or 'document'.") -}}
{%- endif -%}
{%- if messages is not defined or not messages -%}
    {{- raise_exception("NeoMME chat conversations must contain at least one message.") -}}
{%- endif -%}

{%- set state = namespace(text='', has_text=false, image_count=0) -%}
{%- for message in messages -%}
    {%- set content = message.content -%}
    {%- set items = [{'type': 'text', 'text': content}] if content is string else content -%}
    {%- for item in items -%}
        {%- if item.type == 'text' -%}
            {%- if image_token in item.text -%}
                {{- raise_exception(image_token ~ " is reserved for image documents.") -}}
            {%- endif -%}
            {%- set state.has_text = true -%}
            {%- set state.text = state.text + item.text -%}
        {%- elif item.type == 'image' -%}
            {%- if item.image is not defined or item.image is none or item.image == '' -%}
                {{- raise_exception("NeoMME image content must provide an image source.") -}}
            {%- endif -%}
            {%- set state.image_count = state.image_count + 1 -%}
        {%- elif item.type == 'image_url' -%}
            {%- if item.image_url is not defined or not item.image_url -%}
                {{- raise_exception("NeoMME image_url content must provide an image source.") -}}
            {%- endif -%}
            {%- set state.image_count = state.image_count + 1 -%}
        {%- else -%}
            {{- raise_exception("NeoMME chat templates do not support content type " ~ item.type ~ ".") -}}
        {%- endif -%}
    {%- endfor -%}
{%- endfor -%}

{%- if state.image_count and state.has_text -%}
    {{- raise_exception("NeoMME cannot encode text and images in the same conversation.") -}}
{%- endif -%}
{%- if state.image_count > 1 -%}
    {{- raise_exception("NeoMME accepts one image document per conversation.") -}}
{%- endif -%}
{%- if state.image_count and task != 'document' -%}
    {{- raise_exception("NeoMME image content must use task='document'.") -}}
{%- endif -%}

{%- set content = image_token if state.image_count else state.text -%}
{%- if task == 'query' -%}
    {{- query_token + content + mask_token * 10 -}}
{%- else -%}
    {{- document_token + content -}}
{%- endif -%}
"""

# The frozen special block, which the marker convention indexes by name.
_SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>", "<mask>", "<doc>", "<img>", "<query>", "<row>"]
_PATCH_SIZE = 16
_CONFIG = dict(
    vocab_size=64,
    embedding_rank=16,
    embedding_dim=24,
    hidden_size=32,
    intermediate_size=48,
    num_hidden_layers=2,
    layer_types=["sliding_attention", "full_attention"],
    sliding_window=4,
    num_attention_heads=2,
    num_key_value_heads=2,
    head_dim=16,
    patch_size=_PATCH_SIZE,
    max_position_embeddings=512,
)


def _processor(directory: str) -> NeoMMEProcessor:
    """A word-level tokenizer whose specials sit at the frozen ids, wrapped with the image processor."""
    vocabulary = {token: index for index, token in enumerate(_SPECIALS)}
    for word in ("hello", "world"):
        vocabulary[word] = len(vocabulary)

    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    path = os.path.join(directory, "tokenizer.json")
    backend.save(path)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=path,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
        mask_token="<mask>",
        # Registered as named attributes (`tokenizer.image_token`, ...) the processor reads; the tokens
        # already sit in the vocabulary, so nothing is added.
        extra_special_tokens={
            "document_token": "<doc>",
            "image_token": "<img>",
            "query_token": "<query>",
            "row_token": "<row>",
        },
        model_max_length=_CONFIG["max_position_embeddings"],
    )
    return NeoMMEProcessor(
        image_processor=NeoMMEImageProcessor(patch_size=_PATCH_SIZE),
        tokenizer=tokenizer,
        chat_template=_CHAT_TEMPLATE,
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        torch.manual_seed(0)
        processor = _processor(directory)
        model = NeoMMEForRetrieval(NeoMMEConfig(**_CONFIG)).eval()

        images = [
            Image.new("RGB", (64, 64), (200, 30, 30)),
            Image.new("RGB", (48, 96), (30, 200, 30)),
            Image.new("RGB", (80, 48), (30, 30, 200)),
        ]
        # Same shape as the app: one grid per page, trimmed to its real tokens.
        pages = processor(images=images, padding="longest", return_tensors="pt")
        with torch.no_grad():
            embeddings = model(**pages).embeddings
        lengths = pages["attention_mask"].sum(dim=-1).tolist()
        document_grids = [embeddings[row, :length] for row, length in enumerate(lengths)]

        messages = [[{"role": "user", "content": "hello world"}]]
        query = processor.apply_chat_template(
            messages, task="query", tokenize=True, return_dict=True, return_tensors="pt"
        )
        with torch.no_grad():
            query_embeddings = model(**query).embeddings

        # Same scoring as the app: re-pad the ragged grids behind a mask, then MeanMaxSim.
        doc_grids = pad_sequence(document_grids, batch_first=True)
        doc_mask = pad_sequence(
            [torch.ones(grid.shape[0], dtype=torch.bool) for grid in document_grids], batch_first=True
        )
        scores = mean_maxsim(query_embeddings, doc_grids, a_mask=query["attention_mask"], b_mask=doc_mask)[0].tolist()
        assert len(scores) == len(images), scores
        assert all(-1.0 <= score <= 1.0 for score in scores), scores
        top = max(range(len(scores)), key=lambda index: scores[index])
        print(f"[smoke] OK — encoded {len(images)} pages, MeanMaxSim {[round(s, 3) for s in scores]}, top page {top}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
