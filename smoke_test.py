"""Offline end-to-end smoke test for the demo: tiny random model -> encode 3 images -> one query -> a ranking.

Exercises the exact path the app uses (`NeoMMEProcessor` + `NeoMMEForRetrieval` + `score_retrieval`), so it needs
no checkpoint, no Hub access and no index server. Only `transformers` carrying the NeoMME port. Exits 0 on success.

    python smoke_test.py
"""

import os
import sys
import tempfile

import torch
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    NeoMMEConfig,
    NeoMMEForRetrieval,
    NeoMMEImageProcessor,
    NeoMMEProcessor,
    PreTrainedTokenizerFast,
)

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
    global_attn_every_n_layers=2,
    num_attention_heads=2,
    num_key_value_heads=2,
    head_dim=16,
    sliding_window_short=4,
    sliding_window_long=8,
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
        additional_special_tokens=["<doc>", "<img>", "<query>", "<row>"],
        model_max_length=_CONFIG["max_position_embeddings"],
    )
    return NeoMMEProcessor(image_processor=NeoMMEImageProcessor(patch_size=_PATCH_SIZE), tokenizer=tokenizer)


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
        pages = processor(images=images)
        with torch.no_grad():
            embeddings = model(**pages).multivector_embeddings
        lengths = pages["attention_mask"].sum(dim=-1).tolist()
        document_grids = [embeddings[row, :length] for row, length in enumerate(lengths)]

        query = processor(text=["hello world"], text_role="query")
        with torch.no_grad():
            query_embeddings = model(**query).multivector_embeddings

        scores = processor.score_retrieval(query_embeddings, document_grids)[0].tolist()
        assert len(scores) == len(images), scores
        assert all(-1.0 <= score <= 1.0 for score in scores), scores
        top = max(range(len(scores)), key=lambda index: scores[index])
        print(f"[smoke] OK — encoded {len(images)} pages, MaxSim {[round(s, 3) for s in scores]}, top page {top}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
