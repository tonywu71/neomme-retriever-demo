"""Answer generation over the retrieved document pages.

NeoMME does the retrieval; a vision-language model reads the top-ranked page images and writes the answer.
The default runs LOCALLY so a visitor with no API key still gets an answer; the hosted providers (OpenAI /
Anthropic / Gemini) stay available with the user's own key for materially better answers on dense pages.
Provider SDKs are imported lazily, so only the one actually used has to be installed.
"""

import base64
import io
import os

import spaces
from PIL import Image
from pydantic import BaseModel

_ANSWER_INSTRUCTIONS = (
    "You are given a user question and the document pages a retrieval system judged most relevant. "
    "Answer the question using ONLY the content visible in these pages. Cite the page label(s) you "
    "relied on. Reply in the same language as the question. If the pages do not contain the answer, "
    "say so plainly instead of guessing."
)


class ProviderSpec(BaseModel):
    """One selectable answer-generation backend."""

    label: str
    """Human-facing name shown in the provider dropdown."""
    default_model: str
    """Vision-capable model id used unless the user overrides it."""
    key_hint: str
    """Placeholder text describing the expected API key for this provider."""
    needs_key: bool = True
    """False for the local model, which runs on the Space's own GPU and needs no credentials."""


# Runs on the Space GPU, so the demo answers without a key. Fits ZeroGPU `large` (48GB) with room to spare:
# 448.7M params x 2B (bf16) = 0.90GB weights, next to the retriever's 0.53GB. Its KV cache is small because
# only 6 of 16 layers are full attention (the rest are short-conv): 2 x 6 x 8 kv-heads x 64 dims x 2B =
# 12KB/token, i.e. 0.1GB at 8k tokens and 1.6GB at its full 128k context. Total under 2GB of 48GB.
LOCAL_MODEL_ID = os.environ.get("NEOMME_VLM_LOCAL", "LiquidAI/LFM2.5-VL-450M")
LOCAL_PROVIDER = "Local VLM"
_MAX_NEW_TOKENS = int(os.environ.get("NEOMME_VLM_MAX_NEW_TOKENS", "512"))

PROVIDERS: dict[str, ProviderSpec] = {
    LOCAL_PROVIDER: ProviderSpec(
        label=LOCAL_PROVIDER,
        default_model=LOCAL_MODEL_ID,
        key_hint="no key needed",
        needs_key=False,
    ),
    # The hosted defaults are each provider's mid/low-cost vision model, not its flagship: a visitor is spending
    # their own key on a handful of page images. The Model box overrides them when the ids move on.
    "Anthropic": ProviderSpec(
        label="Anthropic", default_model="claude-sonnet-5", key_hint="Anthropic API key (sk-ant-…)"
    ),
    "OpenAI": ProviderSpec(label="OpenAI", default_model="gpt-5.6-luna", key_hint="OpenAI API key (sk-…)"),
    "Gemini": ProviderSpec(label="Gemini", default_model="gemini-3.6-flash", key_hint="Google AI Studio API key"),
}


def generate_answer(provider: str, api_key: str, model: str, query: str, pages: list[tuple[str, Image.Image]]) -> str:
    """Route the query + retrieved pages to the chosen provider and return the generated answer."""
    if PROVIDERS[provider].needs_key and not api_key.strip():
        raise ValueError("Enter an API key for the selected provider to generate an answer.")
    if not pages:
        raise ValueError("No pages retrieved — build the index and search first.")
    model = model.strip() or PROVIDERS[provider].default_model
    prompt = _build_prompt(query, [label for label, _ in pages])
    images = [image for _, image in pages]
    return _DISPATCH[provider](api_key, model, prompt, images)


def _build_prompt(query: str, labels: list[str]) -> str:
    page_list = "\n".join(f"- page {i + 1}: {label}" for i, label in enumerate(labels))
    return f"{_ANSWER_INSTRUCTIONS}\n\nRetrieved pages (in rank order):\n{page_list}\n\nQuestion: {query}"


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _answer_anthropic(api_key: str, model: str, prompt: str, images: list[Image.Image]) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    content: list[dict] = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.standard_b64encode(_png_bytes(image)).decode("utf-8"),
            },
        }
        for image in images
    ]
    content.append({"type": "text", "text": prompt})
    response = client.messages.create(model=model, max_tokens=1024, messages=[{"role": "user", "content": content}])
    return "".join(block.text for block in response.content if block.type == "text").strip()


def _answer_openai(api_key: str, model: str, prompt: str, images: list[Image.Image]) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    content: list[dict] = [{"type": "text", "text": prompt}]
    for image in images:
        data_uri = "data:image/png;base64," + base64.standard_b64encode(_png_bytes(image)).decode("utf-8")
        content.append({"type": "image_url", "image_url": {"url": data_uri}})
    response = client.chat.completions.create(
        model=model, max_tokens=1024, messages=[{"role": "user", "content": content}]
    )
    return (response.choices[0].message.content or "").strip()


def _answer_gemini(api_key: str, model: str, prompt: str, images: list[Image.Image]) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    parts: list = [prompt]
    parts.extend(types.Part.from_bytes(data=_png_bytes(image), mime_type="image/png") for image in images)
    response = client.models.generate_content(model=model, contents=parts)
    return (response.text or "").strip()


# Module-level placement on cuda is what ZeroGPU wants: a CUDA emulation covers it outside @spaces.GPU, and
# transfers done at startup are far cheaper than lazy ones inside the decorated call.
_ZEROGPU = os.environ.get("SPACES_ZERO_GPU") == "true"


def _load_local():
    """Load the local VLM once, at import. Returns `(None, None)` if it is unavailable or disabled."""
    if not LOCAL_MODEL_ID:
        return None, None
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        device = "cuda" if _ZEROGPU else ("mps" if torch.backends.mps.is_available() else "cpu")
        # fp32 only on cpu, which emulates bf16 (~900x slower matmul); on mps bf16 generates 1.6x faster.
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        processor = AutoProcessor.from_pretrained(LOCAL_MODEL_ID)
        model = AutoModelForImageTextToText.from_pretrained(LOCAL_MODEL_ID, dtype=dtype).to(device).eval()
        return processor, model
    except Exception as error:  # a missing weight, no torchvision, no network: fall back to key-only providers
        print(f"[vlm] local model {LOCAL_MODEL_ID!r} unavailable ({type(error).__name__}: {error})")
        return None, None


local_processor, local_model = _load_local()


@spaces.GPU(duration=int(os.environ.get("NEOMME_VLM_GPU_DURATION", "60")))
def _answer_local(api_key: str, model: str, prompt: str, images: list[Image.Image]) -> str:
    """Generate on the Space's own GPU. `api_key` and `model` are ignored; the signature matches the others."""
    if local_model is None:
        raise RuntimeError(f"the local model ({LOCAL_MODEL_ID}) failed to load; pick a provider and paste a key")

    import torch

    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": prompt})
    inputs = local_processor.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(local_model.device)

    with torch.no_grad():
        generated = local_model.generate(**inputs, max_new_tokens=_MAX_NEW_TOKENS, do_sample=False)
    reply = generated[0, inputs["input_ids"].shape[-1] :]  # drop the prompt, keep the continuation
    return local_processor.decode(reply, skip_special_tokens=True).strip()


_DISPATCH = {
    LOCAL_PROVIDER: _answer_local,
    "Anthropic": _answer_anthropic,
    "OpenAI": _answer_openai,
    "Gemini": _answer_gemini,
}
