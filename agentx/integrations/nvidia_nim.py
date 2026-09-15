"""
NVIDIA NIM integration for AgentX production tracing.

NIM (NVIDIA Inference Microservices) serves models behind an OpenAI-compatible
``/v1/chat/completions`` API, so the client you patch is the ordinary ``openai``
Python client pointed at a NIM endpoint - a local NIM container
(``http://localhost:8000/v1``) or NVIDIA's hosted API
(``https://integrate.api.nvidia.com/v1``). This module reuses the OpenAI patch
machinery verbatim and differs in exactly one way: traces are stamped
``framework="nvidia-nim"``, so NIM traffic gets its own row in Monitor's
Platforms chart and the framework filters instead of blending into "openai".

Usage::

    from agentx.integrations.nvidia_nim import patch_nim_client
    import openai

    nim = openai.OpenAI(
        base_url="http://localhost:8000/v1",  # or https://integrate.api.nvidia.com/v1
        api_key=os.environ.get("NVIDIA_API_KEY", "not-needed-for-local-nim"),
    )
    patch_nim_client(nim, agentx.tracer, name="nim-agent")

    # All subsequent nim.chat.completions.create() calls are now traced.

Works with both ``openai.OpenAI`` and ``openai.AsyncOpenAI`` clients. Token
usage comes straight off the response's OpenAI-shaped ``usage`` block; NIM
reports no prompt-cache fields, so cache token counts stay unset.

Streaming calls (``stream=True``) are passed through untouched and are not
currently traced - same posture as ``patch_openai_client``, see its docstring.

Requires: ``pip install "agentx-python[nvidia-nim]"`` (installs the ``openai``
client package; there is no separate NIM SDK dependency).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from agentx.tracing.tracer import Tracer
from agentx.integrations.openai import _patch_chat_completions_create

NIM_FRAMEWORK = "nvidia-nim"


def patch_nim_client(
    client: Any,
    tracer: Tracer,
    name: str = "nim-agent",
    metadata: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
) -> None:
    """
    Monkey-patch ``client.chat.completions.create`` on an OpenAI-compatible
    client pointed at a NIM endpoint, sending a trace for every non-streaming
    call with ``framework="nvidia-nim"``.

    The original method is still called and its return value passed through
    unchanged. Sync and async clients both work; ``stream=True`` calls pass
    through untraced. Patching is idempotent - and because it shares the guard
    with ``patch_openai_client``, whichever of the two patched a given client
    first wins (patch each client with the integration that matches where its
    ``base_url`` actually points).
    """
    chat = getattr(client, "chat", None)
    completions = getattr(chat, "completions", None) if chat is not None else None
    if completions is None:
        raise ValueError("Provided client does not have a .chat.completions attribute")

    _patch_chat_completions_create(completions, tracer, name, metadata, session_id, framework=NIM_FRAMEWORK)
