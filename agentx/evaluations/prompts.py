from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from agentx.evaluations.models import Prompt

if TYPE_CHECKING:
    from agentx.evaluations.client import EvaluationsClient


class PromptClient:
    """Thin wrapper surfaced as ``client.evaluations.prompts`` - the external-agent analog to
    AgentX's native autotune. AgentX doesn't own your agent's code, so instead of branching and
    applying a config, it becomes the prompt's source of truth (the same shape as LangSmith's
    Prompt Hub / Langfuse's Prompt Management): create/pull versions here, use ``prompt.text`` as
    your agent's own system prompt, and tag your eval runs with the pulled version so the
    existing version-comparison view (``client.evaluations`` run comparisons on a dataset) can
    tell you which published version actually scored higher.

    Deliberately read-mostly from here: there is no ``publish`` on this client. A prompt only
    gets a new version through the dashboard's human-approved propose/publish flow, so a
    rewritten prompt never reaches your running agent without someone explicitly approving it.

    Example::

        prompt = client.evaluations.prompts.get("support-agent-system-prompt")
        # ... call your own LLM using prompt.text as the system prompt ...
        client.evaluations.init_run(
            dataset_id=dataset_id,
            subject=EvaluationSubject(
                metadata={
                    "promptName": prompt.name,
                    "version": f"{prompt.name}@v{prompt.version}",
                }
            ),
        )

    ``get()`` also accepts a prompt's ``id`` in place of its name, e.g.
    ``client.evaluations.prompts.get(prompt.id)``.
    """

    def __init__(self, client: "EvaluationsClient"):
        self._client = client

    def create(self, name: str, text: str, description: Optional[str] = None) -> Prompt:
        return self._client.create_prompt({"name": name, "text": text, "description": description})

    def get(self, name: str, version: Optional[int] = None) -> Prompt:
        """Accepts either the prompt's name or its ``id`` (e.g. ``prompt.id`` from an earlier
        ``get``/``create`` call), useful for round-tripping an id you already have without a
        second lookup method."""
        return self._client.get_prompt(name, version=version)

    def list(self) -> List[Prompt]:
        return self._client.list_prompts()
