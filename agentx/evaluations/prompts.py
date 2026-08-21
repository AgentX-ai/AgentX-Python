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

    def examples(self, prompt_id: str, window: Optional[str] = None) -> dict:
        """The merged evidence (worst eval-run results + low-rated online-evaluator traffic)
        a propose() call will rewrite from - version-scoped to the prompt's current version."""
        return self._client.get_prompt_examples(prompt_id, window=window)

    def propose(self, prompt_id: str) -> dict:
        """Ask the judge for a rewrite grounded in examples(). Returns the proposal
        (revisedText/reasoning/sourceBreakdown) without publishing anything."""
        return self._client.propose_prompt(prompt_id)

    def publish_version(
        self, prompt_id: str, *, text: str, source: str = "proposed",
        reasoning: Optional[str] = None, based_on_version: Optional[int] = None,
    ) -> dict:
        """Publish a new version (the human-approval step of the propose flow)."""
        return self._client.publish_prompt_version(
            prompt_id, text=text, source=source, reasoning=reasoning, based_on_version=based_on_version
        )

    def list(self) -> List[Prompt]:
        return self._client.list_prompts()
