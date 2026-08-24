"""Hold the documentation against the SDK it documents.

0.6.36 consolidated grading configs into an LLM Judge Scorer and shipped with no doc
coverage at all: EVALUATIONS.md, TRACING.md, README.md and CICD_EVAL.md had no mention of
a judge scorer between them. Nothing caught that, because nothing checks the docs.

This does. It extracts the fenced python from the docs and resolves what they show
against the installed package - the methods, the keyword arguments, and the
cross-document links. A renamed kwarg or a dropped method is a failing test here rather
than a copy-pasted snippet that raises TypeError in someone's project.

Prose is deliberately out of scope; only fenced ```python blocks are read, which is what
a reader actually copies.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from agentx.evaluations.client import EvaluationsClient, _resolve_scorer_id
from agentx.evaluations.runner import EvaluationsRunner
from agentx.monitor.judge_scorers import JudgeScorerBuilder, JudgeScorersClient

ROOT = Path(__file__).resolve().parent.parent
DOCS = ("EVALUATIONS.md", "TRACING.md", "README.md", "CICD_EVAL.md")

BUILDER_PARAMS = set(inspect.signature(JudgeScorersClient.builder).parameters) - {"self"}
RUN_PARAMS = set(inspect.signature(EvaluationsRunner.run).parameters) - {"self"}

# A regex that quietly stops matching turns this file into a green light wired to nothing,
# so each sweep asserts it still found roughly what it found when written.
MIN_JUDGE_SCORER_CALLS = 10
MIN_BUILDER_KEYWORDS = 10
MIN_RUN_KEYWORDS = 3


def _fenced_python(text: str) -> str:
    return "\n".join(re.findall(r"```python\n(.*?)```", text, re.S))


def _calls(src: str, prefix: str) -> list[tuple[str, str]]:
    """(method, argument-text) for each ``<prefix>.<method>(...)``, parens balanced.

    The docs are not importable - they carry placeholders like ``subject={...}`` - so this
    reads them the way a reader does, by eye, rather than by parsing them as Python.
    """
    found = []
    for match in re.finditer(re.escape(prefix) + r"\s*\.\s*(\w+)\(", src):
        open_paren = match.end() - 1
        depth, index = 0, open_paren
        while index < len(src):
            if src[index] == "(":
                depth += 1
            elif src[index] == ")":
                depth -= 1
                if depth == 0:
                    break
            index += 1
        found.append((match.group(1), src[open_paren + 1 : index]))
    return found


def _keywords(argument_text: str) -> list[str]:
    """Keyword names in a call's argument text, ignoring keys inside a value's own dict."""
    return re.findall(r"(?:^|[(,]\s*|\n\s*)(\w+)\s*=", argument_text)


def _documented() -> dict[str, str]:
    return {name: _fenced_python((ROOT / name).read_text()) for name in DOCS}


def _judge_scorer_calls() -> list[tuple[str, str, str]]:
    return [
        (doc, method, args)
        for doc, src in _documented().items()
        for method, args in _calls(src, "client.monitor.judge_scorers")
    ]


def test_documented_judge_scorer_methods_exist():
    calls = _judge_scorer_calls()
    assert len(calls) >= MIN_JUDGE_SCORER_CALLS, (
        f"only found {len(calls)} judge_scorers calls in the docs - the sweep is no longer "
        "finding what it should, or the surface stopped being documented"
    )
    missing = [
        f"{doc}: client.monitor.judge_scorers.{method}()"
        for doc, method, _ in calls
        if not hasattr(JudgeScorersClient, method)
    ]
    assert not missing, "documented but not on JudgeScorersClient: " + ", ".join(missing)


def test_documented_builder_keywords_are_real_parameters():
    keywords = [
        (doc, keyword)
        for doc, method, args in _judge_scorer_calls()
        if method == "builder"
        for keyword in _keywords(args)
    ]
    assert len(keywords) >= MIN_BUILDER_KEYWORDS, (
        f"only found {len(keywords)} builder keywords in the docs - the sweep is no longer "
        "finding what it should"
    )
    unknown = [f"{doc}: builder({kw}=...)" for doc, kw in keywords if kw not in BUILDER_PARAMS]
    assert not unknown, "documented but not a builder parameter: " + ", ".join(unknown)


def test_documented_run_keywords_are_real_parameters():
    keywords = [
        (doc, keyword)
        for doc, src in _documented().items()
        for method, args in _calls(src, "client.evaluations")
        if method == "run"
        for keyword in _keywords(args)
    ]
    assert len(keywords) >= MIN_RUN_KEYWORDS, (
        f"only found {len(keywords)} evaluations.run keywords in the docs - the sweep is no "
        "longer finding what it should"
    )
    unknown = [f"{doc}: run({kw}=...)" for doc, kw in keywords if kw not in RUN_PARAMS]
    assert not unknown, "documented but not a run() parameter: " + ", ".join(unknown)


def test_builder_publish_is_documented_and_real():
    """Every builder example ends in .publish(); it has to be there."""
    assert any(".publish()" in src for src in _documented().values())
    assert hasattr(JudgeScorerBuilder, "publish")


def test_legacy_views_still_exist():
    """The docs tell readers the pre-consolidation clients keep working. They must."""
    assert hasattr(EvaluationsClient, "settings")
    from agentx.monitor.online_evaluators import MonitorOnlineEvaluatorClient

    assert hasattr(MonitorOnlineEvaluatorClient, "builder")


def test_both_grader_spellings_resolve_to_one_id():
    """EVALUATIONS.md states scorer_id and evaluation_settings_id are the same id, and
    that passing two different ids raises. Both halves are load-bearing for readers
    choosing which to write."""
    assert "scorer_id" in RUN_PARAMS and "evaluation_settings_id" in RUN_PARAMS
    assert _resolve_scorer_id("abc", None) == "abc"
    assert _resolve_scorer_id(None, "abc") == "abc"
    assert _resolve_scorer_id("abc", "abc") == "abc"
    with pytest.raises(ValueError):
        _resolve_scorer_id("abc", "def")


@pytest.mark.parametrize(
    "link, target, heading",
    [
        (
            "EVALUATIONS.md#llm-judge-scorers---reusable-grading-configs",
            "EVALUATIONS.md",
            "### LLM Judge Scorers - reusable grading configs",
        ),
        (
            "TRACING.md#clientmonitoronline_evaluators-self-host-only",
            "TRACING.md",
            "### `client.monitor.online_evaluators` (self-host only)",
        ),
    ],
)
def test_cross_document_links_resolve(link, target, heading):
    """A renamed heading silently breaks every link pointing at it."""
    linking = [name for name in DOCS if link in (ROOT / name).read_text()]
    assert linking, f"nothing links to {link} any more - drop this case or fix the link"
    assert heading in (ROOT / target).read_text(), (
        f"{linking} link to {link}, but {target} has no heading rendering to that anchor"
    )
