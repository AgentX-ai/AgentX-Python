from __future__ import annotations

from agentx.evaluations.models import Report
from agentx.evaluations._term import (
    bold,
    cyan,
    dim,
    green,
    yellow,
    red,
    magenta,
    RESET,
    BOLD,
)

_SEP = "─" * 60
_THIN = "─" * 40

_RATING_COLORS = {"high": green, "medium": yellow, "low": red}
_RATING_ICONS = {"high": "●", "medium": "◑", "low": "○"}
_PRI_COLORS = {"high": red, "medium": yellow, "low": dim}


def _rating_badge(rating: str | None) -> str:
    icon = _RATING_ICONS.get(rating or "", "·")
    color = _RATING_COLORS.get(rating or "", dim)
    label = (rating or "").upper()
    return color(f"{icon} {label}") if label else dim(icon)


def _section(title: str, rating: str | None = None) -> None:
    badge = f"  {_rating_badge(rating)}" if rating else ""
    print(f"\n{bold(title)}{badge}")
    print(dim(_THIN))


def print_report(report: Report) -> None:
    print(cyan(_SEP))
    print(f"  {bold('AgentX Evaluation Report')}")
    print(cyan(_SEP))
    print(f"  {dim('Run     :')} {dim(report.run_id)}")
    print(f"  {dim('Dataset :')} {dim(report.dataset_id)}")
    print(
        f"  {dim('Status  :')} {green(report.status) if report.status == 'completed' else yellow(report.status)}"
    )

    if report.statistics:
        s = report.statistics
        avg = s.average_rating
        avg_color = green if avg >= 7 else (yellow if avg >= 4 else red)
        print(f"  {dim('Cases   :')} {s.number_of_runs}")
        print(
            f"  {dim('Rating  :')} {avg_color(f'{avg:.1f}/10')}  {dim(f'(min {s.min_rating:.1f} / max {s.max_rating:.1f})')}"
        )

    cos = report.cosine_similarity
    if cos is not None:
        cos_color = green if cos >= 0.85 else (yellow if cos >= 0.6 else red)
        print(
            f"  {dim('Cosine  :')} {cos_color(f'{cos * 100:.1f}%')}  {dim('(vector similarity)')}"
        )

    jac = report.jaccard_similarity
    if jac is not None:
        jac_color = green if jac >= 0.6 else (yellow if jac >= 0.3 else red)
        print(
            f"  {dim('Jaccard :')} {jac_color(f'{jac * 100:.1f}%')}  {dim('(token-set overlap)')}"
        )

    if report.consistency_score is not None:
        cs = report.consistency_score
        cs_color = green if cs >= 7 else (yellow if cs >= 4 else red)
        print(f"  {dim('Consist :')} {cs_color(f'{cs:.1f}/10')}")

    if report.overall_rating:
        print(f"  {dim('Overall :')} {_rating_badge(report.overall_rating)}")

    # --- Summary ---
    if report.summary:
        _section("Summary")
        print(f"  {report.summary}")

    # --- Instruction adherence ---
    if report.instruction_adherence:
        ia = report.instruction_adherence
        score_str = f"  {dim(f'{ia.score:.1f}/10')}" if ia.score is not None else ""
        _section("Instruction Adherence", ia.rating)
        if ia.analysis:
            print(f"  {ia.analysis}")
        if ia.deviations:
            print(f"  {dim('Deviations:')}")
            for d in ia.deviations:
                print(f"    {yellow('!')} {d}")

    # --- Response patterns ---
    if report.response_patterns:
        rp = report.response_patterns
        _section("Response Patterns", rp.rating)
        for s in rp.similarities:
            print(f"    {dim('=')} {s}")
        for d in rp.differences:
            print(f"    {cyan('≠')} {d}")
        for o in rp.outliers:
            print(f"    {yellow('*')} {o}")

    # --- Reasoning analysis ---
    if report.reasoning_analysis:
        ra = report.reasoning_analysis
        _section("Reasoning Analysis", ra.rating)
        if ra.cot_quality:
            print(f"  {dim('CoT:')} {ra.cot_quality}")
        for p in ra.reasoning_patterns:
            print(f"    {green('+')} {p}")
        for g in ra.reasoning_gaps:
            print(f"    {red('-')} {g}")

    # --- Tool usage ---
    if report.tool_usage_analysis:
        tu = report.tool_usage_analysis
        _section("Tool Usage", tu.rating)
        if tu.effectiveness:
            print(f"  {tu.effectiveness}")
        for p in tu.patterns:
            print(f"    {green('+')} {p}")
        for i in tu.issues:
            print(f"    {yellow('!')} {i}")

    # --- Strengths / Weaknesses ---
    if report.strengths:
        _section("Strengths")
        for s in report.strengths:
            print(f"  {green('+')} {s}")

    if report.weaknesses:
        _section("Weaknesses")
        for w in report.weaknesses:
            print(f"  {red('-')} {w}")

    # --- Recommendations ---
    if report.recommendations:
        _section("Recommendations")
        priority_order = {"high": 0, "medium": 1, "low": 2}
        sorted_recs = sorted(
            report.recommendations,
            key=lambda r: priority_order.get(r.priority or "low", 2),
        )
        for rec in sorted_recs:
            pri_fn = _PRI_COLORS.get(rec.priority or "low", dim)
            pri = pri_fn(f"[{rec.priority.upper()}]") if rec.priority else ""
            cat = dim(f"({rec.category})") if rec.category else ""
            print(f"  {pri} {cat} {rec.recommendation}")
            if rec.reasoning:
                print(f"       {dim('→')} {dim(rec.reasoning)}")

    # --- Low-scoring cases ---
    if report.low_scoring_cases:
        _section("Low-scoring Cases  (rating < 5)")
        for case in report.low_scoring_cases[:5]:
            q = (case.get("query") or case.get("questionText", ""))[:80]
            rating = case.get("rating", "?")
            justification = case.get("justification", "")
            print(f"  {red(f'[{rating}]')} {q}")
            if justification:
                print(f"       {dim(justification[:120])}")

    # --- Dashboard ---
    if report.dashboard_url:
        print()
        print(f"  {dim('Dashboard:')} {cyan(report.dashboard_url)}")

    print()
    print(cyan(_SEP))
