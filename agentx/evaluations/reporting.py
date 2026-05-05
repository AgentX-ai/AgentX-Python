from __future__ import annotations

from agentx.evaluations.models import Report


def _rating_badge(rating: str | None) -> str:
    icons = {"high": "●", "medium": "◑", "low": "○"}
    return icons.get(rating or "", "·")


def print_report(report: Report) -> None:
    sep = "=" * 60
    thin = "-" * 40

    print(sep)
    print("  AgentX Custom Agent Evaluation Report")
    print(sep)
    print(f"  Run ID   : {report.run_id}")
    print(f"  Dataset  : {report.dataset_id}")
    print(f"  Status   : {report.status}")

    if report.statistics:
        s = report.statistics
        print(f"  Cases    : {s.number_of_runs}")
        print(f"  Avg rating : {s.average_rating:.1f}  (min {s.min_rating:.1f} / max {s.max_rating:.1f})")

    if report.consistency_score is not None:
        print(f"  Consistency : {report.consistency_score:.1f}/10")

    if report.overall_rating:
        print(f"  Overall  : {_rating_badge(report.overall_rating)} {report.overall_rating.upper()}")

    # --- Summary ---
    if report.summary:
        print()
        print("Summary")
        print(thin)
        print(report.summary)

    # --- Instruction adherence ---
    if report.instruction_adherence:
        ia = report.instruction_adherence
        badge = _rating_badge(ia.rating)
        score_str = f"{ia.score:.1f}/10" if ia.score is not None else ""
        print()
        print(f"Instruction Adherence  {badge} {ia.rating or ''}  {score_str}")
        print(thin)
        if ia.analysis:
            print(ia.analysis)
        if ia.deviations:
            print("  Deviations:")
            for d in ia.deviations:
                print(f"    ! {d}")

    # --- Response patterns ---
    if report.response_patterns:
        rp = report.response_patterns
        print()
        print(f"Response Patterns  {_rating_badge(rp.rating)} {rp.rating or ''}")
        print(thin)
        if rp.similarities:
            print("  Similarities:")
            for s in rp.similarities:
                print(f"    = {s}")
        if rp.differences:
            print("  Differences:")
            for d in rp.differences:
                print(f"    ≠ {d}")
        if rp.outliers:
            print("  Outliers:")
            for o in rp.outliers:
                print(f"    * {o}")

    # --- Reasoning analysis ---
    if report.reasoning_analysis:
        ra = report.reasoning_analysis
        print()
        print(f"Reasoning Analysis  {_rating_badge(ra.rating)} {ra.rating or ''}")
        print(thin)
        if ra.cot_quality:
            print(f"  CoT Quality: {ra.cot_quality}")
        if ra.reasoning_patterns:
            print("  Patterns:")
            for p in ra.reasoning_patterns:
                print(f"    + {p}")
        if ra.reasoning_gaps:
            print("  Gaps:")
            for g in ra.reasoning_gaps:
                print(f"    - {g}")

    # --- Tool usage ---
    if report.tool_usage_analysis:
        tu = report.tool_usage_analysis
        print()
        print(f"Tool Usage  {_rating_badge(tu.rating)} {tu.rating or ''}")
        print(thin)
        if tu.effectiveness:
            print(f"  {tu.effectiveness}")
        if tu.patterns:
            for p in tu.patterns:
                print(f"    + {p}")
        if tu.issues:
            for i in tu.issues:
                print(f"    ! {i}")

    # --- Strengths / Weaknesses ---
    if report.strengths:
        print()
        print("Strengths")
        print(thin)
        for s in report.strengths:
            print(f"  + {s}")

    if report.weaknesses:
        print()
        print("Weaknesses")
        print(thin)
        for w in report.weaknesses:
            print(f"  - {w}")

    # --- Recommendations ---
    if report.recommendations:
        print()
        print("Recommendations")
        print(thin)
        priority_order = {"high": 0, "medium": 1, "low": 2}
        sorted_recs = sorted(
            report.recommendations,
            key=lambda r: priority_order.get(r.priority or "low", 2),
        )
        for rec in sorted_recs:
            pri = f"[{rec.priority.upper()}]" if rec.priority else ""
            cat = f"({rec.category})" if rec.category else ""
            print(f"  {pri} {cat} {rec.recommendation}")
            if rec.reasoning:
                print(f"       → {rec.reasoning}")

    # --- Low-scoring cases ---
    if report.low_scoring_cases:
        print()
        print(f"Low-scoring cases  (rating < 5)")
        print(thin)
        for case in report.low_scoring_cases[:5]:
            q = (case.get("query") or case.get("questionText", ""))[:80]
            rating = case.get("rating", "?")
            justification = case.get("justification", "")
            print(f"  [{rating}] {q}")
            if justification:
                print(f"       {justification[:120]}")

    # --- Dashboard ---
    if report.dashboard_url:
        print()
        print(f"Dashboard: {report.dashboard_url}")

    print(sep)
