"""Rendering for diff and band results: plain text, JSON, and markdown for PR descriptions."""

from __future__ import annotations

import math
from dataclasses import asdict

from .core import BandReport, DiffReport, Verdict


def _num(x: float | None) -> str:
    if x is None:
        return "-"
    if math.isinf(x):
        return "inf"
    return f"{x:.3g}"


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and math.isinf(obj):
        return "inf"
    if isinstance(obj, Verdict):
        return obj.value
    return obj


def diff_to_dict(r: DiffReport) -> dict:
    d = asdict(r)
    d["identical"] = r.identical
    return _json_safe(d)


def band_to_dict(r: BandReport) -> dict:
    d = asdict(r)
    d["failed"] = r.failed
    return _json_safe(d)


def _rows_line(r: DiffReport) -> str:
    return (
        f"{r.rows_matched} rows matched on {','.join(r.key)}; "
        f"{r.only_in_a} only in {r.a}, {r.only_in_b} only in {r.b}"
    )


def diff_text(r: DiffReport) -> str:
    lines = [_rows_line(r)]
    width = max((len(c.column) for c in r.columns), default=6)
    lines.append(f"{'column':<{width}}  {'status':<10} {'rows differ':>11}  {'max abs diff':>12}")
    for c in r.columns:
        if c.missing_in:
            status = f"missing in {c.missing_in}"
        elif c.identical:
            status = "identical"
        elif c.type_a != c.type_b:
            status = f"{c.type_a} -> {c.type_b}"
        else:
            status = "differs"
        lines.append(f"{c.column:<{width}}  {status:<10} {c.n_diff:>11}  {_num(c.max_abs):>12}")
    lines.append("identical" if r.identical else "different")
    return "\n".join(lines)


def band_text(r: BandReport) -> str:
    width = max((len(c.column) for c in r.columns), default=6)
    head = (
        f"{'column':<{width}}  {'verdict':<12} {'rows':>6} {'max abs':>10}"
        f"  {'noise rows':>10} {'noise max abs':>13}"
    )
    lines = [head]
    for c in r.columns:
        lines.append(
            f"{c.column:<{width}}  {c.verdict.value:<12} {c.n_diff:>6} {_num(c.max_abs):>10}"
            f"  {c.noise_n_diff:>10} {_num(c.noise_max_abs):>13}"
        )
    if r.rows_changed:
        lines.append("the candidate has rows the baseline lacks, or lacks rows the baseline has")
    lines.append(
        "FAIL: a column changed or is missing"
        if r.failed
        else "PASS: every column is identical or within the noise band"
    )
    return "\n".join(lines)


_BADGE = {
    Verdict.IDENTICAL: "identical",
    Verdict.WITHIN_BAND: "within band",
    Verdict.CHANGED: "**changed**",
    Verdict.MISSING: "**missing**",
}


def band_markdown(r: BandReport, title: str = "colhash band report") -> str:
    out = [f"### {title}", ""]
    out.append(
        "| column | verdict | rows differ | max abs diff | baseline noise rows | baseline noise max abs |"
    )
    out.append("|---|---|---:|---:|---:|---:|")
    for c in r.columns:
        out.append(
            f"| `{c.column}` | {_BADGE[c.verdict]} | {c.n_diff} | {_num(c.max_abs)} "
            f"| {c.noise_n_diff} | {_num(c.noise_max_abs)} |"
        )
    out.append("")
    if r.rows_changed:
        out.append("The candidate's row set differs from the baseline's.")
        out.append("")
    out.append("**Result: FAIL**" if r.failed else "**Result: PASS**")
    return "\n".join(out)


def diff_markdown(r: DiffReport, title: str = "colhash diff report") -> str:
    out = [f"### {title}", "", _rows_line(r), ""]
    out.append("| column | status | rows differ | max abs diff |")
    out.append("|---|---|---:|---:|")
    for c in r.columns:
        if c.missing_in:
            status = f"missing in {c.missing_in}"
        elif c.identical:
            status = "identical"
        elif c.type_a != c.type_b:
            status = f"type {c.type_a} to {c.type_b}"
        else:
            status = "differs"
        out.append(f"| `{c.column}` | {status} | {c.n_diff} | {_num(c.max_abs)} |")
    out.append("")
    out.append("**Result: identical**" if r.identical else "**Result: different**")
    return "\n".join(out)
