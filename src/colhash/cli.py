"""colhash command line: hash, diff, band."""

from __future__ import annotations

import argparse
import json
import sys

import pyarrow as pa

from . import __version__, core, report


def _keys(value: str | None) -> list[str] | None:
    return [k.strip() for k in value.split(",") if k.strip()] if value else None


def _emit(args: argparse.Namespace, as_dict, as_text, as_markdown) -> None:
    if args.json:
        print(json.dumps(as_dict(), indent=2))
    elif args.markdown:
        print(as_markdown())
    else:
        print(as_text())


def cmd_hash(args: argparse.Namespace) -> int:
    results = [core.hash_file(p, _keys(args.key)) for p in args.files]
    if args.json:
        print(json.dumps(
            [{"path": r.path, "rows": r.rows, "combined": r.combined, "columns": r.columns} for r in results],
            indent=2,
        ))
        return 0
    for r in results:
        print(f"## {r.path}  rows={r.rows}  cols={len(r.columns)}  all={r.combined}")
        width = max((len(c) for c in r.columns), default=0)
        for name, digest in r.columns.items():
            print(f"  {name:<{width}}  {digest}")
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    r = core.diff_report(args.a, args.b, _keys(args.key))
    _emit(args, lambda: report.diff_to_dict(r), lambda: report.diff_text(r), lambda: report.diff_markdown(r))
    return 0 if r.identical else 1


def cmd_band(args: argparse.Namespace) -> int:
    r = core.band_report(args.base1, args.base2, args.candidate, _keys(args.key), tol=args.tol)
    _emit(args, lambda: report.band_to_dict(r), lambda: report.band_text(r), lambda: report.band_markdown(r))
    return 1 if r.failed else 0


def _output_flags(p: argparse.ArgumentParser) -> None:
    group = p.add_mutually_exclusive_group()
    group.add_argument("--json", action="store_true", help="machine-readable output for CI")
    group.add_argument("--markdown", action="store_true", help="a table for a PR description")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="colhash", description=__doc__)
    parser.add_argument("--version", action="version", version=f"colhash {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("hash", help="per-column content hashes")
    p.add_argument("files", nargs="+")
    p.add_argument("--key", help="comma-separated sort key, e.g. id or game,frame")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_hash)

    p = sub.add_parser("diff", help="compare two files column by column (exit 1 if they differ)")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--key", required=True, help="comma-separated columns that identify a row")
    _output_flags(p)
    p.set_defaults(func=cmd_diff)

    p = sub.add_parser(
        "band",
        help="classify a candidate against the baseline's run-to-run noise (exit 1 on changed or missing)",
    )
    p.add_argument("base1", help="baseline run 1")
    p.add_argument("base2", help="baseline run 2: the same unchanged pipeline, run again")
    p.add_argument("candidate")
    p.add_argument("--key", required=True, help="comma-separated columns that identify a row")
    p.add_argument(
        "--tol", type=float, default=0.0, help="numeric differences up to this size are never flagged"
    )
    _output_flags(p)
    p.set_defaults(func=cmd_band)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (KeyError, ValueError, FileNotFoundError, pa.ArrowException) as e:
        print(f"colhash: error: {e.args[0] if isinstance(e, KeyError) and e.args else e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
