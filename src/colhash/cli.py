"""colhash command line."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__, core


def _keys(value: str | None) -> list[str] | None:
    return [k.strip() for k in value.split(",") if k.strip()] if value else None


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="colhash", description=__doc__)
    parser.add_argument("--version", action="version", version=f"colhash {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("hash", help="per-column content hashes")
    p.add_argument("files", nargs="+")
    p.add_argument("--key", help="comma-separated sort key, e.g. id or game,frame")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_hash)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
