"""Time hash, diff and band on a synthetic file.

    python benchmarks/bench.py --rows 10_000_000 --cols 50

Columns are a mix: 40% float64, 30% int64, 20% short strings, 10% timestamps.
The candidate differs from the baseline by a handful of float values.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from colhash import core


def make_chunk(rng, start, n, cols):
    arrays = {"id": pa.array(np.arange(start, start + n, dtype=np.int64))}
    kinds = ["f"] * int(cols * 0.4) + ["i"] * int(cols * 0.3) + ["s"] * int(cols * 0.2)
    kinds += ["t"] * (cols - len(kinds))
    words = np.array([f"word{i:04d}" for i in range(5000)])
    for j, kind in enumerate(kinds):
        name = f"{kind}{j}"
        if kind == "f":
            arrays[name] = pa.array(rng.standard_normal(n))
        elif kind == "i":
            arrays[name] = pa.array(rng.integers(0, 1_000_000, n))
        elif kind == "s":
            arrays[name] = pa.array(words[rng.integers(0, 5000, n)])
        else:
            seconds = rng.integers(1_600_000_000, 1_700_000_000, n)
            arrays[name] = pa.array(seconds * 1_000_000, pa.timestamp("us"))
    return pa.table(arrays)


def write(path, rows, cols, seed, perturb=None, chunk=1_000_000):
    rng = np.random.default_rng(seed)
    writer = None
    for start in range(0, rows, chunk):
        t = make_chunk(rng, start, min(chunk, rows - start), cols)
        if perturb and start == 0:
            col = next(c for c in t.column_names if c.startswith("f"))
            vals = t.column(col).to_numpy().copy()
            vals[:5] += perturb
            t = t.set_column(t.column_names.index(col), col, pa.array(vals))
        writer = writer or pq.ParquetWriter(path, t.schema, compression="snappy")
        writer.write_table(t)
    writer.close()


def timed(label, fn):
    t0 = time.perf_counter()
    out = fn()
    print(f"{label:<28}{time.perf_counter() - t0:8.1f} s")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=lambda s: int(s.replace("_", "")), default=10_000_000)
    ap.add_argument("--cols", type=int, default=50)
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as d:
        a, b, c = (str(Path(d) / f"{n}.parquet") for n in "abc")
        timed("write baseline", lambda: write(a, args.rows, args.cols, 0))
        timed("write baseline run 2", lambda: write(b, args.rows, args.cols, 0, perturb=1e-9))
        timed("write candidate", lambda: write(c, args.rows, args.cols, 0, perturb=1e-3))
        print(f"{args.rows:,} rows x {args.cols} columns, {Path(a).stat().st_size / 1e6:.0f} MB on disk\n")
        timed("hash (1 file, with key)", lambda: core.hash_file(a, ["id"]))
        timed("diff (2 files)", lambda: core.diff_report(a, b, ["id"]))
        timed("band (3 files)", lambda: core.band_report(a, b, c, ["id"]))


if __name__ == "__main__":
    main()
