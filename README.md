# colhash

**Did my change alter the pipeline's output, and if it did, is that real or noise?**

You refactor a pipeline, speed it up, or upgrade a library, and the output should be the same. Two things make that hard to check:

1. **Bytes are the wrong comparison.** The same data written with a different codec, row-group size or row order produces a different parquet file. colhash hashes values, column by column, so only real differences show up.
2. **Some pipelines aren't deterministic.** GPU kernels, randomized estimators and floating-point summation order can change the last bits of the output on every run, with no code change at all. So "byte-identical" is too strict, and "close enough" has no definition.

colhash answers the second problem with a **noise band**. Run the unchanged pipeline twice to measure how much it varies on its own. A candidate passes if it stays inside that band. You don't pick a tolerance; the pipeline sets its own.

| Verdict | Meaning |
|---|---|
| `identical` | Same values and type as the baseline |
| `within_band` | Differs, but in no more rows and by no more than the baseline differs from itself |
| `changed` | Outside the band, or the type changed: a real behavior change that needs review |
| `missing` | The column was added or removed |

## Install

```bash
git clone https://github.com/cvbourne/colhash && cd colhash
pip install -e '.[dev]'
```

Python 3.10 or newer. It depends on `pyarrow` and `numpy` and reads parquet files only.

## Commands

```bash
# Per-column hashes. With --key, rows are sorted by the key first, so row order doesn't matter.
colhash hash before.parquet after.parquet --key id

# What changed, column by column. Exit 1 if anything differs.
colhash diff before.parquet after.parquet --key id

# Real change or noise? Exit 1 if any column is changed or missing.
colhash band baseline_run1.parquet baseline_run2.parquet candidate.parquet --key id
```

`--key` is a comma-separated list of columns that identify a row (`--key game,frame`). `diff` and `band` need one; it must be unique and free of nulls, and colhash raises an error rather than guess. Exit code 2 means a usage or file error.

`diff` and `band` take `--json` for CI and `--markdown` for a table you can paste into a PR description. `band --tol 1e-9` sets an absolute floor: numeric differences no larger than that are never flagged.

## A worked example

`examples/make_example.py` builds a made-up pipeline that totals each user's purchases. It sums floats in a different order on each run, which is what a parallel job does, so two runs of the same pipeline disagree in the last bits.

```bash
python examples/make_example.py examples/data
cd examples/data
```

Two runs of the unchanged pipeline are not byte-for-byte the same:

```console
$ colhash diff baseline_1.parquet baseline_2.parquet --key user_id
2000 rows matched on user_id; 0 only in baseline_1.parquet, 0 only in baseline_2.parquet
column       status     rows differ  max abs diff
total_spend  differs           1760      9.09e-12
n_purchases  identical            0             0
tier         identical            0             -
different
```

That is the noise: 1,760 totals move, by at most 9e-12. Now check a refactor that should change nothing:

```console
$ colhash band baseline_1.parquet baseline_2.parquet refactor.parquet --key user_id
column       verdict        rows    max abs  noise rows noise max abs
total_spend  within_band    1718   9.09e-12        1760      9.09e-12
n_purchases  identical         0          0           0             0
tier         identical         0          -           0             -
PASS: every column is identical or within the noise band
```

The refactor differs from the baseline in 1,718 rows, but by no more than the pipeline differs from itself, so it passes. Now a rewrite that silently drops purchases under $0.05:

```console
$ colhash band baseline_1.parquet baseline_2.parquet bug.parquet --key user_id
column       verdict        rows    max abs  noise rows noise max abs
total_spend  changed        1703     0.0799        1760      9.09e-12
n_purchases  changed         175          2           0             0
tier         identical         0          -           0             -
FAIL: a column changed or is missing
```

`total_spend` moved in fewer rows than the noise did, but by 0.08 instead of 9e-12, so it fails. No tolerance was chosen for any of these checks.

## How the band works

1. `diff(baseline_1, baseline_2)` measures how much the unchanged pipeline varies: per column, how many rows differ and by how much at most.
2. `diff(baseline_1, candidate)` measures how far the change moved the output.
3. A numeric column is `within_band` only if it passes **both** measures: no more rows differ than in the noise, and no difference is larger than the noise's largest. Both are needed. Noise that touches many rows by a tiny amount and a change that hits one row by a lot are different things, and either measure alone would pass one of them.

The edges:

- **Non-numeric columns** (strings, booleans, timestamps, nested types) have no magnitude, so only the row count is compared, and only if the baseline itself varies on that column. Otherwise any difference is `changed`.
- **A deterministic baseline** has a band of zero. Any difference is `changed`, which is the right answer.
- **A null or NaN against a value** has no finite size, so its magnitude is `inf` and it fails any band that didn't see the same thing in the baseline.
- **Rows.** If the candidate has rows the baseline lacks, or lacks rows it has, beyond what the baseline's own runs disagree on, every column is `changed`.
- **Types.** A changed column type is `changed` even if the numbers agree: `int32` to `int64` is a schema change.

## Hashing rules

- Values, not file bytes: codec, dictionary encoding, row groups and chunk boundaries don't matter.
- With `--key`, rows are sorted by the key first, so row order doesn't matter.
- All NaNs hash alike, and `-0.0` hashes like `0.0`. A null is never equal to 0 or to an empty string.
- The column type is part of the hash: `int32` and `int64` columns holding the same numbers differ.

## Use it in CI

```yaml
- run: colhash band baseline_1.parquet baseline_2.parquet candidate.parquet --key id --markdown >> "$GITHUB_STEP_SUMMARY"
```

The step fails when a column changed or is missing, and the summary shows the table. Regenerate the two baseline runs when the code you consider the baseline changes.

## Python API

```python
from colhash import core

report = core.band_report("base1.parquet", "base2.parquet", "candidate.parquet", key=["id"])
report.failed          # True if any column is changed or missing
report.verdicts        # {"score": Verdict.WITHIN_BAND, ...}

core.diff_report("a.parquet", "b.parquet", key=["id"])
core.hash_file("a.parquet", key=["id"]).columns
```

## Performance

On a 10M-row, 50-column file (3 GB on disk; 40% float, 30% int, 20% string, 10% timestamp columns), on a 10-core laptop with 24 GB of RAM:

| Command | Time |
|---|---|
| `hash` (one file) | 9.9 s |
| `diff` (two files) | 10.4 s |
| `band` (three files) | 17.8 s |

Reproduce with `python benchmarks/bench.py --rows 10_000_000 --cols 50`.

**Memory is the limit.** Files are read whole, and the peak was about 10.8 GB for that 3 GB file, roughly 3 to 4 times its size. Files that don't fit in memory are not supported yet.

## Limits

- **Two baseline runs are one sample of the noise.** The band is the largest variation seen between those two runs, so a pipeline that varies more on a third run could fail a candidate that is fine. Using more than two baseline runs is not built.
- **Parquet only.** No CSV, Delta, Iceberg, Spark or warehouse tables.
- **No schema migrations.** A renamed column is one `missing` column and one new one.
- **Whole-file reads.** See the memory note above.

## Prior art

As far as I could find in November 2026, these tools compare data but none of them measure a pipeline's own noise and use it as the tolerance:

- [datacompy](https://github.com/capitalone/datacompy) compares dataframes, including parquet inputs, with absolute and relative tolerances that you set.
- [data-diff](https://github.com/datafold/data-diff) compared tables across SQL databases; Datafold archived it in May 2024.
- [syrupy](https://github.com/syrupy-project/syrupy) is snapshot testing with exact comparison.

If you know of one that does, please open an issue.

## Development

```bash
pip install -e '.[dev]'
pytest
ruff check .
```

## License

MIT
