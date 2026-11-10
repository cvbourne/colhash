# Changelog

## 0.1.0

First release.

- `colhash hash`: per-column value hashes, independent of codec, dictionary encoding, row groups and (with `--key`) row order.
- `colhash diff`: per-column comparison of two files aligned on a key, with row counts, the largest absolute difference, and rows found in only one file.
- `colhash band`: classify a candidate against the baseline's own run-to-run noise as `identical`, `within_band`, `changed` or `missing`.
- `--json` and `--markdown` output, and exit codes for CI.
- Faster hashing for string, binary, list and struct columns.
