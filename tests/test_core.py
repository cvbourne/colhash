import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from colhash import core


def _table(**overrides):
    cols = {
        "id": pa.array([3, 1, 2], pa.int64()),
        "score": pa.array([0.5, 1.25, -2.0], pa.float64()),
        "label": pa.array(["c", "a", "b"]),
        "tags": pa.array([["x"], [], ["y", "z"]]),
    }
    cols.update(overrides)
    return pa.table(cols)


def _write(tmp_path, name, table, **kwargs):
    path = tmp_path / name
    pq.write_table(table, path, **kwargs)
    return str(path)


def test_same_data_different_encoding_hashes_equal(tmp_path):
    a = _write(tmp_path, "a.parquet", _table(), compression="snappy")
    b = _write(tmp_path, "b.parquet", _table(), compression="zstd", row_group_size=1)
    assert core.hash_file(a).columns == core.hash_file(b).columns


def test_row_order_is_ignored_with_key(tmp_path):
    t = _table()
    a = _write(tmp_path, "a.parquet", t)
    b = _write(tmp_path, "b.parquet", t.take([2, 0, 1]))
    assert core.hash_file(a).columns != core.hash_file(b).columns
    assert core.hash_file(a, ["id"]).combined == core.hash_file(b, ["id"]).combined


def test_one_changed_value_flags_only_that_column(tmp_path):
    a = _write(tmp_path, "a.parquet", _table())
    b = _write(tmp_path, "b.parquet", _table(score=pa.array([0.5, 1.25, -2.0000001])))
    ha, hb = core.hash_file(a).columns, core.hash_file(b).columns
    assert [c for c in ha if ha[c] != hb[c]] == ["score"]


def test_nan_and_negative_zero_are_canonical():
    x = pa.array([np.nan, -0.0, 1.0])
    y = pa.array([float("nan"), 0.0, 1.0])
    assert core.column_hash(x) == core.column_hash(y)


def test_null_is_not_zero():
    assert core.column_hash(pa.array([None, 1], pa.int64())) != core.column_hash(pa.array([0, 1], pa.int64()))


def test_missing_key_raises(tmp_path):
    a = _write(tmp_path, "a.parquet", _table())
    with pytest.raises(KeyError):
        core.hash_file(a, ["frame"])
