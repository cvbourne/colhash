"""Hashing across column types: encoding must not matter, values and types must."""

import pyarrow as pa
import pyarrow.parquet as pq

from colhash import core


def h(arr):
    return core.column_hash(arr)


def test_dictionary_encoding_does_not_change_hash():
    plain = pa.array(["a", "b", "a", None])
    assert h(plain) == h(plain.dictionary_encode())


def test_chunk_boundaries_do_not_change_hash():
    whole = pa.array(["alpha", "b", None, "gamma"])
    chunked = pa.chunked_array([whole[:1], whole[1:3], whole[3:]])
    assert h(whole) == h(chunked)


def test_slice_does_not_change_hash():
    full = pa.array(["x", "alpha", "b", "gamma", "y"])
    assert h(full.slice(1, 3)) == h(pa.array(["alpha", "b", "gamma"]))


def test_string_value_change_is_detected():
    assert h(pa.array(["a", "b"])) != h(pa.array(["a", "c"]))


def test_string_boundaries_matter():
    # "ab","c" and "a","bc" have the same concatenated bytes.
    assert h(pa.array(["ab", "c"])) != h(pa.array(["a", "bc"]))


def test_null_string_is_not_empty_string():
    assert h(pa.array([None, "x"])) != h(pa.array(["", "x"]))


def test_int_width_is_part_of_the_hash():
    assert h(pa.array([1, 2], pa.int32())) != h(pa.array([1, 2], pa.int64()))


def test_float_precision_is_part_of_the_hash():
    assert h(pa.array([1.0], pa.float32())) != h(pa.array([1.0], pa.float64()))


def test_booleans():
    assert h(pa.array([True, False, None])) == h(pa.array([True, False, None]))
    assert h(pa.array([True, False])) != h(pa.array([False, True]))


def test_temporal_types():
    ts = pa.array([1, 2, None], pa.timestamp("us", tz="UTC"))
    assert h(ts) == h(pa.array([1, 2, None], pa.timestamp("us", tz="UTC")))
    assert h(ts) != h(pa.array([1, 3, None], pa.timestamp("us", tz="UTC")))
    assert h(pa.array([1, 2], pa.date32())) != h(pa.array([1, 3], pa.date32()))
    assert h(pa.array([1, 2], pa.duration("s"))) != h(pa.array([1, 3], pa.duration("s")))


def test_nested_lists_and_structs():
    lists = pa.array([[1, 2], [], None, [3]])
    assert h(lists) == h(pa.array([[1, 2], [], None, [3]]))
    assert h(lists) != h(pa.array([[1, 2], [], None, [4]]))
    # [[1],[2]] and [[1,2],[]] flatten to the same values: lengths must separate them.
    assert h(pa.array([[1], [2]])) != h(pa.array([[1, 2], []]))
    structs = pa.array([{"a": 1, "b": "x"}, None, {"a": 2, "b": None}])
    assert h(structs) == h(pa.array([{"a": 1, "b": "x"}, None, {"a": 2, "b": None}]))
    assert h(structs) != h(pa.array([{"a": 1, "b": "x"}, None, {"a": 3, "b": None}]))


def test_empty_and_all_null_columns():
    assert h(pa.array([], pa.string())) == h(pa.array([], pa.string()))
    assert h(pa.array([], pa.string())) != h(pa.array([], pa.int64()))
    assert h(pa.array([None, None], pa.float64())) != h(pa.array([0.0, 0.0]))
    assert h(pa.array([None, None], pa.string())) != h(pa.array(["", ""]))


def test_unusual_types_still_hash():
    from decimal import Decimal
    d = pa.array([Decimal("1.50"), None], pa.decimal128(5, 2))
    assert h(d) == h(pa.array([Decimal("1.50"), None], pa.decimal128(5, 2)))
    assert h(d) != h(pa.array([Decimal("1.51"), None], pa.decimal128(5, 2)))


def test_roundtrip_through_parquet_with_every_codec(tmp_path):
    t = pa.table({"x": pa.array([1.5, None, -0.0]), "s": pa.array(["a", None, "ccc"])})
    seen = set()
    for codec in ("none", "snappy", "zstd", "gzip"):
        path = str(tmp_path / f"{codec}.parquet")
        pq.write_table(t, path, compression=codec, use_dictionary=codec != "gzip")
        seen.add(core.hash_file(path).combined)
    assert len(seen) == 1
