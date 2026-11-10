"""diff and band: alignment, noise, and the places they should refuse to guess."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from colhash import core
from colhash.core import Verdict


def write(tmp_path, name, **cols):
    path = str(tmp_path / name)
    pq.write_table(pa.table(cols), path)
    return path


def base(**over):
    cols = {
        "id": pa.array([1, 2, 3, 4], pa.int64()),
        "x": pa.array([1.0, 2.0, 3.0, 4.0]),
        "s": pa.array(["a", "b", "c", "d"]),
    }
    cols.update(over)
    return cols


# ---- diff

def test_diff_identical_files(tmp_path):
    a, b = write(tmp_path, "a.parquet", **base()), write(tmp_path, "b.parquet", **base())
    r = core.diff_report(a, b, ["id"])
    assert r.identical and r.rows_matched == 4 and r.only_in_a == r.only_in_b == 0


def test_diff_aligns_on_key_not_position(tmp_path):
    a = write(tmp_path, "a.parquet", **base())
    shuffled = {k: v.take([3, 1, 0, 2]) for k, v in base().items()}
    b = write(tmp_path, "b.parquet", **shuffled)
    assert core.diff_report(a, b, ["id"]).identical


def test_diff_counts_rows_in_only_one_file(tmp_path):
    a = write(tmp_path, "a.parquet", **base())
    b = write(tmp_path, "b.parquet", **{k: v.take([0, 1]) for k, v in base().items()})
    r = core.diff_report(a, b, ["id"])
    assert (r.rows_matched, r.only_in_a, r.only_in_b) == (2, 2, 0)
    assert not r.identical


def test_diff_composite_key(tmp_path):
    cols = {"g": pa.array([1, 1, 2]), "f": pa.array([1, 2, 1]), "v": pa.array([1.0, 2.0, 3.0])}
    a = write(tmp_path, "a.parquet", **cols)
    b = write(tmp_path, "b.parquet", **{**cols, "v": pa.array([1.0, 2.5, 3.0])})
    by = {d.column: d for d in core.diff(a, b, ["g", "f"])}
    assert by["v"].n_diff == 1 and by["v"].max_abs == pytest.approx(0.5)


def test_duplicate_key_raises(tmp_path):
    a = write(tmp_path, "a.parquet", **base(id=pa.array([1, 1, 2, 3])))
    b = write(tmp_path, "b.parquet", **base())
    with pytest.raises(ValueError, match="duplicate"):
        core.diff(a, b, ["id"])


def test_null_key_raises(tmp_path):
    a = write(tmp_path, "a.parquet", **base(id=pa.array([1, None, 3, 4])))
    b = write(tmp_path, "b.parquet", **base())
    with pytest.raises(ValueError, match="nulls"):
        core.diff(a, b, ["id"])


def test_missing_key_column_raises(tmp_path):
    a = write(tmp_path, "a.parquet", **base())
    with pytest.raises(KeyError):
        core.diff(a, a, ["nope"])


def test_diff_requires_a_key(tmp_path):
    a = write(tmp_path, "a.parquet", **base())
    with pytest.raises(ValueError):
        core.diff(a, a, [])


def test_nan_inf_and_null_rules(tmp_path):
    a = write(tmp_path, "a.parquet", **base(x=pa.array([np.nan, -0.0, None, 1.0])))
    b = write(tmp_path, "b.parquet", **base(x=pa.array([np.nan, 0.0, None, 1.0])))
    assert core.diff_report(a, b, ["id"]).identical  # NaN==NaN, -0.0==0.0, null==null


def test_null_against_value_is_infinite_magnitude(tmp_path):
    a = write(tmp_path, "a.parquet", **base(x=pa.array([1.0, 2.0, None, 4.0])))
    b = write(tmp_path, "b.parquet", **base(x=pa.array([1.0, 2.0, 0.0, 4.0])))
    d = {c.column: c for c in core.diff(a, b, ["id"])}["x"]
    assert d.n_diff == 1 and d.max_abs == float("inf")


def test_integer_diffs_are_exact(tmp_path):
    big = 2**60
    a = write(tmp_path, "a.parquet", **base(x=pa.array([big, 1, 1, 1], pa.int64())))
    b = write(tmp_path, "b.parquet", **base(x=pa.array([big + 1, 1, 1, 1], pa.int64())))
    d = {c.column: c for c in core.diff(a, b, ["id"])}["x"]
    assert d.n_diff == 1  # float64 would round this away


def test_column_in_one_file_only_is_missing(tmp_path):
    a = write(tmp_path, "a.parquet", **base())
    without_s = {k: v for k, v in base().items() if k != "s"}
    b = write(tmp_path, "b.parquet", **without_s, extra=pa.array([1, 2, 3, 4]))
    by = {c.column: c for c in core.diff(a, b, ["id"])}
    assert by["s"].missing_in == "b" and by["extra"].missing_in == "a"


def test_changed_type_is_a_change_even_when_values_agree(tmp_path):
    a = write(tmp_path, "a.parquet", **base(x=pa.array([1, 2, 3, 4], pa.int32())))
    b = write(tmp_path, "b.parquet", **base(x=pa.array([1, 2, 3, 4], pa.int64())))
    d = {c.column: c for c in core.diff(a, b, ["id"])}["x"]
    assert not d.identical and d.n_diff == 0 and (d.type_a, d.type_b) == ("int32", "int64")


def test_string_and_nested_columns_compare_by_equality(tmp_path):
    a = write(tmp_path, "a.parquet", **base(tags=pa.array([[1], [2], [], None])))
    b = write(
        tmp_path, "b.parquet",
        **base(s=pa.array(["a", "B", "c", None]), tags=pa.array([[1], [9], [], None])),
    )
    by = {c.column: c for c in core.diff(a, b, ["id"])}
    assert by["s"].n_diff == 2 and by["s"].max_abs is None
    assert by["tags"].n_diff == 1


def test_diff_handles_empty_files(tmp_path):
    empty = {"id": pa.array([], pa.int64()), "x": pa.array([], pa.float64())}
    a, b = write(tmp_path, "a.parquet", **empty), write(tmp_path, "b.parquet", **empty)
    assert core.diff_report(a, b, ["id"]).identical


# ---- band

def noisy(tmp_path, name, x):
    return write(tmp_path, name, **base(x=pa.array(x)))


def test_band_identical_when_candidate_matches_baseline(tmp_path):
    b1 = noisy(tmp_path, "b1.parquet", [1.0, 2.0, 3.0, 4.0])
    b2 = noisy(tmp_path, "b2.parquet", [1.0, 2.0, 3.0 + 1e-9, 4.0])
    c = noisy(tmp_path, "c.parquet", [1.0, 2.0, 3.0, 4.0])
    assert core.band(b1, b2, c, ["id"])["x"] == Verdict.IDENTICAL


def test_band_needs_both_magnitude_and_count(tmp_path):
    # Noise: one row moves by 1e-6.
    b1 = noisy(tmp_path, "b1.parquet", [1.0, 2.0, 3.0, 4.0])
    b2 = noisy(tmp_path, "b2.parquet", [1.0, 2.0, 3.0 + 1e-6, 4.0])
    # Many rows move by less than the noise magnitude: too many rows.
    many = noisy(tmp_path, "many.parquet", [1.0 + 1e-8, 2.0 + 1e-8, 3.0 + 1e-8, 4.0])
    # One row moves by far more than the noise: too large.
    big = noisy(tmp_path, "big.parquet", [1.0, 2.0, 3.0 + 1e-3, 4.0])
    fine = noisy(tmp_path, "fine.parquet", [1.0, 2.0 + 1e-7, 3.0, 4.0])
    verdicts = {n: core.band(b1, b2, p, ["id"])["x"] for n, p in
                [("many", many), ("big", big), ("fine", fine)]}
    assert verdicts == {"many": Verdict.CHANGED, "big": Verdict.CHANGED, "fine": Verdict.WITHIN_BAND}


def test_deterministic_baseline_has_a_zero_band(tmp_path):
    b1 = noisy(tmp_path, "b1.parquet", [1.0, 2.0, 3.0, 4.0])
    b2 = noisy(tmp_path, "b2.parquet", [1.0, 2.0, 3.0, 4.0])
    c = noisy(tmp_path, "c.parquet", [1.0, 2.0, 3.0 + 1e-12, 4.0])
    assert core.band(b1, b2, c, ["id"])["x"] == Verdict.CHANGED


def test_tol_is_an_absolute_floor(tmp_path):
    b1 = noisy(tmp_path, "b1.parquet", [1.0, 2.0, 3.0, 4.0])
    b2 = noisy(tmp_path, "b2.parquet", [1.0, 2.0, 3.0, 4.0])
    c = noisy(tmp_path, "c.parquet", [1.0 + 1e-9, 2.0 + 1e-9, 3.0 + 1e-9, 4.0 + 1e-9])
    assert core.band(b1, b2, c, ["id"])["x"] == Verdict.CHANGED
    assert core.band(b1, b2, c, ["id"], tol=1e-6)["x"] == Verdict.WITHIN_BAND


def test_non_numeric_column_is_identical_or_changed_unless_baseline_varies(tmp_path):
    b1 = write(tmp_path, "b1.parquet", **base())
    b2 = write(tmp_path, "b2.parquet", **base())
    c = write(tmp_path, "c.parquet", **base(s=pa.array(["a", "b", "c", "X"])))
    assert core.band(b1, b2, c, ["id"])["s"] == Verdict.CHANGED
    b2v = write(tmp_path, "b2v.parquet", **base(s=pa.array(["a", "b", "Y", "d"])))
    assert core.band(b1, b2v, c, ["id"])["s"] == Verdict.WITHIN_BAND  # one row varies on its own


def test_missing_and_extra_columns(tmp_path):
    b1 = write(tmp_path, "b1.parquet", **base())
    b2 = write(tmp_path, "b2.parquet", **base())
    c = write(tmp_path, "c.parquet", id=base()["id"], x=base()["x"], new=pa.array([1, 2, 3, 4]))
    v = core.band(b1, b2, c, ["id"])
    assert v["s"] == Verdict.MISSING and v["new"] == Verdict.MISSING and v["x"] == Verdict.IDENTICAL


def test_changed_type_is_changed(tmp_path):
    b1 = write(tmp_path, "b1.parquet", **base(x=pa.array([1, 2, 3, 4], pa.int32())))
    b2 = write(tmp_path, "b2.parquet", **base(x=pa.array([1, 2, 3, 4], pa.int32())))
    c = write(tmp_path, "c.parquet", **base(x=pa.array([1, 2, 3, 4], pa.int64())))
    assert core.band(b1, b2, c, ["id"])["x"] == Verdict.CHANGED


def test_different_row_set_is_a_change_in_every_column(tmp_path):
    b1 = write(tmp_path, "b1.parquet", **base())
    b2 = write(tmp_path, "b2.parquet", **base())
    c = write(tmp_path, "c.parquet", **{k: v.take([0, 1, 2]) for k, v in base().items()})
    report = core.band_report(b1, b2, c, ["id"])
    assert report.rows_changed and report.failed
    assert set(report.verdicts.values()) == {Verdict.CHANGED}


def test_baselines_with_different_columns_raise(tmp_path):
    b1 = write(tmp_path, "b1.parquet", **base())
    b2 = write(tmp_path, "b2.parquet", id=base()["id"], x=base()["x"])
    with pytest.raises(ValueError, match="baseline"):
        core.band(b1, b2, b1, ["id"])
