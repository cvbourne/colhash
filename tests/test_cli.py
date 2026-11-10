import json

import pyarrow as pa
import pyarrow.parquet as pq

from colhash.cli import main


def write(tmp_path, name, **cols):
    path = str(tmp_path / name)
    pq.write_table(pa.table(cols), path)
    return path


def cols(x=(1.0, 2.0, 3.0)):
    return {"id": pa.array([1, 2, 3]), "x": pa.array(list(x)), "s": pa.array(["a", "b", "c"])}


def test_hash_json(tmp_path, capsys):
    a = write(tmp_path, "a.parquet", **cols())
    assert main(["hash", a, "--key", "id", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out[0]["rows"] == 3 and set(out[0]["columns"]) == {"id", "x", "s"}


def test_diff_exit_codes(tmp_path, capsys):
    a = write(tmp_path, "a.parquet", **cols())
    same = write(tmp_path, "same.parquet", **cols())
    other = write(tmp_path, "other.parquet", **cols(x=(1.0, 2.5, 3.0)))
    assert main(["diff", a, same, "--key", "id"]) == 0
    assert main(["diff", a, other, "--key", "id"]) == 1


def test_diff_json_shape(tmp_path, capsys):
    a = write(tmp_path, "a.parquet", **cols())
    other = write(tmp_path, "other.parquet", **cols(x=(1.0, 2.5, 3.0)))
    assert main(["diff", a, other, "--key", "id", "--json"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["identical"] is False
    x = next(c for c in out["columns"] if c["column"] == "x")
    assert x["n_diff"] == 1 and x["max_abs"] == 0.5


def test_band_pass_and_fail(tmp_path, capsys):
    b1 = write(tmp_path, "b1.parquet", **cols())
    b2 = write(tmp_path, "b2.parquet", **cols(x=(1.0, 2.0, 3.0 + 1e-6)))
    ok = write(tmp_path, "ok.parquet", **cols(x=(1.0 + 1e-7, 2.0, 3.0)))
    bad = write(tmp_path, "bad.parquet", **cols(x=(1.0, 2.5, 3.0)))
    assert main(["band", b1, b2, ok, "--key", "id", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["failed"] is False
    assert main(["band", b1, b2, bad, "--key", "id"]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_band_markdown(tmp_path, capsys):
    b1 = write(tmp_path, "b1.parquet", **cols())
    b2 = write(tmp_path, "b2.parquet", **cols())
    bad = write(tmp_path, "bad.parquet", **cols(x=(1.0, 2.5, 3.0)))
    assert main(["band", b1, b2, bad, "--key", "id", "--markdown"]) == 1
    out = capsys.readouterr().out
    assert out.startswith("### colhash band report") and "**changed**" in out and "Result: FAIL" in out


def test_errors_exit_2_with_a_message(tmp_path, capsys):
    a = write(tmp_path, "a.parquet", **cols())
    assert main(["diff", a, a, "--key", "nope"]) == 2
    assert "key column" in capsys.readouterr().err
    assert main(["diff", a, str(tmp_path / "missing.parquet"), "--key", "id"]) == 2


def test_json_and_markdown_are_mutually_exclusive(tmp_path):
    import pytest
    a = write(tmp_path, "a.parquet", **cols())
    with pytest.raises(SystemExit):
        main(["diff", a, a, "--key", "id", "--json", "--markdown"])


def test_a_file_that_is_not_parquet_exits_2(tmp_path, capsys):
    bad = tmp_path / "notes.parquet"
    bad.write_text("this is not parquet")
    assert main(["hash", str(bad)]) == 2
    assert capsys.readouterr().err.startswith("colhash: error:")


def test_key_types_that_cannot_be_joined_exit_2(tmp_path, capsys):
    a = write(tmp_path, "a.parquet", id=pa.array([1, 2, 3], pa.int64()), x=pa.array([1.0, 2.0, 3.0]))
    b = write(tmp_path, "b.parquet", id=pa.array(["1", "2", "3"]), x=pa.array([1.0, 2.0, 3.0]))
    assert main(["diff", a, b, "--key", "id"]) == 2
    assert "colhash: error:" in capsys.readouterr().err
