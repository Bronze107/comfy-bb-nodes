import importlib.util
import os
from pathlib import Path

import pytest
from folder_paths import base_path

_IMPL = Path(__file__).resolve().parent.parent / "__init__.py"
_spec = importlib.util.spec_from_file_location("comfy_bb_nodes", _IMPL)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)

load_wildcard_lines = _impl.load_wildcard_lines
pick_random_line = _impl.pick_random_line
resolve_path = _impl.resolve_path


def test_resolve_path_absolute_unchanged():
    assert resolve_path(r"E:\data\card.txt") == r"E:\data\card.txt"


def test_resolve_path_relative_falls_back_to_base():
    assert resolve_path("card.txt") == os.path.join(base_path, "card.txt")


def test_resolve_path_relative_prefers_input(tmp_path, monkeypatch):
    import folder_paths
    monkeypatch.setattr(folder_paths, "input_directory", str(tmp_path))
    (tmp_path / "card.txt").write_text("red\n", encoding="utf-8")
    assert resolve_path("card.txt") == os.path.join(str(tmp_path), "card.txt")


def test_load_wildcard_lines_reads_lines(tmp_path):
    f = tmp_path / "list.txt"
    f.write_text("red\nblue\ngreen\n", encoding="utf-8")
    assert load_wildcard_lines(str(f)) == ["red", "blue", "green"]


def test_load_wildcard_lines_skips_blank_and_comment_lines(tmp_path):
    f = tmp_path / "list.txt"
    f.write_text("red\n\n  # comment\n  blue  \n", encoding="utf-8")
    assert load_wildcard_lines(str(f)) == ["red", "blue"]


def test_load_wildcard_lines_strips_whitespace(tmp_path):
    f = tmp_path / "list.txt"
    f.write_text("  red  \n\tblue\n", encoding="utf-8")
    assert load_wildcard_lines(str(f)) == ["red", "blue"]


def test_load_wildcard_lines_handles_bom(tmp_path):
    f = tmp_path / "list.txt"
    f.write_bytes(b"\xef\xbb\xbfred\nblue\n")
    assert load_wildcard_lines(str(f)) == ["red", "blue"]


def test_load_wildcard_lines_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_wildcard_lines(str(tmp_path / "nope.txt"))


def test_load_wildcard_lines_only_comments_raises(tmp_path):
    f = tmp_path / "list.txt"
    f.write_text("# only comments\n\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_wildcard_lines(str(f))


def test_pick_random_line_reproducible_with_seed():
    lines = ["red", "blue", "green"]
    assert pick_random_line(lines, 42) == pick_random_line(lines, 42)


def test_pick_random_line_returns_member_of_list():
    lines = ["red", "blue", "green"]
    assert pick_random_line(lines, 12345) in lines


def test_pick_random_line_different_seeds_vary():
    lines = ["red", "blue", "green"]
    picked = {pick_random_line(lines, s) for s in range(100)}
    assert len(picked) > 1


def test_pick_random_line_empty_list_raises():
    with pytest.raises(ValueError):
        pick_random_line([], 0)
