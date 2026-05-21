"""Tests for :mod:`pandera.io._check_io` helpers."""

from pandera.io._check_io import checks_dict_to_list


def test_checks_dict_to_list_passes_through_none():
    assert checks_dict_to_list(None) is None


def test_checks_dict_to_list_passes_through_list():
    payload = [{"options": {"check_name": "gt"}, "value": 1}]
    assert checks_dict_to_list(payload) is payload


def test_checks_dict_to_list_normalizes_mapping():
    out = checks_dict_to_list({"greater_than": {"value": 0}})
    assert len(out) == 1
    assert out[0]["options"]["check_name"] == "greater_than"
    assert out[0]["value"] == 0
