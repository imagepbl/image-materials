import json
from pathlib import Path
from pytest import mark
import pytest
from imagematerials.util import summarize_prep_data

def _get_new_compare(data, key_list):
    all_compares = []
    for name, cur_data in data.items():
        if "dims" in cur_data:
            all_compares.append([key_list + [name], cur_data])
        else:
            all_compares.extend(_get_new_compare(cur_data, key_list + [name]))
    return all_compares


def find_data_items(fp):
    with open(fp, "r", encoding="utf8") as handle:
        summary = json.load(handle)
    return _get_new_compare(summary, [])

@pytest.fixture(scope="module")
def vhc_summary(vhc_prep_data):
    return summarize_prep_data(vhc_prep_data)

@pytest.fixture(scope="module")
def bld_summary(bld_prep_data):
    return summarize_prep_data(bld_prep_data)


@mark.parametrize("key_list,expected",
                  find_data_items(Path("tests", "data", "vehicles_summary.json")))
def test_vehicles_prep(vhc_summary, key_list, expected):
    data = vhc_summary
    while len(key_list) > 0:
        data = data[key_list[0]]
        key_list.pop(0)
    if data != expected:
        assert data["name"] == expected["name"]
        assert data["dims"] == expected["dims"]
        assert data["attrs"] == expected["attrs"]
        assert data["coords"] == expected["coords"]
        assert data["data"] == expected["data"]


@mark.parametrize("key_list,expected",
                  find_data_items(Path("tests", "data", "buildings_summary.json")))
def test_buildings_prep(bld_summary, key_list, expected):
    data = bld_summary
    while len(key_list) > 0:
        data = data[key_list[0]]
        key_list.pop(0)
    if data != expected:
        assert data["name"] == expected["name"]
        assert data["dims"] == expected["dims"]
        assert data["attrs"] == expected["attrs"]
        assert data["coords"] == expected["coords"]
        assert data["data"] == expected["data"]
