import json
from pathlib import Path
from pytest import mark
import pytest
from imagematerials.util import summarize_prep_data, import_from_netcdf, export_to_netcdf

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

def _check_data_same(orig_data, new_data, name=""):
    if isinstance(orig_data, dict):
        for cur_name in orig_data:
            _check_data_same(orig_data[cur_name], new_data[cur_name], f"{name}{cur_name}")
        return
    assert orig_data.shape == new_data.shape, f"Wrong array shape: {name}"
    assert orig_data.dims == new_data.dims
    for coord in orig_data.coords:
        assert all(orig_data[coord] == new_data[coord])

@mark.parametrize("prep_data_name", ["vhc_prep_data"])#, "bld_prep_data"])
def test_save_load_netcdf(prep_data_name, request, tmpdir):
    prep_data = request.getfixturevalue(prep_data_name)
    netcdf_fp = tmpdir / "test.netcdf"
    export_to_netcdf(prep_data, netcdf_fp)
    new_prep_data = import_from_netcdf(netcdf_fp)
    _check_data_same(prep_data, new_prep_data)