import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from pytest import mark

from imagematerials.concepts import KnowledgeGraph
from imagematerials.util import export_to_netcdf, import_from_netcdf, summarize_prep_data


def _get_new_compare(data, key_list):
    all_compares = []
    for name, cur_data in data.items():
        if "dims" in cur_data:
            all_compares.append([key_list + [name], cur_data])
        elif isinstance(cur_data, str):
            all_compares.append([[name], cur_data])
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
    # Key list contains the name of the dataset, i.e. ["stocks", "Region"] is
    # concerns the stocks summed over all dimensions except Region.
    key_list_copy = [k for k in key_list]
    while len(key_list_copy) > 0:
        data = data[key_list_copy[0]]
        key_list_copy.pop(0)

    if data != expected:
        assert data["name"] == expected["name"]
        assert data["dims"] == expected["dims"]
        assert data["attrs"] == expected["attrs"]
        assert data["coords"] == expected["coords"]
        assert np.allclose(data["data"], expected["data"]), (
            f"Name: {data['name']}, New: {np.sum(data['data'])}, Old: {np.sum(expected['data'])}")


@mark.parametrize("key_list,expected",
                  find_data_items(Path("tests", "data", "buildings_summary.json")))
def test_buildings_prep(bld_summary, key_list, expected):
    data = bld_summary
    # Key list contains the name of the dataset, i.e. ["stocks", "Region"] is
    # concerns the stocks summed over all dimensions except Region.
    key_list_copy = [k for k in key_list]
    while len(key_list_copy) > 0:
        data = data[key_list_copy[0]]
        key_list_copy.pop(0)

    if data != expected:
        assert data["name"] == expected["name"]
        assert data["dims"] == expected["dims"]
        assert data["attrs"] == expected["attrs"]
        assert data["coords"] == expected["coords"]
        assert  np.allclose(data["data"], expected["data"]), (
            f"Name: {data['name']}, New: {np.sum(data['data'])}, Old: {np.sum(expected['data'])}")

def _check_data_same(orig_data, new_data, name=""):
    assert type(orig_data) is type(new_data)
    if isinstance(orig_data, dict):
        for cur_name in orig_data:
            _check_data_same(orig_data[cur_name], new_data[cur_name], f"{name}{cur_name}")
        return
    if isinstance(orig_data, KnowledgeGraph):
        assert len(orig_data._items) == len(new_data._items)
        return
    if orig_data is None:
        assert orig_data == new_data, f"New data is {new_data}, where it used to be None."
        return
    if isinstance(orig_data, str):
        assert orig_data == new_data, (
            f"Wrong string value: {name}, "
            f"Old: {orig_data}, New: {new_data}."
        )
        return

    if isinstance(orig_data, xr.DataArray):
        assert orig_data.shape == new_data.shape, f"Wrong array shape: {name}"
    assert orig_data.dims == new_data.dims
    for coord in orig_data.coords:
        assert all(orig_data[coord] == new_data[coord])

@mark.parametrize("prep_data_name", ["vhc_prep_data", "bld_prep_data"])
def test_save_load_netcdf(prep_data_name, request, tmpdir):
    prep_data = request.getfixturevalue(prep_data_name)
    netcdf_fp = tmpdir / "test.netcdf"
    export_to_netcdf(prep_data, netcdf_fp)
    new_prep_data = import_from_netcdf(netcdf_fp)
    _check_data_same(prep_data, new_prep_data)
