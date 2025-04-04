#!/usr/bin/env python

import json
import warnings
from pathlib import Path

import xarray as xr

from imagematerials import buildings as bld
from imagematerials import vehicles as vhc


def _summarize_data(data):
    all_summary = {}
    for data_name, array in data.items():
        if isinstance(array, (dict, xr.Dataset)):
            all_summary[data_name] = _summarize_data(array)
        elif isinstance(array, xr.DataArray):
            all_summary[data_name] = _summarize_array(array)
        else:
            raise ValueError(f"Cannot compare data with name '{data_name}' with type {type(array)}")
    return all_summary

def _summarize_array(array):
    all_summary = {}
    for drop_coor in array.coords.keys():
        sum_over = set(x for x in list(array.coords.keys())) - set({drop_coor})
        summary = array.sum(sum_over)
        all_summary[drop_coor] = summary.to_dict()
    return all_summary

if __name__ == "__main__":
    # Vehicles summary
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prep_data_vhc = vhc.preprocess(Path("data", "raw"))
    summary_vhc = _summarize_data(prep_data_vhc)
    with open(Path("tests", "data", "vehicles_summary.json"), "w", encoding="utf8") as handle:
        json.dump(summary_vhc, handle)

    # Buildings summary
    prep_data_bld = bld.preprocess(Path("IMAGE-Mat_old_version", "IMAGE-Mat", "BUMA"))
    summary_bld = _summarize_data(prep_data_bld)
    with open(Path("tests", "data", "buildings_summary.json"), "w", encoding="utf8") as handle:
        json.dump(summary_bld, handle)
