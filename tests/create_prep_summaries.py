#!/usr/bin/env python

import json
import warnings
from pathlib import Path

import xarray as xr

from imagematerials import buildings as bld
from imagematerials import vehicles as vhc
from imagematerials.concepts import create_building_graph, create_vehicle_graph
from imagematerials.util import rebroadcast_prep_data, summarize_prep_data

# def _summarize_data(data):
#     all_summary = {}
#     for data_name, array in data.items():
#         if isinstance(array, (dict, xr.Dataset)):
#             all_summary[data_name] = _summarize_data(array)
#         elif isinstance(array, xr.DataArray):
#             all_summary[data_name] = _summarize_array(array)
#         else:
#             raise ValueError(f"Cannot compare data with name '{data_name}' with type {type(array)}")
#     return all_summary

# def _summarize_array(array):
#     all_summary = {}
#     for drop_coor in array.coords.keys():
#         sum_over = set(x for x in list(array.coords.keys())) - set({drop_coor})
#         summary = array.sum(sum_over)
#         all_summary[drop_coor] = summary.to_dict()
#     return all_summary

if __name__ == "__main__":
    # Vehicles summary
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prep_data_vhc = vhc.preprocess(Path("data", "raw"))

    knowledge_graph = create_vehicle_graph()
    new_prep_data = rebroadcast_prep_data(prep_data_vhc, knowledge_graph, dim="Type", output_coords=prep_data_vhc["shares"].coords["Type"].values)
    new_prep_data = rebroadcast_prep_data(new_prep_data, knowledge_graph, dim="Region", output_coords=prep_data_vhc["shares"].coords["Region"].values)
    new_prep_data["knowledge_graph"] = knowledge_graph
    new_prep_data["weights"] = new_prep_data.pop("vehicle_weights")

    summary_vhc = summarize_prep_data(new_prep_data)
    with open(Path("tests", "data", "vehicles_summary.json"), "w", encoding="utf8") as handle:
        json.dump(summary_vhc, handle)

    # Buildings summary
    prep_data_bld = bld.preprocess(Path("IMAGE-Mat_old_version", "IMAGE-Mat", "BUMA"))
    new_prep_data = {k: v for k, v in prep_data_bld.items()}
    new_prep_data["knowledge_graph"] = create_building_graph()

    summary_bld = summarize_prep_data(new_prep_data)
    with open(Path("tests", "data", "buildings_summary.json"), "w", encoding="utf8") as handle:
        json.dump(summary_bld, handle)
