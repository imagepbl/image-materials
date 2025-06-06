import warnings
from pathlib import Path

import numpy as np

from imagematerials.buildings.preprocessing.main import buildings_preprocessing as prep_bld
from imagematerials.factory import Sector
from imagematerials.util import (
    export_to_netcdf,
    import_from_netcdf,
    read_circular_economy_config,
    read_climate_policy_config,
    rebroadcast_prep_data,
)
from imagematerials.vehicles.preprocessing import preprocess as prep_vhc


def _get_vehicles_prep_data(base_dir):
    climate_policy_scenario_dir = base_dir / 'SSP2'
    circular_economy_scenario_dirs = {"slow": base_dir / 'circular_economy_scenarios' / 'slow'}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        climate_policy_config = read_climate_policy_config(climate_policy_scenario_dir)
        circular_economy_config = read_circular_economy_config(circular_economy_scenario_dirs)
        prep_data = prep_vhc(base_dir, climate_policy_config, circular_economy_config)

    return prep_data


def _get_buildings_prep_data(base_dir):
    climate_policy_scenario_dir = base_dir / 'SSP2'
    circular_economy_scenario_dirs = {"slow": base_dir / 'circular_economy_scenarios' / 'slow'}
    climate_policy_config = read_climate_policy_config(climate_policy_scenario_dir)
    circular_economy_config = read_circular_economy_config(circular_economy_scenario_dirs)
    prep_data = prep_bld(base_dir, climate_policy_config, circular_economy_config)
    return prep_data


def _get_buildings_sector(prep_data):
    return Sector("buildings", prep_data)


def _get_vehicles_sector(prep_data):
    output_coords_type = list(prep_data["stocks"].Type.values)
    knowledge_graph = prep_data["knowledge_graph"]
    new_prep_data = rebroadcast_prep_data(prep_data, knowledge_graph, dim="Type",
                                          output_coords=output_coords_type)
    region_coords = np.sort(prep_data["stocks"].coords["Region"].values.astype(int)).astype(str)[:-2]
    new_prep_data = rebroadcast_prep_data(new_prep_data, knowledge_graph, dim="Region",
                                          output_coords=region_coords)

    sec_vhc = Sector("vehicles", new_prep_data)
    return sec_vhc


def get_preprocessing_data(sector, base_dir, cache=False, **kwargs):
    if cache is False or not Path(cache).is_file():
        if sector == "vehicles":
            prep_data = _get_vehicles_prep_data(base_dir)
        elif sector == "buildings":
            prep_data = _get_buildings_prep_data(base_dir)
        else:
            raise ValueError(f"Unknown sector {sector}")
        if cache:
            export_to_netcdf(prep_data, cache)
    else:
        prep_data = import_from_netcdf(cache)

    if sector == "vehicles":
        return _get_vehicles_sector(prep_data)
    elif sector == "buildings":
        return _get_buildings_sector(prep_data)
    raise ValueError(f"Unknown sector {sector}")
