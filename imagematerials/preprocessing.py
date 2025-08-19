import warnings
from pathlib import Path
from typing import Optional, Union

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


def _get_vehicles_prep_data(base_dir, climate_policy_scenario_dir, circular_economy_scenario_dirs):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        climate_policy_config = read_climate_policy_config(climate_policy_scenario_dir)
        circular_economy_config = read_circular_economy_config(circular_economy_scenario_dirs)
        prep_data = prep_vhc(base_dir, climate_policy_config, circular_economy_config)

    return prep_data


def _get_buildings_prep_data(base_dir, climate_policy_scenario_dir, circular_economy_scenario_dirs):
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
    region_coords = np.sort(prep_data["stocks"].coords["Region"].values.astype(int)).astype(str)
    new_prep_data = rebroadcast_prep_data(new_prep_data, knowledge_graph, dim="Region",
                                          output_coords=region_coords)

    sec_vhc = Sector("vehicles", new_prep_data)
    return sec_vhc


def get_preprocessing_data(
        sector, base_dir=None,
        climate_policy_scenario_dir: Union[str, Path, None] = None,
        circular_economy_scenario_dirs: Optional[dict[str, Union[Path, str]]] = None,
        cache: Union[bool, Path, str] = False,
        standard_scenario:str = "SSP2"):
    """Get preprocessing data with optional caching.

    Parameters
    ----------
    sector
        Sector to get the preprocessing data for.
    base_dir, optional
        Base directory in which to find the SSP2/slow scenario, by default None
    climate_policy_scenario_dir, optional
        The climate policy scenario directory, by default None
    circular_economy_scenario_dirs, optional
        The circular economy scenario directories, by default None
    cache, optional
        Where to cache the preprocessing data, by default False in which case the result won't be
        cached.
    standard_scenario, optional
        The standard scenario to use, by default "SSP2". Change if different scenario should be selected

    Returns
    -------
    sector:
        Sector object for either vehicles or buildings or another sector.

    Raises
    ------
    ValueError
        If the sector is unknown or the wrong arguments are supplied.

    """
    if climate_policy_scenario_dir is None and circular_economy_scenario_dirs is None:
        climate_policy_scenario_dir = base_dir / standard_scenario
        circular_economy_scenario_dirs = {}
    elif circular_economy_scenario_dirs is None and climate_policy_scenario_dir is not None:
        climate_policy_scenario_dir = climate_policy_scenario_dir
        circular_economy_scenario_dirs = {}
    elif climate_policy_scenario_dir is None or circular_economy_scenario_dirs is not None:
            raise ValueError("Provide both climate_policy_scenario_dir and "
                             "circular_economy_scenario_dirs  or neither as arguments.")

    if cache is False or not Path(cache).is_file():
        if sector == "vehicles":
            prep_data = _get_vehicles_prep_data(base_dir, climate_policy_scenario_dir,
                                                circular_economy_scenario_dirs)
        elif sector == "buildings":
            prep_data = _get_buildings_prep_data(base_dir, climate_policy_scenario_dir,
                                                 circular_economy_scenario_dirs)
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
