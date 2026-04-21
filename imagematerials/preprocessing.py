import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np

from imagematerials.buildings.preprocessing.main import buildings_preprocessing as prep_bld
from imagematerials.rest_of.preprocessing.main import rest_of_preprocessing as prep_rest
from imagematerials.vehicles.preprocessing.main import vehicles_preprocessing as prep_vhc
from imagematerials.eol.preprocessing import eol_preprocessing as prep_eol
from imagematerials.electricity.preprocessing import get_preprocessing_data_gen as prep_elc_gen
from imagematerials.electricity.preprocessing import get_preprocessing_data_grid as prep_elc_grid
from imagematerials.electricity.preprocessing import get_preprocessing_data_stor as prep_elc_stor

from imagematerials.fossil_fuels.preprocessing.main import get_preprocessing_data_extraction as prep_ff_extr
from imagematerials.fossil_fuels.preprocessing.main import get_preprocessing_data_processing as prep_ff_proc
from imagematerials.fossil_fuels.preprocessing.main import get_preprocessing_data_pipelines as prep_ff_pipe
from imagematerials.fossil_fuels.preprocessing.main import get_preprocessing_data_transport as prep_ff_trp

from imagematerials.factory import Sector
from imagematerials.util import (
    export_to_netcdf,
    import_from_netcdf,
    read_circular_economy_config,
    read_climate_policy_config,
    rebroadcast_prep_data,
)
from imagematerials.constants import IMAGE_REGIONS


def _get_vehicles_prep_data(base_dir, climate_policy_scenario_dir, circular_economy_scenario_dirs):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        climate_policy_config = read_climate_policy_config(climate_policy_scenario_dir)
        circular_economy_config = read_circular_economy_config(circular_economy_scenario_dirs)
        prep_data = prep_vhc(base_dir, climate_policy_config, circular_economy_config)

    return prep_data

def _get_fossilfuels_prep_data(base_dir, climate_policy_scenario_dir, circular_economy_scenario_dirs, scenario, year_start, year_end, year_out):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        climate_policy_config = read_climate_policy_config(climate_policy_scenario_dir)
        circular_economy_config = read_circular_economy_config(circular_economy_scenario_dirs)
        prep_data_extraction = prep_ff_extr(base_dir, climate_policy_config, circular_economy_config, scenario, year_start, year_end, year_out)
        prep_data_processing = prep_ff_proc(base_dir, climate_policy_config, circular_economy_config, scenario, year_start, year_end, year_out)
        prep_data_pipelines = prep_ff_pipe(base_dir, climate_policy_config, circular_economy_config, scenario, year_start, year_end, year_out)
        prep_data_transport = prep_ff_trp(base_dir, climate_policy_config, circular_economy_config, scenario, year_start, year_end, year_out)
            
        prep_data = {
            "prep_data_extraction": prep_data_extraction,
            "prep_data_processing": prep_data_processing,
            "prep_data_pipelines": prep_data_pipelines,
            "prep_data_transport": prep_data_transport,

        }
    return prep_data

def _get_electricity_prep_data(base_dir, climate_policy_scenario_dir, circular_economy_scenario_dirs, scenario, year_start, year_end, year_out):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        climate_policy_config = read_climate_policy_config(climate_policy_scenario_dir)
        circular_economy_config = read_circular_economy_config(circular_economy_scenario_dirs)
        prep_data_gen = prep_elc_gen(base_dir, climate_policy_config, circular_economy_config, scenario, year_start, year_end, year_out)
        prep_data_grid_lines, prep_data_grid_add = prep_elc_grid(base_dir, climate_policy_config, circular_economy_config, scenario, year_start, year_end, year_out)
        prep_data_stor_phs, prep_data_stor_other = prep_elc_stor(base_dir, climate_policy_config, circular_economy_config, scenario, year_start, year_end, year_out)
        

        prep_data = {
            "prep_data_gen": prep_data_gen,
            "prep_data_grid_lines": prep_data_grid_lines,
            "prep_data_grid_add": prep_data_grid_add,
            "prep_data_stor_phs": prep_data_stor_phs,
            "prep_data_stor_other": prep_data_stor_other
        }
    return prep_data

def _get_buildings_prep_data(base_dir, climate_policy_scenario_dir, circular_economy_scenario_dirs):
    climate_policy_config = read_climate_policy_config(climate_policy_scenario_dir)
    circular_economy_config = read_circular_economy_config(circular_economy_scenario_dirs)
    prep_data = prep_bld(base_dir, climate_policy_config, circular_economy_config)
    return prep_data


def _get_rest_prep_data(base_dir, climate_policy_scenario_dir, scenario_name):
    prep_data = prep_rest(base_dir, climate_policy_scenario_dir, scenario_name)
    sector_rest = Sector("rest_of", prep_data)
    return sector_rest

def _get_end_of_life_prep_data(base_dir, circular_economy_scenario_dirs):
    prep_data = prep_eol(base_dir, circular_economy_scenario_dirs)
    return prep_data

def _get_vehicles_sector(prep_data):
    output_coords_type = list(prep_data["stocks"].Type.values)
    knowledge_graph = prep_data["knowledge_graph"]
    new_prep_data = rebroadcast_prep_data(prep_data, knowledge_graph, dim="Type",
                                          output_coords=output_coords_type)
    # region_coords = np.sort(prep_data["stocks"].coords["Region"].values.astype(int)).astype(str)
    region_coords = IMAGE_REGIONS
    new_prep_data = rebroadcast_prep_data(new_prep_data, knowledge_graph, dim="Region",
                                          output_coords=region_coords)

    sec_vhc = Sector("vehicles", new_prep_data)
    return sec_vhc

def _get_fossilfuels_sector(prep_data):

    sec_ff_extr = Sector("ff_extr", prep_data["prep_data_extraction"])
    sec_ff_proc = Sector("ff_proc", prep_data["prep_data_processing"])
    sec_ff_pipe = Sector("ff_pipe", prep_data["prep_data_pipelines"])
    sec_ff_trp = Sector("ff_trp", prep_data["prep_data_transport"])


    sec_ff = [sec_ff_extr, sec_ff_proc, sec_ff_pipe, sec_ff_trp]

    return sec_ff

def _get_electricity_sector(prep_data):

    sec_elc_gen = Sector("elc_gen", prep_data["prep_data_gen"])
    sec_elc_grid_lines = Sector("elc_grid_lines", prep_data["prep_data_grid_lines"])
    sec_elc_grid_add = Sector("elc_grid_add", prep_data["prep_data_grid_add"])
    sec_elc_stor_phs = Sector("elc_stor_phs", prep_data["prep_data_stor_phs"])
    sec_elc_stor_other = Sector("elc_stor_other", prep_data["prep_data_stor_other"])

    sec_elc = [sec_elc_gen, sec_elc_grid_lines, sec_elc_grid_add, sec_elc_stor_phs, sec_elc_stor_other]

    return sec_elc

def _get_buildings_sector(prep_data):
    return Sector("buildings", prep_data)

def _get_end_of_life_sector(prep_data):
    return Sector("eol", prep_data)

def get_preprocessing_data(
        sector, base_dir=None,
        climate_policy_scenario_dir: Union[str, Path, None] = None,
        circular_economy_scenario_dirs: Optional[dict[str, Union[Path, str]]] = None,
        scenario_name=None,
        cache: Union[bool, Path, str] = False,
        standard_scenario: str = "SSP2",
        year_start: int = 1971,
        year_end: int = 2100,
        year_out: int = 2100):
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
    scenario_name, optional
        The scenario name for the rest_of sector, by default None
    cache, optional
        Where to cache the preprocessing data, by default False in which case the result won't be
        cached.
    standard_scenario, optional
        The standard scenario to use, by default "SSP2". Change if different scenario should be selected

    year_start: int, optional
        The first year of the simulation timeline (typically the beginning of the
        historical dataset). Default is 1970.

    year_end: int, optional
        The final year of the simulation horizon. Defines the last year for which
        model results will be computed. Default is 2100.

    year_out: int, optional
        The target year of interest for reporting or extracting results. This is
        often a milestone year (e.g., 2050 or 2100) used in scenario analysis.
        Must fall between `year_start` and `year_end`. Default is 2100.


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
    elif climate_policy_scenario_dir is not None and circular_economy_scenario_dirs is not None:
        pass
    elif climate_policy_scenario_dir is None and circular_economy_scenario_dirs is not None:
        raise ValueError("if circular_economy_scenario_dirs is set, climate_policy_scenario_dir has to be set too")

    if cache is False or not Path(cache).is_file():
        if sector == "vehicles":
            prep_data = _get_vehicles_prep_data(base_dir, climate_policy_scenario_dir,
                                                circular_economy_scenario_dirs)
        elif sector == "buildings":
            prep_data = _get_buildings_prep_data(base_dir, climate_policy_scenario_dir,
                                                 circular_economy_scenario_dirs)
        elif sector == "electricity":
            prep_data = _get_electricity_prep_data(base_dir, climate_policy_scenario_dir,
                                                   circular_economy_scenario_dirs,
                                                   standard_scenario,
                                                   year_start,
                                                   year_end,
                                                   year_out)
        elif sector == "fossil_fuels":
            prep_data = _get_fossilfuels_prep_data(base_dir, climate_policy_scenario_dir,
                                                   circular_economy_scenario_dirs,
                                                   standard_scenario,
                                                   year_start,
                                                   year_end,
                                                   year_out)
        elif sector == "rest_of":
            prep_data = _get_rest_prep_data(base_dir, climate_policy_scenario_dir,
                                            scenario_name)

        elif sector == "eol": 
            prep_data = _get_end_of_life_prep_data(base_dir,circular_economy_scenario_dirs)

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
    elif sector == "electricity":
        return _get_electricity_sector(prep_data)
    elif sector == "fossil_fuels":
        return _get_fossilfuels_sector(prep_data)
    elif sector == "rest_of":
        return _get_rest_prep_data(base_dir, climate_policy_scenario_dir, scenario_name)
    elif sector == "eol":
        return _get_end_of_life_sector(prep_data)
    raise ValueError(f"Unknown sector {sector}")
