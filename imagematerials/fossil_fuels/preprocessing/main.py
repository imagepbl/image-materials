#%% Import general constants and modules 
from pathlib import Path
from typing import Any

import prism

from imagematerials.util import convert_lifetime
from imagematerials.concepts import create_fossil_fuel_graph
#Import xarrays from lifetimes, materials and capacity folders
from imagematerials.fossil_fuels.preprocessing.inputs import load_inputs_image, load_inputs
from imagematerials.fossil_fuels.preprocessing.drivers import compute_all_infra
from imagematerials.fossil_fuels.preprocessing.lifetimes import (
    compute_extraction_lifetimes,
    compute_processing_lifetimes,
    compute_transport_lifetimes,
    compute_pipelines_lifetimes,
    compute_storage_lifetimes
)
from imagematerials.fossil_fuels.preprocessing.materials import (
    compute_extraction_materials,
    compute_processing_materials,
    compute_transport_materials,
    compute_pipelines_materials,
    compute_storage_materials
)
from imagematerials.fossil_fuels.preprocessing.capacity import (
    compute_extraction_capacity,
    compute_processing_capacity,
    compute_transport_capacity,
    compute_pipelines_capacity,
    compute_storage_capacity
)
from imagematerials.fossil_fuels.constants import START_YEAR, END_YEAR

# The input data and the infrastructure stocks (coal, oil, gas) can only be computed once we know
# path_base and climate_policy_config, which are given to the get_preprocessing_data_* functions below.
# get_inputs_and_infra() does this when it is called for the first time and remembers the result,
# so the other stages do not repeat the work.
_inputs_and_infra = {}

def get_inputs_and_infra(
        path_base: str | Path,
        climate_policy_config: dict,
    ):
    """Return (inputs, infra) for this scenario; load and compute them only the first time.

    Parameters
    ----------
    path_base: Folder that contains the "fossil_fuels" data folder (the same path_base that is 
        passed to the get_preprocessing_data_* functions).
    climate_policy_config: Scenario configuration.

    Returns
    -------
    inputs: all input data, read by load_inputs() in inputs.py
    infra:  infrastructure stocks, {"coal": {...}, "gas": {...}, "oil": {...}}; the respective
            functions below pick the right ones (e.g. infra["oil"]["storage"])

    """
    # a new scenario (different config folder) needs new data, so it is part of the key
    key = (str(path_base), str(climate_policy_config["config_file_path"]))
    if key not in _inputs_and_infra:
        inputs_image = load_inputs_image(climate_policy_config)
        inputs = load_inputs(path_base, climate_policy_config)
        infra = compute_all_infra(inputs_image, inputs)
        _inputs_and_infra[key] = (inputs_image, inputs, infra)
    return _inputs_and_infra[key]


####################################################################################################
#%%Extraction stage
####################################################################################################
def get_preprocessing_data_extraction(
        path_base: str | Path,
        climate_policy_config: dict[str, Any],
        circular_economy_config: dict[str, Any],
        resource_efficiency_flags: dict = None,
    ) -> dict[str, Any]:
    """Collect the preprocessing data of the extraction stage in the format the model expects.

    Extraction covers coal mines (open cast, underground) and oil and gas extraction
    (onshore, offshore). The function builds the lifetimes, material intensities and
    stocks of this stage.

    The input data and the infrastructure stocks are loaded and calculated only the
    first time any stage asks for them; the other stages reuse them (see get_inputs_and_infra()).

    Parameters
    ----------
    path_base: Folder that contains the "fossil_fuels" data folder.
    climate_policy_config: Scenario configuration. Gives the scenario name and the
        location of the IMAGE output files (config_file_path and data_files).
    circular_economy_config: Currently not used for fossil fuels. It is part of the
        signature because the calling script passes it to all sectors.

    Returns
    -------
    Dictionary with the keys:
        - "lifetimes": lifetimes in years, one DataArray per distribution parameter (format created 
            by convert_lifetime()).
        - "stocks": extraction infrastructure in kg, dimensions (time, Region, Type).
        - "material_intensities": material use per unit of stock in kg/kg, dimensions (Cohort, Type, 
            material).
        - "knowledge_graph": the fossil fuel knowledge graph (create_fossil_fuel_graph()).
        - "set_unit_flexible": the unit of the stocks (from prism.U_). The model
            uses it as a flexible placeholder unit at the start, before it knows the
            data (see model.py).

    """
    # read in input data (IMAGE data and technology specific data (e.g. lifetimes))
    year_start = START_YEAR
    year_end = END_YEAR

    inputs_image, inputs, infra = get_inputs_and_infra(path_base, climate_policy_config)
    
    # The lifetimes are converted to the proper format for the model (dictionary with keys:distribution name, values:datarrays containing distribution parameters)
    extraction_lifetime_xr = compute_extraction_lifetimes(inputs["lifetimes"]["extraction"])
    extraction_lifetime_xr = convert_lifetime(extraction_lifetime_xr)

    extraction_materials_xr = compute_extraction_materials(inputs["materials"]["extraction"])
    extractioncap_xr = compute_extraction_capacity(infra, year_start, year_end)

    # bring preprocessing data into a generic format for the model
    prep_data_extraction = {}
    prep_data_extraction["lifetimes"] = extraction_lifetime_xr
    prep_data_extraction["stocks"] = extractioncap_xr
    prep_data_extraction["material_intensities"] = extraction_materials_xr
    prep_data_extraction["knowledge_graph"] = create_fossil_fuel_graph() 
    prep_data_extraction["set_unit_flexible"] = prism.U_(prep_data_extraction["stocks"]) # prism.U_ gives the unit back

    return prep_data_extraction


####################################################################################################
#%%Processing stage (coal preparation and oil refinery)
####################################################################################################
def get_preprocessing_data_processing(
        path_base: str | Path,
        climate_policy_config: dict[str, Any],
        circular_economy_config: dict[str, Any],
        resource_efficiency_flags: dict = None,
    ) -> dict[str, Any]:
    """Collect the preprocessing data of the processing stage in the format the model expects.

    Extraction covers coal preparation, gas processing, oil refinery. 
    The function builds the lifetimes, material intensities and stocks of this stage.

    The input data and the infrastructure stocks are loaded and calculated only the
    first time any stage asks for them; the other stages reuse them (see get_inputs_and_infra()).

    Parameters
    ----------
    path_base: Folder that contains the "fossil_fuels" data folder.
    climate_policy_config: Scenario configuration. Gives the scenario name and the
        location of the IMAGE output files (config_file_path and data_files).
    circular_economy_config: Currently not used for fossil fuels. It is part of the
        signature because the calling script passes it to all sectors.

    Returns
    -------
    Dictionary with the keys:
        - "lifetimes": lifetimes in years, one DataArray per distribution parameter (format created 
            by convert_lifetime()).
        - "stocks": processing infrastructure in kg, dimensions (time, Region, Type).
        - "material_intensities": material use per unit of stock in kg/kg, dimensions (Cohort, Type, 
            material).
        - "knowledge_graph": the fossil fuel knowledge graph (create_fossil_fuel_graph()).
        - "set_unit_flexible": the unit of the stocks (from prism.U_). The model
            uses it as a flexible placeholder unit at the start, before it knows the
            data (see model.py).

    """
    year_start = START_YEAR
    year_end = END_YEAR

    # read in input data (IMAGE data and technology specific data (e.g. lifetimes))
    inputs_image, inputs, infra = get_inputs_and_infra(path_base, climate_policy_config)

    # The lifetimes are converted to the proper format for the model (dictionary with keys:distribution name, values:datarrays containing distribution parameters)
    processing_lifetime_xr = compute_processing_lifetimes(inputs["lifetimes"]["processing"])
    processing_lifetime_xr = convert_lifetime(processing_lifetime_xr)

    processing_materials_xr = compute_processing_materials(inputs["materials"]["processing"])
    processingcap_xr = compute_processing_capacity(infra, year_start, year_end)
    
    # bring preprocessing data into a generic format for the model
    prep_data_processing = {}
    prep_data_processing["lifetimes"] = processing_lifetime_xr
    prep_data_processing["stocks"] = processingcap_xr
    prep_data_processing["material_intensities"] = processing_materials_xr
    prep_data_processing["knowledge_graph"] = create_fossil_fuel_graph() 
    prep_data_processing["set_unit_flexible"] = prism.U_(prep_data_processing["stocks"]) # prism.U_ gives the unit back

    return prep_data_processing


####################################################################################################
#%%Processing stage (gas processing and oil storage)
####################################################################################################
def get_preprocessing_data_storage(
        path_base: str | Path,
        climate_policy_config: dict[str, Any],
        circular_economy_config: dict[str, Any],
        resource_efficiency_flags: dict = None,
    ) -> dict[str, Any]:
    """Collect the preprocessing data of the storage stage in the format the model expects.

    Extraction covers oil storage (crude and products added up). 
    The function builds the lifetimes, material intensities and stocks of this stage.

    The input data and the infrastructure stocks are loaded and calculated only the
    first time any stage asks for them; the other stages reuse them (see get_inputs_and_infra()).

    Parameters
    ----------
    path_base: Folder that contains the "fossil_fuels" data folder.
    climate_policy_config: Scenario configuration. Gives the scenario name and the
        location of the IMAGE output files (config_file_path and data_files).
    circular_economy_config: Currently not used for fossil fuels. It is part of the
        signature because the calling script passes it to all sectors.

    Returns
    -------
    Dictionary with the keys:
        - "lifetimes": lifetimes in years, one DataArray per distribution parameter (format created 
            by convert_lifetime()).
        - "stocks": storage infrastructure in m3, dimensions (time, Region, Type).
        - "material_intensities": material use per unit of stock in kg/m3, dimensions (Cohort, Type, 
            material).
        - "knowledge_graph": the fossil fuel knowledge graph (create_fossil_fuel_graph()).
        - "set_unit_flexible": the unit of the stocks (from prism.U_). The model
            uses it as a flexible placeholder unit at the start, before it knows the
            data (see model.py).

    """
    year_start = START_YEAR
    year_end = END_YEAR

    # read in input data (IMAGE data and technology specific data (e.g. lifetimes))
    inputs_image, inputs, infra = get_inputs_and_infra(path_base, climate_policy_config)

    # The lifetimes are converted to the proper format for the model (dictionary with keys:distribution name, values:datarrays containing distribution parameters)
    storage_lifetime_xr = compute_storage_lifetimes(inputs["lifetimes"]["storage"])
    storage_lifetime_xr = convert_lifetime(storage_lifetime_xr)

    storage_materials_xr = compute_storage_materials(inputs["materials"]["storage"])
    storagecap_xr = compute_storage_capacity(infra, year_start, year_end)
    # bring preprocessing data into a generic format for the model
    prep_data_storage = {}
    prep_data_storage["lifetimes"] = storage_lifetime_xr
    prep_data_storage["stocks"] = storagecap_xr
    prep_data_storage["material_intensities"] = storage_materials_xr
    prep_data_storage["knowledge_graph"] = create_fossil_fuel_graph() 
    prep_data_storage["set_unit_flexible"] = prism.U_(prep_data_storage["stocks"]) # prism.U_ gives the unit back

    return prep_data_storage


####################################################################################################
#%% Transport stage
####################################################################################################
def get_preprocessing_data_transport(
        path_base: str | Path,
        climate_policy_config: dict[str, Any],
        circular_economy_config: dict[str, Any],
        resource_efficiency_flags: dict = None,
    ) -> dict[str, Any]:
    """Collect the preprocessing data of the transport stage in the format the model expects.

    Extraction covers freight transport vehicles (ships, rail, trucks) for coal, oil and gas. 
    The function builds the lifetimes, material intensities and stocks of this stage.

    The input data and the infrastructure stocks are loaded and calculated only the
    first time any stage asks for them; the other stages reuse them (see get_inputs_and_infra()).

    Parameters
    ----------
    path_base: Folder that contains the "fossil_fuels" data folder.
    climate_policy_config: Scenario configuration. Gives the scenario name and the
        location of the IMAGE output files (config_file_path and data_files).
    circular_economy_config: Currently not used for fossil fuels. It is part of the
        signature because the calling script passes it to all sectors.

    Returns
    -------
    Dictionary with the keys:
        - "lifetimes": lifetimes in years, one DataArray per distribution parameter (format created 
            by convert_lifetime()).
        - "stocks": transport infrastructure in kg, dimensions (time, Region, Type).
        - "material_intensities": material use per unit of stock in kg/kg, dimensions (Cohort, Type, 
            material).
        - "knowledge_graph": the fossil fuel knowledge graph (create_fossil_fuel_graph()).
        - "set_unit_flexible": the unit of the stocks (from prism.U_). The model
            uses it as a flexible placeholder unit at the start, before it knows the
            data (see model.py).

    """
    year_start = START_YEAR
    year_end = END_YEAR

    # read in input data (IMAGE data and technology specific data (e.g. lifetimes))
    inputs_image, inputs, infra = get_inputs_and_infra(path_base, climate_policy_config)

    transport_lifetime_xr = compute_transport_lifetimes(inputs["lifetimes"]["transport"])
    transport_lifetime_xr = convert_lifetime(transport_lifetime_xr)
    transport_materials_xr = compute_transport_materials(inputs["materials"]["transport"])
    transportcap_xr = compute_transport_capacity(infra, year_start, year_end)

    #  # bring preprocessing data into a generic format for the model
    prep_data_transport = {}
    prep_data_transport["lifetimes"] = transport_lifetime_xr
    prep_data_transport["stocks"] = transportcap_xr
    prep_data_transport["material_intensities"] = transport_materials_xr
    prep_data_transport["knowledge_graph"] = create_fossil_fuel_graph() 
    prep_data_transport["set_unit_flexible"] = prism.U_(prep_data_transport["stocks"]) # prism.U_ gives the unit back

    return prep_data_transport


####################################################################################################
#%% Pipelines stage (oil, gas)
####################################################################################################
def get_preprocessing_data_pipelines(
        path_base: str | Path,
        climate_policy_config: dict[str, Any],
        circular_economy_config: dict[str, Any],
        resource_efficiency_flags: dict = None,
    ) -> dict[str, Any]:
    """Collect the preprocessing data of the pipeline stage in the format the model expects.

    Extraction covers gas and oil pipelines. 
    The function builds the lifetimes, material intensities and stocks of this stage.

    The input data and the infrastructure stocks are loaded and calculated only the
    first time any stage asks for them; the other stages reuse them (see get_inputs_and_infra()).

    Parameters
    ----------
    path_base: Folder that contains the "fossil_fuels" data folder.
    climate_policy_config: Scenario configuration. Gives the scenario name and the
        location of the IMAGE output files (config_file_path and data_files).
    circular_economy_config: Currently not used for fossil fuels. It is part of the
        signature because the calling script passes it to all sectors.

    Returns
    -------
    Dictionary with the keys:
        - "lifetimes": lifetimes in years, one DataArray per distribution parameter (format created 
            by convert_lifetime()).
        - "stocks": pipeline infrastructure in km, dimensions (time, Region, Type).
        - "material_intensities": material use per unit of stock in kg/km, dimensions (Cohort, Type, 
            material).
        - "knowledge_graph": the fossil fuel knowledge graph (create_fossil_fuel_graph()).
        - "set_unit_flexible": the unit of the stocks (from prism.U_). The model
            uses it as a flexible placeholder unit at the start, before it knows the
            data (see model.py).

    """
    year_start = START_YEAR
    year_end = END_YEAR

    # read in input data (IMAGE data and technology specific data (e.g. lifetimes))
    inputs_image, inputs, infra = get_inputs_and_infra(path_base, climate_policy_config)

    pipelines_lifetime_xr = compute_pipelines_lifetimes(inputs["lifetimes"]["pipelines"]) 
    pipelines_lifetime_xr = convert_lifetime(pipelines_lifetime_xr)
    
    pipelines_materials_xr = compute_pipelines_materials(inputs["materials"]["pipelines"])
    pipelinecap_xr = compute_pipelines_capacity(infra, inputs["drivers"]["on_offshore_oil_pipeline"], year_start, year_end)

    #  bring preprocessing data into a generic format for the model
    prep_data_pipelines = {}
    prep_data_pipelines["lifetimes"] = pipelines_lifetime_xr
    prep_data_pipelines["stocks"] = pipelinecap_xr
    prep_data_pipelines["material_intensities"] = pipelines_materials_xr
    prep_data_pipelines["knowledge_graph"] = create_fossil_fuel_graph() 
    prep_data_pipelines["set_unit_flexible"] = prism.U_(prep_data_pipelines["stocks"]) # prism.U_ gives the unit back

    return prep_data_pipelines