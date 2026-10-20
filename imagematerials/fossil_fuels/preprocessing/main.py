#%% Import general constants and modules 
import pandas as pd
import numpy as np
from pathlib import Path
import pint
import xarray as xr

import prism

from imagematerials.util import convert_lifetime
from imagematerials.concepts import create_fossil_fuel_graph, create_region_graph
from imagematerials.constants import IMAGE_REGIONS
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

region_knowledge_graph = create_region_graph()
fossil_fuel_knowledge_graph = create_fossil_fuel_graph()

# The input data and the infrastructure stocks (coal, oil, gas) can only be computed once we know
# path_base and climate_policy_config, which are given to the get_preprocessing_data_* functions below.
# get_inputs_and_infra() does this when it is called for the first time and remembers the result,
# so the other stages do not repeat the work.
_inputs_and_infra = {}

def get_inputs_and_infra(path_base, climate_policy_config):
    """Return (inputs, infra) for this scenario; load and compute them only the first time.

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
def get_preprocessing_data_extraction(path_base: str,
                                      climate_policy_config: dict,
                                      circular_economy_config: dict,
                                      scenario: str,
                                      year_start: int,
                                      year_end: int,
                                      year_out: int):

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
    # add units
    prep_data_extraction["stocks"] = prism.Q_(prep_data_extraction["stocks"], "kg")
    prep_data_extraction["material_intensities"] = prism.Q_(prep_data_extraction["material_intensities"], "kg/kg")
    prep_data_extraction["set_unit_flexible"] = prism.U_(prep_data_extraction["stocks"]) # prism.U_ gives the unit back

    region_knowledge_graph = create_region_graph()
    prep_data_extraction["stocks"] = prep_data_extraction["stocks"].assign_coords(Region=prep_data_extraction["stocks"].coords["Region"].astype(str))
    prep_data_extraction["stocks"] = region_knowledge_graph.rebroadcast_xarray(prep_data_extraction["stocks"], output_coords=IMAGE_REGIONS, dim="Region")
    # set_unit_flexible is needed by the model to deal with the fact the in the beginning of the model it doesn't know th data yet and needs to work with a placeholder/flexible unit (see model.py) 

    return prep_data_extraction


####################################################################################################
#%%Processing stage (coal preparation and oil refinery)
####################################################################################################
def get_preprocessing_data_processing(path_base: str,
                                      climate_policy_config: dict,
                                      circular_economy_config: dict,
                                      scenario: str,
                                      year_start: int,
                                      year_end: int,
                                      year_out: int):

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
    # add units
    prep_data_processing["stocks"] = prism.Q_(prep_data_processing["stocks"], "kg")
    prep_data_processing["material_intensities"] = prism.Q_(prep_data_processing["material_intensities"], "kg/kg")
    prep_data_processing["set_unit_flexible"] = prism.U_(prep_data_processing["stocks"]) # prism.U_ gives the unit back

    region_knowledge_graph = create_region_graph()
    prep_data_processing["stocks"] = prep_data_processing["stocks"].assign_coords(Region=prep_data_processing["stocks"].coords["Region"].astype(str))
    prep_data_processing["stocks"] = region_knowledge_graph.rebroadcast_xarray(prep_data_processing["stocks"], output_coords=IMAGE_REGIONS, dim="Region")
    #     # set_unit_flexible is needed by the model to deal with the fact the in the beginning of the model it doesn't know th data yet and needs to work with a placeholder/flexible unit (see model.py) 

    return prep_data_processing


####################################################################################################
#%%Processing stage (gas processing and oil storage)
####################################################################################################
def get_preprocessing_data_storage(path_base: str,
                                   climate_policy_config: dict,
                                   circular_economy_config: dict,
                                   scenario: str,
                                   year_start: int,
                                   year_end: int,
                                   year_out: int):

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
    # add units
    prep_data_storage["stocks"] = prism.Q_(prep_data_storage["stocks"], "meter**3")
    prep_data_storage["material_intensities"] = prism.Q_(prep_data_storage["material_intensities"], "kg/meter**3")
    prep_data_storage["set_unit_flexible"] = prism.U_(prep_data_storage["stocks"]) # prism.U_ gives the unit back

    region_knowledge_graph = create_region_graph()
    prep_data_storage["stocks"] = prep_data_storage["stocks"].assign_coords(Region=prep_data_storage["stocks"].coords["Region"].astype(str))
    prep_data_storage["stocks"] = region_knowledge_graph.rebroadcast_xarray(prep_data_storage["stocks"], output_coords=IMAGE_REGIONS, dim="Region")
    #     # set_unit_flexible is needed by the model to deal with the fact the in the beginning of the model it doesn't know th data yet and needs to work with a placeholder/flexible unit (see model.py) 

    return prep_data_storage


####################################################################################################
#%% Transport stage
####################################################################################################
def get_preprocessing_data_transport(path_base: str,
                                     climate_policy_config: dict,
                                     circular_economy_config: dict,
                                     scenario: str,
                                     year_start: int,
                                     year_end: int,
                                     year_out: int):

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

    # add units
    prep_data_transport["stocks"] = prism.Q_(prep_data_transport["stocks"], "kg")
    prep_data_transport["material_intensities"] = prism.Q_(prep_data_transport["material_intensities"], "kg/kg")
    prep_data_transport["set_unit_flexible"] = prism.U_(prep_data_transport["stocks"]) # prism.U_ gives the unit back

    region_knowledge_graph = create_region_graph()
    prep_data_transport["stocks"] = prep_data_transport["stocks"].assign_coords(Region=prep_data_transport["stocks"].coords["Region"].astype(str))
    prep_data_transport["stocks"] = region_knowledge_graph.rebroadcast_xarray(prep_data_transport["stocks"], output_coords=IMAGE_REGIONS, dim="Region")
        # set_unit_flexible is needed by the model to deal with the fact the in the beginning of the model it doesn't know th data yet and needs to work with a placeholder/flexible unit (see model.py) 
    return prep_data_transport


####################################################################################################
#%% Pipelines stage (oil, gas)
####################################################################################################
def get_preprocessing_data_pipelines(path_base: str,
                                     climate_policy_config: dict,
                                     circular_economy_config: dict,
                                     scenario: str,
                                     year_start: int,
                                     year_end: int,
                                     year_out: int):

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
    # add units
    prep_data_pipelines["stocks"] = prism.Q_(prep_data_pipelines["stocks"], "km")
    prep_data_pipelines["material_intensities"] = prism.Q_(prep_data_pipelines["material_intensities"], "kg/km")
    prep_data_pipelines["set_unit_flexible"] = prism.U_(prep_data_pipelines["stocks"]) # prism.U_ gives the unit back

    region_knowledge_graph= create_region_graph()
    prep_data_pipelines["stocks"] = prep_data_pipelines["stocks"].assign_coords(Region=prep_data_pipelines["stocks"].coords["Region"].astype(str))
    prep_data_pipelines["stocks"] = region_knowledge_graph.rebroadcast_xarray(prep_data_pipelines["stocks"], output_coords=IMAGE_REGIONS, dim="Region")
    # set_unit_flexible is needed by the model to deal with the fact the in the beginning of the model it doesn't know th data yet and needs to work with a placeholder/flexible unit (see model.py) 

    return prep_data_pipelines