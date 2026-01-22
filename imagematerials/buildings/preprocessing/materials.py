"""Preprocessing module for the material intensities."""
from pathlib import Path

import pandas as pd
import prism
import xarray as xr

from imagematerials.buildings.constants import END_YEAR, HIST_YEAR
from imagematerials.buildings.preprocessing.circular_economy_measures import (
    circular_economy_measures_material_intensities_commercial,
    circular_economy_measures_material_intensities_residential,
)
from imagematerials.util import dataset_to_array

idx = pd.IndexSlice

def compute_mat_intensities_residential(database_dir: Path,
                                        circular_economy_config: dict | None = None,
                                        **_,) -> xr.DataArray:
    """Compute the material intensities for residential buildings.

    Parameters
    ----------
    database_dir:
        Directory containing the data for the material intensities.
    circular_economy_config:
        Configuration for the circular economy settings.


    Returns
    -------
    xr_mat_res_intensities:
        Material intensities for all residential buildings in kg/m^2.

    """
    # Building_materials; unit: kg/m2; meaning: the average material use per square meter
    # (by building type, by region & by area)
    building_materials = pd.read_csv(database_dir / 'Building_materials_rasmi.csv',
                                     index_col = [0,1,2])

    if 'resource_efficient' in circular_economy_config.keys():
        # Building_materials; unit: kg/m2; meaning: the average material use per square meter
        # (by building type, by region & by area)
        building_materials = pd.read_csv(
            database_dir / 'Building_materials_rasmi_resource_efficient.csv',
            index_col = [0,1,2])
        print("Applied using resource_efficient building materials intensities "
              "for residential buildings.")
    building_materials_dynamic = pd.DataFrame(index=pd.MultiIndex.from_product(
        [list(range(HIST_YEAR, END_YEAR + 1)), list(range(1,27)), list(range(1,5))]),
                                              columns=building_materials.columns)

    # interpolate material intensity data from files (residential buildings)
    for material in building_materials.columns:
        for building in list(building_materials.index.levels[2]):
            selection = building_materials.loc[idx[:,:,building],material].droplevel(
                2, axis=0).unstack()
            selection.loc[HIST_YEAR,:] = selection.loc[selection.first_valid_index(),:]
            selection.loc[END_YEAR + 1,:] = selection.loc[selection.last_valid_index(),:]
            selection = selection.reindex(list(range(HIST_YEAR, END_YEAR + 1))).interpolate()
            building_materials_dynamic.loc[idx[:,:,building], material] = selection.stack()

    xr_mat_res_intensities = dataset_to_array(building_materials_dynamic.to_xarray(),
                                              ["Cohort", "Region", "Type"], ["material"])
    xr_mat_res_intensities.coords["Type"] = ["Detached", "Semi-detached", "Appartment", "High-rise"]
    xr_mat_res_intensities.coords["Region"] = [
        str(x) for x in xr_mat_res_intensities.coords["Region"].values]

    # applying material intensity changes for residential buildings
    if "narrow_product" in circular_economy_config.keys(): 
        xr_mat_res_intensities = circular_economy_measures_material_intensities_residential(xr_mat_res_intensities, 
                                                                                            circular_economy_config)

    xr_mat_res_intensities = prism.Q_(xr_mat_res_intensities, "kg/m^2") # assign unit

    return xr_mat_res_intensities


def compute_mat_intensities_commercial(
        database_dir: Path,
        circular_economy_config: dict | None=None,
        **_,) -> xr.DataArray:
    """Compute material intensities for commercial buildings.

    Parameters
    ----------
    database_dir:
        Base path to the data used for reading material intensities.
    circular_economy_config:
        Configuration for the circular economy settings.

    Returns
    -------
    xr_mat_comm_intensities:
        Material intensities in kg/m^2 for commercial buildings.

    """
     # 7 building materials in 4 commercial building types; unit: kg/m2; meaning:
     # the average material use per square meter (by commercial building type)
    materials_commercial = pd.read_csv(database_dir / 'materials_commercial_rasmi.csv',
                                       index_col = [0,1])

    #First: interpolate the dynamic material intensity data
    materials_commercial_dynamic = pd.DataFrame(index=pd.MultiIndex.from_product(
        [list(range(HIST_YEAR, END_YEAR + 1)), list(materials_commercial.index.levels[1])]),
                                                columns=materials_commercial.columns)


    # interpolate material intensity data from files (commercial buildings)
    for building in materials_commercial.columns:
        selection = materials_commercial.loc[idx[:,:], building].unstack()
        selection.loc[HIST_YEAR,:] = selection.loc[selection.first_valid_index(),:]
        selection.loc[END_YEAR + 1,:] = selection.loc[selection.last_valid_index(),:]
        selection = selection.reindex(list(range(HIST_YEAR, END_YEAR + 1))).interpolate()
        materials_commercial_dynamic.loc[idx[:,:], building] = selection.stack()

    xr_mat_comm_intensities = dataset_to_array(materials_commercial_dynamic.to_xarray(),
                                               ["Cohort", "material"], ["Type"])
    xr_mat_comm_intensities.coords["Type"] = ["Office", "Retail+", "Hotels+", "Govt+"]

    # broadcast to Regions and order dims
    model_regions = [str(i) for i in range(1, 27)]
    xr_mat_comm_intensities = xr_mat_comm_intensities.expand_dims(Region=model_regions)
    xr_mat_comm_intensities = xr_mat_comm_intensities.transpose("Cohort", "Region",
                                                                "Type", "material")

    # apply CE changes (per material, per region)
    if "narrow_product" in circular_economy_config.keys():
        xr_mat_comm_intensities = circular_economy_measures_material_intensities_commercial(xr_mat_comm_intensities, circular_economy_config, model_regions)

    xr_mat_comm_intensities = prism.Q_(xr_mat_comm_intensities, "kg/m^2") # assign unit

    return xr_mat_comm_intensities
