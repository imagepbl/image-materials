import pandas as pd

from imagematerials.buildings.constants import END_YEAR, HIST_YEAR
from imagematerials.util import dataset_to_array
from imagematerials.vehicles.modelling_functions import apply_change_per_region

idx = pd.IndexSlice



def compute_mat_intensities_residential(database_dir, circular_economy_config):
    building_materials = pd.read_csv(database_dir / 'Building_materials_rasmi.csv', index_col = [0,1,2])   # Building_materials; unit: kg/m2; meaning: the average material use per square meter (by building type, by region & by area)
    building_materials_dynamic   = pd.DataFrame(index=pd.MultiIndex.from_product([list(range(HIST_YEAR, END_YEAR + 1)), list(range(1,27)), list(range(1,5))]), columns=building_materials.columns)

    # interpolate material intensity data from files (residential buildings)
    for material in building_materials.columns:
        for building in list(building_materials.index.levels[2]):
            selection = building_materials.loc[idx[:,:,building],material].droplevel(2, axis=0).unstack()
            selection.loc[HIST_YEAR,:] = selection.loc[selection.first_valid_index(),:]
            selection.loc[END_YEAR + 1,:] = selection.loc[selection.last_valid_index(),:]
            selection = selection.reindex(list(range(HIST_YEAR, END_YEAR + 1))).interpolate()
            building_materials_dynamic.loc[idx[:,:,building], material] = selection.stack()

    xr_mat_res_intensities = dataset_to_array(building_materials_dynamic.to_xarray(), ["Cohort", "Region", "Type"], ["material"])
    xr_mat_res_intensities.coords["Type"] = ["Detached", "Semi-detached", "Appartment", "High-rise"]
    xr_mat_res_intensities.coords["Region"] = [str(x) for x in xr_mat_res_intensities.coords["Region"].values]
    xr_mat_res_intensities_lightweight = xr_mat_res_intensities.copy(deep=True)
    
    # applying material intensity changes for residential buildings
    if 'narrow' in circular_economy_config.keys():
        target_year = circular_economy_config['narrow']['buildings']['target_year']
        base_year = circular_economy_config['narrow']['buildings']['base_year']
        material_intensity_change = circular_economy_config['narrow']['buildings']['material_intensity_change']
        implementation_rate = circular_economy_config['narrow']['buildings']['implementation_rate']
        
        for mat, pct_change in material_intensity_change.items():
            if "material" in xr_mat_res_intensities_lightweight.coords and mat in xr_mat_res_intensities_lightweight.coords['material'].values:
                cur = xr_mat_res_intensities_lightweight.sel(material=mat)
                updated = apply_change_per_region(
                    cur, base_year, target_year, pct_change, implementation_rate
                )
                xr_mat_res_intensities_lightweight.loc[dict(material=mat)] = updated

        print("implemented 'narrow' for Residential Buildings (lightweighting)")
    
    return xr_mat_res_intensities, xr_mat_res_intensities_lightweight


def compute_mat_intensities_commercial(database_dir,circular_economy_config):
    materials_commercial = pd.read_csv(database_dir / 'materials_commercial.csv', index_col = [0,1]) # 7 building materials in 4 commercial building types; unit: kg/m2; meaning: the average material use per square meter (by commercial building type) 

    #First: interpolate the dynamic material intensity data
    materials_commercial_dynamic = pd.DataFrame(index=pd.MultiIndex.from_product([list(range(HIST_YEAR, END_YEAR + 1)), list(materials_commercial.index.levels[1])]), columns=materials_commercial.columns)


    # interpolate material intensity data from files (commercial buildings)
    for building in materials_commercial.columns:
        selection = materials_commercial.loc[idx[:,:], building].unstack()
        selection.loc[HIST_YEAR,:] = selection.loc[selection.first_valid_index(),:]
        selection.loc[END_YEAR + 1,:] = selection.loc[selection.last_valid_index(),:]
        selection = selection.reindex(list(range(HIST_YEAR, END_YEAR + 1))).interpolate()
        materials_commercial_dynamic.loc[idx[:,:], building] = selection.stack()

    xr_mat_comm_intensities = dataset_to_array(materials_commercial_dynamic.to_xarray(), ["Cohort", "material"], ["Type"])
    xr_mat_comm_intensities.coords["Type"] = ["Office", "Retail+", "Hotels+", "Govt+"]

    #  applying material intensity changes for commercial buildings
    xr_mat_comm_intensities_lightweight = xr_mat_comm_intensities.copy(deep=True)
    if 'narrow' in circular_economy_config.keys():
        
        target_year = circular_economy_config['narrow']['buildings']['target_year']
        base_year = circular_economy_config['narrow']['buildings']['base_year']
        material_intensity_change = circular_economy_config['narrow']['buildings']['material_intensity_change']
        implementation_rate = circular_economy_config['narrow']['buildings']['implementation_rate']
        
        for mat, pct_change in material_intensity_change.items():
            if "material" in xr_mat_comm_intensities_lightweight.coords and mat in xr_mat_comm_intensities_lightweight.coords['material'].values:
                cur = xr_mat_comm_intensities_lightweight.sel(material=mat)
                updated = apply_change_per_region(
                    cur, base_year, target_year, pct_change, implementation_rate
                )
                xr_mat_comm_intensities_lightweight.loc[dict(material=mat)] = updated

        print("implemented 'narrow' for Commercial Buildings (lightweighting)")

    return xr_mat_comm_intensities, xr_mat_comm_intensities_lightweight
