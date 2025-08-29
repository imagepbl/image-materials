import pandas as pd

import numpy as np
import xarray as xr
import prism


from imagematerials.concepts import create_region_graph


from imagematerials.buildings.constants import END_YEAR, HIST_YEAR
from imagematerials.util import dataset_to_array, apply_change_per_region

idx = pd.IndexSlice

def compute_mat_intensities_residential(database_dir, circular_economy_config: dict | None=None, **_,):
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
 
    
    # applying material intensity changes for residential buildings
    if 'narrow' in circular_economy_config.keys():
        
        # rename Cohort to Time for compatibility with apply_change_per_region function
        if "Cohort" in xr_mat_res_intensities.dims:
            xr_mat_res_intensities = xr_mat_res_intensities.rename({"Cohort": "Time"}) if "Cohort" in xr_mat_res_intensities.dims else xr_mat_res_intensities
        
        # import parameters from config file 
        target_year = circular_economy_config['narrow']['buildings']['target_year']
        base_year = circular_economy_config['narrow']['buildings']['base_year']
        implementation_rate = circular_economy_config['narrow']['buildings']['implementation_rate']
        mat_changes = circular_economy_config['narrow']['buildings']['material_intensity_change'] 

        region_knowledge_graph = create_region_graph()
        model_regions = list(xr_mat_res_intensities.coords["Region"].values)
        materials_all = set(xr_mat_res_intensities.coords["material"].values)

        for mat in ("Steel", "Concrete", "Aluminium"):
            if mat not in mat_changes or mat not in materials_all:
                continue

            # 1) TOML -> 1-D DA over region names
            change_dict = mat_changes[mat]
            raw = xr.DataArray(
                list(change_dict.values()),
                coords={"Region": list(change_dict.keys())},
                dims=["Region"],
                name=f"material_intensity_change_{mat}",
            )

            # 2) map region names -> region codes, then align to model order 
            regions_mapped = list(region_knowledge_graph.find_relations_inverse(model_regions, raw.coords["Region"].values))
            changes_mapped = region_knowledge_graph.rebroadcast_xarray(raw, output_coords=regions_mapped, dim="Region")
            changes_mapped = changes_mapped.sel(Region=model_regions).astype(float)

            # 3) apply once per material
            cur = xr_mat_res_intensities.sel(material=mat)
            updated = apply_change_per_region(cur, base_year, target_year, changes_mapped, implementation_rate)
            updated = updated.reindex(Region=cur.coords["Region"])
            xr_mat_res_intensities.loc[dict(material=mat)] = updated
            
        # rename back
        if "Time" in xr_mat_res_intensities.dims:
            xr_mat_res_intensities = xr_mat_res_intensities.rename({"Time": "Cohort"})

        print("implemented 'narrow' for Residential Buildings (lightweighting)")

    xr_mat_res_intensities = prism.Q_(xr_mat_res_intensities, "kg/m^2") # assign unit

    return xr_mat_res_intensities

def compute_mat_intensities_commercial(database_dir, circular_economy_config: dict | None=None, **_,):
    materials_commercial = pd.read_csv(database_dir / 'materials_commercial_rasmi.csv', index_col = [0,1]) # 7 building materials in 4 commercial building types; unit: kg/m2; meaning: the average material use per square meter (by commercial building type) 

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


    # broadcast to Regions and order dims
    model_regions = [str(i) for i in range(1, 27)]
    xr_mat_comm_intensities = xr_mat_comm_intensities.expand_dims(Region=model_regions)
    xr_mat_comm_intensities = xr_mat_comm_intensities.transpose("Cohort", "Region", "Type", "material")

    # apply TOML changes (per material, per region)
    if 'narrow' in circular_economy_config.keys():
        # work array with Time dim
        xr_mat = xr_mat_comm_intensities.rename({"Cohort": "Time"}) if "Cohort" in xr_mat_comm_intensities.dims else xr_mat_comm_intensities

        base_year = circular_economy_config['narrow']['buildings']['base_year']
        target_year = circular_economy_config['narrow']['buildings']['target_year']
        impl_rate = circular_economy_config['narrow']['buildings']['implementation_rate']
        mat_changes = circular_economy_config['narrow']['buildings']['material_intensity_change'] 

        region_graph = create_region_graph()
        materials_order = list(xr_mat.coords["material"].values)

        updated_slices = []
        for mat in materials_order:
            cur = xr_mat.sel(material=mat)

            # only apply for those present in TOML; others pass through unchanged
            if mat in mat_changes:
                change_dict = mat_changes[mat] 
                raw = xr.DataArray(
                    list(change_dict.values()),
                    coords={"Region": list(change_dict.keys())},
                    dims=["Region"],
                    name=f"mi_change_pc_{mat}",
                )

                # map region names -> region codes; align to model order
                regions_mapped = list(region_graph.find_relations_inverse(model_regions, raw.coords["Region"].values))
                changes_mapped = region_graph.rebroadcast_xarray(raw, output_coords=regions_mapped, dim="Region")
                changes_mapped = changes_mapped.sel(Region=model_regions).astype(float)

                # apply once per material
                upd = apply_change_per_region(cur, base_year, target_year, changes_mapped, impl_rate)
                # keep Region order & dim order identical to cur
                upd = upd.reindex(Region=cur.coords["Region"]).transpose(*cur.dims)
            else:
                upd = cur

            # attach the material coord and collect
            updated_slices.append(upd.expand_dims(material=[mat]))

        xr_mat_updated = xr.concat(updated_slices, dim="material")

        # rename back to Cohort if needed
        xr_mat_comm_intensities = xr_mat_updated.rename({"Time": "Cohort"}) if "Time" in xr_mat_updated.dims else xr_mat_updated

        print("implemented 'narrow' for Commercial Buildings (lightweighting)")

    xr_mat_comm_intensities = prism.Q_(xr_mat_comm_intensities, "kg/m^2") # assign unit

    return xr_mat_comm_intensities
