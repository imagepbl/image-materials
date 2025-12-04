import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from imagematerials.concepts import create_region_graph
from imagematerials.constants import IMAGE_REGIONS
from imagematerials.rest_of.metals_projections import (
    steel_projection, 
    aluminium_projection, 
    copper_projection)


from imagematerials.rest_of.nmm_projections import (cement_projection, 
                                                    sand_projections, 
                                                    limestone_projection,
                                                    clay_projections) 

from pathlib import Path

def fit_models_all_materials(scenarios_list: list = ["SSP2_M_CP"], path_input_data=None, path_input_data_image=None):
    if path_input_data is None:
        path_input_data = Path("../../../data/raw/rest-of/")
    if path_input_data_image is None:
        path_input_data_image = Path("../../../data/raw/image/")

    results = {}

    for scenario in scenarios_list:
        print(scenario)
        # Run all projections for this scenario
        copper = copper_projection(scenario=scenario, 
                                   path_input_data=path_input_data,
                                   path_input_data_image=path_input_data_image)
        steel = steel_projection(scenario=scenario, 
                                 path_input_data=path_input_data,
                                 path_input_data_image=path_input_data_image)
        aluminium = aluminium_projection(scenario=scenario, 
                                         path_input_data=path_input_data,
                                         path_input_data_image=path_input_data_image)
        cement = cement_projection(scenario=scenario, 
                                   path_input_data=path_input_data,
                                   path_input_data_image=path_input_data_image)
        sand = sand_projections(scenario=scenario, 
                                path_input_data=path_input_data,
                                path_input_data_image=path_input_data_image)
        limestone = limestone_projection(scenario=scenario, 
                                         path_input_data=path_input_data,
                                         path_input_data_image=path_input_data_image)
        clay = clay_projections(scenario=scenario, 
                                path_input_data=path_input_data,
                                path_input_data_image=path_input_data_image)
        # biomass = biomass_data(scenario=scenario)
        # fossil_fuel = fossil_fuel_data(scenario=scenario)
        # water = water_consumption(scenario=scenario)
        
        # Store model objects or just their outputs
        results[scenario] = {
            'copper': copper,
            'steel': steel,
            'aluminium': aluminium,
            'cement': cement,
            'sand': sand,
            'limestone': limestone,
            'clay': clay,
            # 'biomass': biomass,
            # 'fossil_fuel': fossil_fuel,
            # 'water': water
    }
        
    return results



def make_gompertz_coefs_da(results_models, material_order=None, region_order=None, 
                           start_year=1971, end_year=2100):
    """
    Create a DataArray of Gompertz coefficients with desired material and region order.

    Parameters:
        rows (list of dict): Each dict must have keys 'Region', 'material', 'a', 'b', 'c'
        material_order (list of str): Desired order of materials, e.g. ['Steel', 'Cement']
        region_order (list of str): Desired order of regions as strings, e.g. ['1', '2', ..., '26']

    Returns:
        xr.DataArray: (Region, material, coef)
    """
    material_list_complete_fit = ['steel', 'cement', 'limestone', 'clay', 'sand', 'copper']
    material_list_sub_regions_fit = ['aluminium']
    region_order = [str(i) for i in range(1, 27)]
    rows = []

    for material in material_list_complete_fit + material_list_sub_regions_fit:
        print(material)
        # if material in material_list_complete_fit:
        region_model_match = results_models['SSP2_M_CP'][material].new_region_model_match
        # else:
        #     region_model_match = results_models['SSP2_M_CP'][material].region_model_match_per_image

        for region, model in region_model_match.items():
            if model != None:
                rows.append({
                    "Region": region,
                    "material": material,
                    "a": model.coefs[0],
                    "b": model.coefs[1],
                    "c": model.coefs[2]
                })
            else:
                rows.append({
                    "Region": region,
                    "material": material,
                    "a": np.nan,
                    "b": np.nan,
                    "c": np.nan
                })

    coefs_df = pd.DataFrame(rows)
    coefs_df['Region'] = coefs_df['Region'].map(str).str.replace('class_ ', '')
    coefs_df['material'] = coefs_df['material']

    if material_order is None:
        material_order = sorted(coefs_df['material'].unique())
    if region_order is None:
        region_order = sorted(coefs_df['Region'].unique(), key=lambda x: int(x))

    # Set index and reindex to enforce order
    coefs_df = coefs_df.set_index(['Region', 'material']).reindex(
        pd.MultiIndex.from_product([region_order, material_order], names=['Region', 'material'])
    )

    # Convert to xarray Dataset
    coefs_xr = coefs_df[['a', 'b', 'c']].to_xarray()

    # Stack the 'a', 'b', 'c' variables into a new 'coef' dimension
    coefs_da = xr.concat(
        [coefs_xr['a'], coefs_xr['b'], coefs_xr['c']],
        dim='coef'
    ).assign_coords(coef=['a', 'b', 'c'])

    # Expand to include 'Time' dimension 
    years = np.arange(start_year, end_year + 1)
    coefs_da = coefs_da.expand_dims(Time=years)
    coefs_da = xr.ones_like(coefs_da) * coefs_da

    coefs_da = coefs_da.rename("gompertz_coefs")  # <-- Set a descriptive name
    # replace
    
    # convert region names to the standard IMAGE regions
    knowledge_graph_region = create_region_graph()
    coefs_da = knowledge_graph_region.rebroadcast_xarray(coefs_da, output_coords=IMAGE_REGIONS, dim="Region")

    # coefs_da now has dims ('Region', 'material', 'coef')
    coefs_da.to_netcdf('../../../data/raw/rest-of/gompertz_values/coefs_gompertz.nc')
    return coefs_da



def mean_historic_other_fraction_consumption_to_xr(results_models):
    """
    Create a DataArray of mean of last 5 years of historic other fraction consumption for all materials.

    """
    no_full_data_avaiable_on_region_list = [] # limestone, clay
    material_list_complete_fit = ['steel', 'cement', 'limestone', 'clay', 'sand', 'copper']
    material_list_sub_regions_fit = ['aluminium']

    # create an empty xarray dataset
    diff_cons_all = xr.Dataset()

    # save both in one xarray
    for material in material_list_complete_fit + material_list_sub_regions_fit:
        print(material)
        if material in no_full_data_avaiable_on_region_list:
            diff_cons = results_models['SSP2_M_CP']['steel'].historic_other_fraction_consumption.iloc[-5:]
            diff_cons = diff_cons.mean(axis=0)
            # replace all values with np.nan
            diff_cons = diff_cons.mask(diff_cons != 0, np.nan)

        else:
            diff_cons = results_models['SSP2_M_CP'][material].historic_other_fraction_consumption.iloc[-5:]
            diff_cons = diff_cons.mean(axis=0)
        
        # to xarray
        diff_cons = diff_cons.to_xarray()
        # rename coords
        diff_cons = diff_cons.rename({'index': 'Region'})
        # replace dimension of coords Region to '1', '2', 3,... instead of class_ 1, class_ 2, ...
        diff_cons['Region'] = diff_cons['Region'].str.replace('class_ ', '')
        diff_cons_all[material] = diff_cons


    diff_cons_all = diff_cons_all.to_array(dim='material')
    # sort material alphabetically
    diff_cons_all = diff_cons_all.sortby('material')
    # Save the results to a file

    # convert region names to the standard IMAGE regions
    knowledge_graph_region = create_region_graph()
    diff_cons_all = knowledge_graph_region.rebroadcast_xarray(diff_cons_all, output_coords=IMAGE_REGIONS, dim="Region")

    diff_cons_all.to_netcdf('../../../data/raw/rest-of/gompertz_values/diff_cons_all_mean.nc')


def historic_other_fraction_consumption_to_xr(results_models):
    """
    Create a DataArray of historic other fraction consumption for all materials.

    """
    # fill if should be excluded for some materials
    no_full_data_avaiable_on_region_list = []
    material_list_complete_fit = ['steel', 'cement', 'limestone', 'clay', 'sand', 'copper']
    material_list_sub_regions_fit = ['aluminium']

    # create an empty xarray dataset
    diff_cons_all = xr.Dataset()

    # save both in one xarray
    for material in material_list_complete_fit + material_list_sub_regions_fit:
        print(material)
        if material in no_full_data_avaiable_on_region_list:
            # take steel data and replace all values with np.nan for limestone and clay because no diff data available
            diff_cons = results_models['SSP2_M_CP']['steel'].historic_other_fraction_consumption
            diff_cons = diff_cons.mask(diff_cons != 0, np.nan)

        else:
            # take acutal material data
            diff_cons = results_models['SSP2_M_CP'][material].historic_other_fraction_consumption
        
        # to xarray
        diff_cons = diff_cons.to_xarray().to_array()
        # rename coords
        if material in ['cement', 'sand', 'limestone', 'clay']:
            diff_cons = diff_cons.rename({'t': 'Time', 'variable': 'Region'})
        elif material in ['copper']:
            diff_cons = diff_cons.rename({'year': 'Time', 'variable': 'Region'})
        else:
            diff_cons = diff_cons.rename({'index': 'Time', 'variable': 'Region'})

        # replace dimension of coords Region to '1', '2', 3,... instead of class_ 1, class_ 2, ...
        diff_cons['Region'] = diff_cons['Region'].str.replace('class_ ', '')
        # extend years to 2100 and fill with np.nan
        all_years = np.arange(1971, 2101)
        diff_cons = diff_cons.reindex(Time=all_years, fill_value=np.nan)
        diff_cons_all[material] = diff_cons

    diff_cons_all = diff_cons_all.to_array(dim='material')
    # sort material alphabetically
    diff_cons_all = diff_cons_all.sortby('material')
    # Save the results to a file

    # convert region names to the standard IMAGE regions
    knowledge_graph_region = create_region_graph()
    diff_cons_all = knowledge_graph_region.rebroadcast_xarray(diff_cons_all, output_coords=IMAGE_REGIONS, dim="Region")

    diff_cons_all.to_netcdf('../../../data/raw/rest-of/gompertz_values/diff_cons_all.nc')

    return diff_cons_all