
import math

import numpy as np
import pandas as pd
import xarray as xr

from imagematerials.buildings.constants import (
    END_YEAR,
    FLAG_ALPHA,
    FLAG_EXPDEC,
    GOMPERTZ_EXPDEC,
    HIST_YEAR,
    INFLATION,
    REGIONS,
    REGIONS_RANGE,
    START_YEAR,
    YEARS,
    SCENARIO_SELECT
)
from imagematerials.read_mym import read_mym_df
from imagematerials.util import dataset_to_array, merge_dims

far_start_year = 1721
start_year = 1820
end_year = 1970
idx = pd.IndexSlice

years_1721_1820 = xr.DataArray(np.arange(far_start_year, start_year), dims=["Time"], coords={"Time": np.arange(far_start_year, start_year)})
years_1820_1971 = xr.DataArray(np.arange(start_year, end_year+1), dims=["Time"], coords={"Time": np.arange(start_year, end_year+1)})
years_1820_1970 = xr.DataArray(np.arange(start_year, end_year), dims=["Time"], coords={"Time": np.arange(start_year, end_year)})


def get_gompertz(base_directory):
    # Load fitted regression parameters
    if FLAG_ALPHA == 0:
        gompertz = pd.read_csv(base_directory / 'buildings/files_commercial/Gompertz_parameters.csv', index_col = [0])
    else:
        gompertz = pd.read_csv(base_directory / 'buildings/files_commercial/Gompertz_parameters_alpha.csv', index_col = [0])
    return gompertz

def get_service_value_added(image_directory):
    # we use the inflation corrected SVA to adjust for the fact that IMAGE provides gdp/cap in 2005 US$
    # TODO: check with IMAGE team if SVA is still given in 2005 values --> if not correct inflation factor
    service_value_added_2005: pd.DataFrame = read_mym_df(image_directory.joinpath("Socioeconomic", "sva_pc.scn"))
    service_value_added = service_value_added_2005 * INFLATION
    
    # extrapolate to 1970, therefore first add empty 1970 value with nans
    service_value_added.loc[1970] = np.nan
    # TODO cubic does not work, replaced with linear for now, Sebastiaans code had cubic, seems like now just value from 1971 is copied
    service_value_added = service_value_added.sort_index().interpolate(method='linear', limit_direction="both") 

    return service_value_added

def get_image_floorspace(image_directory, base_directory):
    gompertz = get_gompertz(base_directory)
    service_value_added = get_service_value_added(image_directory)
    commercial_m2_cap_sum = compute_commercial_floor_m2_cap_sum(gompertz, service_value_added)
    commercial_m2_cap, minimum_comm = compute_commercial_floor_m2_cap(gompertz, commercial_m2_cap_sum, service_value_added)
    floorspace_urb_rur = get_floorspace_urban_rural(image_directory)

    floorspace_all = floorspace_urb_rur.merge(commercial_m2_cap, how = "left", left_index=True, right_index=True)
    floorspace_dataset = floorspace_all.to_xarray()

    floorspace_xr = xr.DataArray(0.0, dims=("Time", "Region", "Type"), coords={
        "Time": floorspace_dataset.coords["Time"],
        "Region": floorspace_dataset.coords["Region"],
        "Type": ["Urban", "Rural", "Office", "Retail+", "Hotels+", "Govt+"],
    })

    for data_name, data_var in floorspace_dataset.data_vars.items():
        floorspace_xr.loc[:, :, data_name] = data_var
    floorspace_xr.coords["Region"] = [str(x.values) for x in floorspace_xr.coords["Region"]]
    return floorspace_xr, minimum_comm

def extrapolate_floorspace(floorspace_image, minimum_comm):
    # For the RESIDENTIAL & COMMERCIAL floorspace: Derive the annual trend (in m2/cap) 
    # over the initial 10 years of IMAGE data
    # Get the growth by year (for the first 10 years)
    interp_coor = floorspace_image.sel(Time=range(1971, 1981)).coords
    trend_1971_1981 = xr.DataArray(
        floorspace_image.sel(Time=range(1971, 1981)).to_numpy()/floorspace_image.sel(Time=range(1972, 1982)).to_numpy(),
        dims=("Time", "Region", "Type"),
        coords=interp_coor
    )

    # Average global annual decline in floorspace/cap in %, rural: 1%; urban 1.2%;  commercial: 1.26-2.18% /yr 
    avg_trend_1971_1981 = trend_1971_1981.mean(["Time", "Region"])

    # Find minumum or maximum values in the original IMAGE data 
    # (Just for residential, commercial minimum values have been calculated above)
    # min_floorspace = floorspace_image.min(["Time", "Region"])
    min_floorspace = xr.concat((floorspace_image.sel(Type=["Urban", "Rural"]).min(["Time", "Region"]), minimum_comm), dim="Type")
    # Compute the floorspace between 1820 and 1970 with extrapolation
    floor_1820_1970 = floorspace_image.loc[1971]*avg_trend_1971_1981**(end_year+1-years_1820_1971)
    floor_1820_1970 = floor_1820_1970.where(floor_1820_1970 > min_floorspace, min_floorspace).transpose()

    floor_1721_1820 = floor_1820_1970[0]*(1-(start_year - years_1721_1820)/(start_year-far_start_year+1)).transpose()

    # combine historic with IMAGE data here
    floorspace = xr.concat((floor_1721_1820, floor_1820_1970, floorspace_image), dim="Time")
    return floorspace.transpose()


def get_floorspace_urban_rural(image_directory):
    # load IMAGE data-files (MyM file format)
    floorspace: pd.DataFrame = read_mym_df(image_directory.joinpath("EnergyServices", "res_FloorSpace.out"))
    floorspace = floorspace[['time','DIM_1',2,3]].rename(columns={"DIM_1": "Region", 'time':'t', 2:'Urban', 3:'Rural'})
    # the other columns are average per capita floorspace per quintile (we also exclude the average per capita floorspace of the total population in column 1, 
    # because we use the urban & rural specific totals)
    floorspace = floorspace[floorspace.Region != REGIONS + 1] #removing region 27
    floorspace = floorspace[floorspace['t'].isin(list(range(START_YEAR, END_YEAR+1)))]
    # remove all data beyond 2060 to save runtime, we have not yet generated scenario results beyond 2060
    floorspace = floorspace.rename({"t":"Time"}, axis = 1)
    floorspace = floorspace.set_index(["Time", "Region"])
    floorspace = floorspace.rename_axis("Type", axis = 1)
    return floorspace


def compute_commercial_floor_m2_cap_sum(gompertz, service_value_added):
    # Select gompertz curve paramaters for the total commercial m2 demand (stock)
    alpha, beta, gamma = (
        (gompertz['All']['a'], gompertz['All']['b'], gompertz['All']['c'])
        if FLAG_EXPDEC == 0 else GOMPERTZ_EXPDEC          
    )
    # alpha_low = alpha * LOWCOMM 
    alpha_low = alpha

    # find the total commercial m2 stock (in Millions of m2)
    commercial_m2_cap = pd.DataFrame(index=YEARS, columns=REGIONS_RANGE)
    commercial_m2_cap_low = commercial_m2_cap.copy()

    # Compute commercial floorspace using Gompertz curves
    for year in YEARS:
        for region in REGIONS_RANGE:
            exp_factor = math.exp((-gamma/1000) * service_value_added[region][year])
            if FLAG_EXPDEC == 0:
                commercial_m2_cap[region][year] = alpha * math.exp(-beta * exp_factor)
                commercial_m2_cap_low[region][year] = alpha_low * math.exp(-beta * exp_factor)
            else:
                commercial_m2_cap[region][year] = max(0.542, alpha - beta * exp_factor)
            

    # commercial floorspace is scaled here (in case lowComm is not 1)
    scale_comm = pd.Series([1.0, 0.0], index=[2020, 2060], name='time').reindex(YEARS).interpolate(
        method='linear', limit=300, limit_direction='both')
    commercial_m2_cap = commercial_m2_cap.mul(scale_comm, axis=0) + commercial_m2_cap_low.mul((1-scale_comm), axis=0)
    return commercial_m2_cap

def compute_commercial_floor_m2_cap(gompertz, commercial_m2_cap_sum, service_value_added):
    # Subdivide the total commercial floorspace across Offices, Retail+, Govt+ & Hotels+
    types = ["Office", "Retail+", "Hotels+", "Govt+"]
    index = pd.MultiIndex.from_product([types, REGIONS_RANGE, YEARS], names=["Type", "Region", "Time"])
    commercial_m2_cap_all = pd.DataFrame(index=index, columns=["m2_per_cap"]).fillna(0)
    minimum_comm = xr.DataArray(25.0, dims=["Type"], coords={"Type": types})
    for year in YEARS[1:]:
        for region in REGIONS_RANGE:
            # Calculate floorspace for all types and update the minimum values
            floorspace_commercial_list = {}
            for type_ in types:
                #value = gompertz_value(type_, region, year, service_value_added)
                params = gompertz[type_]
                value = params['a'] * math.exp(-params['b'] * math.exp((-params['c'] / 1000) * service_value_added[region][year]))
                floorspace_commercial_list[type_] = value
                minimum_comm.loc[type_] = min(minimum_comm.loc[type_], value)

            # Sum all floorspace values for normalization
            commercial_sum = sum(floorspace_commercial_list.values())

            # Calculate and assign the floorspace for each type
            for type_ in types:
                commercial_m2_cap_all.loc[(type_, region, year), "m2_per_cap"] = (
                    commercial_m2_cap_sum.loc[year, region] * (floorspace_commercial_list[type_] / commercial_sum)
                )

    commercial_m2_cap_all = commercial_m2_cap_all.unstack("Type")
    commercial_m2_cap_all = commercial_m2_cap_all.droplevel(0, axis = 1)
    return commercial_m2_cap_all, minimum_comm

def compute_housing_type(database_directory):
    housing_type_data: pd.DataFrame = pd.read_csv(database_directory.joinpath('Housing_type_dynamic.csv'), index_col = [0,1,2]) 
    # also interpolate housing type data
    index_ht = pd.MultiIndex.from_product([list(range(HIST_YEAR, END_YEAR + 1)), 
                                        list(range(1,REGIONS + 1)), 
                                        ['Urban', 'Rural'] ]) 
    housing_type = pd.DataFrame(np.nan, index=index_ht, columns=housing_type_data.columns)
    housing_type.index.names = ['Time','Region','Area']

    for year in list(housing_type_data.index.levels[0]):
        housing_type.loc[idx[year,:,:],:] = housing_type_data.loc[idx[year,:,:],:]

    for region in list(range(1,REGIONS + 1)):
        for area in ['Urban', 'Rural']:
            housing_types_interpolated = housing_type.loc[idx[:,region,area],:].interpolate(method='linear', limit_direction='both')
            housing_type.loc[idx[:,region,area],:] = housing_types_interpolated.values   

    housing_type_xr = dataset_to_array(housing_type.to_xarray(), ["Time", "Region", "Area"], ["Type"])
    housing_type_xr.coords["Region"] = [str(x.values) for x in housing_type_xr.coords["Region"]]
    return housing_type_xr

def compute_average_m2_capita(base_directory):
    average_m2_capita_df: pd.DataFrame = pd.read_csv(base_directory.joinpath('buildings', 'files_DB','Average_m2_per_cap.csv'), index_col = [0,1]) 
    column_mapping = {'1': 'Detached', '2': 'Semi-detached', '3': 'Appartment', '4': 'High-rise'}
    average_m2_capita_df.rename(columns=column_mapping, inplace=True)
    average_m2_capita = dataset_to_array(average_m2_capita_df.to_xarray(), ["Region", "Area"], ["Type"])
    average_m2_capita.coords["Region"] = [str(x.values) for x in average_m2_capita.coords["Region"]]
    return average_m2_capita

def compute_housing_residential(population, average_m2_capita, housing_type, floorspace_rururb):
    m2_housing_per_capita = average_m2_capita * housing_type
    m2_housing_share = m2_housing_per_capita / m2_housing_per_capita.sum(["Type"])
    total_m2_housing_per_cap = m2_housing_share*floorspace_rururb

    if 'base' in circular_economy_config.keys():
        base_year = circular_economy_config["base"]["buildings"]["base_year"]
        target_year = circular_economy_config["base"]["buildings"]["target_year"]
        floor_pc_2020 = circular_economy_config["base"]["buildings"]["residential"]["2020"]["useful_floor_pc"]
        region_knowledge_graph = create_region_graph()
        regions = total_m2_housing_per_cap.coords["Region"].values
        floor_pc_2020_xr = xr.DataArray(
            list(floor_pc_2020.values()),
            coords={"Region": list(floor_pc_2020.keys())},
            dims=["Region"],
            name="floor_pc_2020"
        )

        #regions_mapped = region_knowledge_graph.find_relations_inverse(regions, floor_pc_2020.keys())
        floor_pc_2020_mapped = region_knowledge_graph.rebroadcast_xarray(floor_pc_2020_xr, output_coords=regions, dim="Region")
        target_vals = floor_pc_2020_mapped
        current_vals = total_m2_housing_per_cap.sel(Time=2020)\
                                            .sum(dim="Type")\
                                            .mean(dim="Area")

        scaling_factors = target_vals / current_vals

        total_m2_housing_per_cap = total_m2_housing_per_cap * scaling_factors

    total_m2_housing = total_m2_housing_per_cap * population.sel({"Area": ["Rural", "Urban"]})
    floorspace_residential = merge_dims(total_m2_housing, "Type", "Area")
    return floorspace_residential.transpose("Time", "Region", "Type")

# #TODO move to a util file
# # Define a function to calculate Gompertz growth
# def gompertz_value(category, region, year, sva_data):
#     """Calculate the Gompertz value for a given category, region, and year."""
#     params = gompertz[category]
#     return params['a'] * math.exp(-params['b'] * math.exp((-params['c'] / 1000) * sva_data[str(region)][year]))
