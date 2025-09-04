#%%
import pandas as pd
import numpy as np
import os
import scipy
import warnings
from pathlib import Path
import pint
import xarray as xr
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from matplotlib.ticker import ScalarFormatter

# path_current = Path(__file__).resolve().parent # absolute path of file
# path_base = path_current.parent.parent # base path of the project -> image-materials
# sys.path.append(str(path_base))

import prism
from imagematerials.distribution import ALL_DISTRIBUTIONS, NAME_TO_DIST
from imagematerials.read_mym import read_mym_df
from imagematerials.util import dataset_to_array, pandas_to_xarray, convert_life_time_vehicles
from imagematerials.model import GenericMainModel, GenericMaterials, GenericStocks, Maintenance, MaterialIntensities
from imagematerials.factory import ModelFactory, Sector
from imagematerials.concepts import create_electricity_graph
from imagematerials.electricity.utils import MNLogit, stock_tail, create_prep_data,print_df_info


from imagematerials.electricity.constants import (
    STANDARD_SCEN_EXTERNAL_DATA,
    YEAR_START,
    YEAR_FIRST,
    YEAR_FIRST_GRID,
    YEAR_END,
    YEAR_OUT,
    YEAR_SWITCH,
    YEAR_LAST,
    # COHORTS, # necessary?
    SCEN,
    VARIANT,
    SENS_ANALYSIS,
    REGIONS,
    TECH_GEN,
    STD_LIFETIMES_ELECTR,
    MEGA_TO_TERA,
    PKMS_TO_VKMS,
    TONNES_TO_KGS,
    LOAD_FACTOR,
    BEV_CAPACITY_CURRENT,
    PHEV_CAPACITY_CURRENT,
    unit_mapping,
    gen_tech_to_category,
    dict_gentech_styles,
    dict_gentechcat_colors,
    dict_materials_colors,
    dict_grid_colors,
    dict_grid_styles,
    dict_grid_styles2
)

from imagematerials.electricity.electr_external_data import (
    df_iea_cu_aps,
    df_iea_cu_nzs,
    df_iea_co_aps,
    df_iea_mn_aps,
    df_iea_ni_aps
)
VARIANT = "VLHO"
# VARIANT = "M_CP"
# Define paths ----------------------------------------------------------------------
# scen_folder = SCEN + "_" + VARIANT
# # path_base = Path().resolve() # TODO absolute path of file "preprocessing.py" ? current solution can differ depending on IDE used (?) 
# path_current = Path().resolve()
# path_base = path_current.parent.parent # base path of the project -> image-materials

# # path_image_output = Path(path_base, "data", "raw", SCEN, "EnergyServices")
# path_image_output = Path(path_base, "data", "raw", scen_folder, "EnergyServices")
# path_external_data_standard = Path(path_base, "data", "raw", "electricity", "standard_data")
# path_external_data_scenario = Path(path_base, "data", "raw", "electricity", scen_folder)

# assert path_image_output.is_dir()
# assert path_external_data_standard.is_dir()
# assert path_external_data_scenario.is_dir()

# create the folder out_test if it does not exist
# if not (path_base / 'imagematerials' / 'electricity' / 'out_test').is_dir():
#     (path_base / 'imagematerials' / 'electricity' / 'out_test').mkdir(parents=True)



years = YEAR_END - YEAR_START  + 1


# from past.builtins import execfile
# execfile('read_mym.py')
idx = pd.IndexSlice             # needed for slicing multi-index


# TODO: decide on variable naming convention
# V1: sector_variableinquestion_additionalinformation (gcap_lifetime_interpolated)
# V2: variableinquestion_sector_additionalinformation (lifetime_gcap_interpolated)

###########################################################################################################
###########################################################################################################
#%% 1) Generation 
###########################################################################################################
###########################################################################################################



def get_preprocessing_data_gen(path_base: str, SCEN, VARIANT): #, climate_policy_config: dict, circular_economy_config: dict

    scen_folder = SCEN + "_" + VARIANT
    path_image_output = Path(path_base, "data", "raw", "image", scen_folder, "EnergyServices")
    path_external_data_standard = Path(path_base, "data", "raw", "electricity", "standard_data")
    path_external_data_scenario = Path(path_base, "data", "raw", "electricity", scen_folder) #test
        # test if path_external_data_scenario exists and if not set to standard scenario
    if not path_external_data_scenario.exists():
        path_external_data_scenario = Path(path_base, "data", "raw", "electricity", STANDARD_SCEN_EXTERNAL_DATA)
    print(f"Path to image output: {path_image_output}")
    assert path_image_output.is_dir()
    assert path_external_data_standard.is_dir()
    assert path_external_data_scenario.is_dir()

    years = YEAR_END - YEAR_START  + 1

    idx = pd.IndexSlice   

    ###########################################################################################################
    # Read in files #

    # 1. External Data --------------------------------------------- 

    # lifetimes of Gcap tech (original data according to van Vuuren 2006, PhD Thesis)
    gcap_lifetime = pd.read_csv(path_external_data_scenario / 'LTTechnical_dynamic.csv', index_col=['Year','DIM_1'])        
    # material compositions (generation capacity)
    composition_generation = pd.read_csv(path_external_data_scenario / 'composition_generation.csv',index_col=[0,1]).transpose()  # in gram/MW
    kilometrage = pd.read_csv(path_external_data_scenario / 'kilometrage.csv', index_col='t')  # to get region list without running storage - TODO: get regions from different source

    # 2. IMAGE/TIMER files -----------------------------------------

    # Generation capacity (stock & inflow/new) in MW peak capacity, FILES from TIMER
    gcap_data = read_mym_df(path_image_output / 'GCap.out')

    ###########################################################################################################
    # Prepare model specific variables #
    
    region_list = list(kilometrage.columns.values)   

    gcap_tech_list = list(composition_generation.loc[:,idx[2020,:]].droplevel(axis=1, level=0).columns)    #list of names of the generation technologies (workaround to retain original order)
    gcap_material_list = list(composition_generation.index.values)  #list of materials the generation technologies

    gcap_data = gcap_data.loc[~gcap_data['DIM_1'].isin([27,28])]    # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
    gcap = pd.pivot_table(gcap_data[gcap_data['time'].isin(list(range(YEAR_START,YEAR_END+1)))], index=['time','DIM_1'], values=list(range(1,TECH_GEN+1)))  #gcap as multi-index (index = years & regions (26); columns = technologies (34));  the last column in gcap_data (= totals) is now removed

    # renaming multi-index dataframe: generation capacity, based on the regions in grid_length_Hv & technologies as given
    gcap.index = pd.MultiIndex.from_product([list(range(YEAR_START,YEAR_END+1)), region_list], names=['years', 'regions'])
    gcap.columns = gcap_tech_list

    # Interpolate material intensities (dynamic content for gcap & storage technologies between 1926 to 2100, based on data files)
    index = pd.MultiIndex.from_product([list(range(YEAR_FIRST_GRID, YEAR_OUT+1)), list(composition_generation.index)])
    gcap_materials_interpol = pd.DataFrame(index=index, columns=composition_generation.columns.levels[1])

    # material intensities for gcap
    for cat in list(composition_generation.columns.levels[1]):
        gcap_materials_1st   = composition_generation.loc[:,idx[composition_generation.columns[0][0],cat]]
        gcap_materials_interpol.loc[idx[YEAR_FIRST_GRID ,:],cat] = gcap_materials_1st.to_numpy()                # set the first year (1926) values to the first available values in the dataset (for the year 2000) 
        gcap_materials_interpol.loc[idx[composition_generation.columns.levels[0].min(),:],cat] = composition_generation.loc[:, idx[composition_generation.columns.levels[0].min(),cat]].to_numpy()                # set the middle year (2000) values to the first available values in the dataset (for the year 2000) 
        gcap_materials_interpol.loc[idx[composition_generation.columns.levels[0].max(),:],cat] = composition_generation.loc[:, idx[composition_generation.columns.levels[0].max(),cat]].to_numpy()                # set the last year (2100) values to the last available values in the dataset (for the year 2050) 
        gcap_materials_interpol.loc[idx[:,:],cat] = gcap_materials_interpol.loc[idx[:,:],cat].unstack().astype('float64').interpolate().stack()

    # interpolate Gcap (technical) lifetime data
    gcap_lifetime.index = gcap_lifetime.index.set_levels(gcap_tech_list, level=1)
    gcap_lifetime = gcap_lifetime.unstack().droplevel(axis=1, level=0)
    gcap_lifetime = gcap_lifetime.reindex(list(range(YEAR_FIRST_GRID,YEAR_OUT+1)), axis=0).interpolate(limit_direction='both')

    # Calculate the historic tail to the Gcap (stock) 
    gcap_new = pd.DataFrame(index=pd.MultiIndex.from_product([range(YEAR_FIRST_GRID,YEAR_OUT+1), region_list], names=['years', 'regions']), columns=gcap.columns)
    for tech in gcap_tech_list:
        gcap_new.loc[idx[:,:],tech] = stock_tail(gcap.loc[idx[:,:],tech].unstack(level=1)).stack()


    # Bring dataframes into correct shape for the results_dict
    # I need:
    #   A. with only Types: stocks Types (GW), lifetimes Types, material intensities Types
    #   B. with also SubTypes: stocks Types (GW), lifetimes SubTypes, material intensities SubTypes, market shares SubTypes

    # A.
    # stocks: (years, regions) index and technologies as columns -> years as index and (technology, region) as columns
    gcap_stock = gcap_new.unstack(level='regions')

    # lifetimes
    df_mean = gcap_lifetime.copy()
    df_stdev = df_mean * STD_LIFETIMES_ELECTR
    df_mean.columns = [(col, 'mean') for col in df_mean.columns] # Rename columns to multi-level tuples
    df_stdev.columns = [(col, 'stdev') for col in df_stdev.columns]
    gcap_lifetime_distr = pd.concat([df_mean, df_stdev], axis=1) # Concatenate along columns

    # MIs: (years, material) index and technologies as columns -> years as index and (technology, Material) as columns
    gcap_materials_interpol.index.names = ["Year", "Material"]
    # gcap_materials_interpol = gcap_materials_interpol.loc[:, ~(gcap_materials_interpol == 0.0).all()] # delete empty columns
    gcap_types_materials = gcap_materials_interpol.unstack(level='Material')


    ###########################################################################################################
    # Prep_data File #
    
    # Conversion table for all coordinates, to be removed/adapted after input tables are fixed.
    conversion_table = {
        "gcap_stock": (["Time"], ["Type", "Region"],),
        "gcap_types_materials": (["Cohort"], ["Type", "material"],)
        # "gcap_materials_interpol": (["Cohort"], ["Type", "SubType", "material"], {"Type": ["Type", "SubType"]})
    }

    results_dict = {
            'gcap_stock': gcap_stock,
            'gcap_types_materials': gcap_types_materials,
            'gcap_lifetime_distr': gcap_lifetime_distr,
    }

    prep_data = create_prep_data(results_dict, conversion_table, unit_mapping)

    prep_data["stocks"] = prism.Q_(prep_data["stocks"], "MW")
    prep_data["material_intensities"] = prism.Q_(prep_data["material_intensities"], "g/MW")
    prep_data["set_unit_flexible"] = prism.U_(prep_data["stocks"]) # prism.U_ gives the unit back
    # set_unit_flexible is needed by the model to deal with the fact the in the beginning of the model it doesn't know th data yet and needs to work with a placeholder/flexible unit (see model.py) 

    return prep_data



###########################################################################################################
###########################################################################################################
#%% 1) Grid 
###########################################################################################################
###########################################################################################################

def get_preprocessing_data_grid(path_base: str, SCEN, VARIANT): #, climate_policy_config: dict, circular_economy_config: dict

    scen_folder = SCEN + "_" + VARIANT
    path_image_output = Path(path_base, "data", "raw", "image", scen_folder, "EnergyServices")
    path_external_data_standard = Path(path_base, "data", "raw", "electricity", "standard_data")
    path_external_data_scenario = Path(path_base, "data", "raw", "electricity", scen_folder)
    
    # test if path_external_data_scenario exists and if not set to standard scenario
    if not path_external_data_scenario.exists():
        path_external_data_scenario = Path(path_base, "data", "raw", "electricity", STANDARD_SCEN_EXTERNAL_DATA)

    scen_BL_folder = SCEN + "_M_CP"  # baseline scenario
    path_image_output_BL = Path(path_base, "data", "raw", "image", scen_BL_folder, "EnergyServices")
    # TODO: check if this is necessary (shouldn't historical periode anyway be the same for all scenarios?)
    # + if it is, should the baseline scenario be given as a parameter or can it be inferred from the scenario name?

    assert path_image_output.is_dir()
    assert path_external_data_standard.is_dir()
    assert path_external_data_scenario.is_dir()

    years = YEAR_END - YEAR_START  + 1

    idx = pd.IndexSlice   

    ###########################################################################################################
    # Read in files #

    # 1. External Data --------------------------------------------- 

    grid_length_Hv = pd.read_csv(path_external_data_standard /'grid_length_Hv.csv', index_col=0, names=None).transpose()    # lenght of the High-voltage (Hv) lines in the grid, based on Open Street Map (OSM) analysis (km)
    ratio_Hv = pd.read_csv(path_external_data_standard / 'Hv_ratio.csv', index_col=0)                                       # Ratio between the length of Medium-voltage (Mv) and Low-voltage (Lv) lines in relation to Hv lines (km Lv /km Hv) & (km Mv/ km Hv)
    underground_ratio = pd.read_csv(path_external_data_standard / 'underground_ratio.csv', index_col=[0,1])                 # these contain the definition of the constants in the linear function used to determine the relation between income & the percentage of underground power-lines (% underground = mult * gdp/cap + add)
    grid_additions = pd.read_csv(path_external_data_standard / 'grid_additions.csv', index_col=0)                           # Transformers & substations per km of grid (Hv, Mv & Lv, in units/km)


    # dynamic or scenario-dependent data (lifetimes & material intensity)

    lifetime_grid_elements = pd.read_csv(path_external_data_scenario  / 'operational_lifetime_grid.csv', index_col=0)        # Average lifetime in years of grid elements

    # dynamic material intensity files (kg/km or kg/unit)
    materials_grid = pd.read_csv(path_external_data_scenario / 'Materials_grid_dynamic.csv', index_col=[0,1])                # Material intensity of grid lines specific material content for Hv, Mv & Lv lines, & specific for underground vs. aboveground lines. (kg/km)
    materials_grid_additions = pd.read_csv(path_external_data_scenario / 'Materials_grid_additions.csv', index_col=[0,1])    # (not part of the SA yet) Additional infrastructure required for grid connections, such as transformers & substations (material compositin in kg/unit)

    # IMAGE file: GDP per capita (US-dollar 2005, ppp), used to derive underground-aboveground ratio based on income levels
    gdp_pc = pd.read_csv(path_external_data_scenario / 'gdp_pc.csv', index_col=0)  # TODO: check why it says this is an IMAGE file (why is it .csv?)


    # 2. IMAGE/TIMER files ---------------------------------------------

    # Generation capacity (stock & inflow/new) in MW peak capacity, FILES from TIMER
    gcap_data = read_mym_df(path_image_output / 'GCap.out')
    # gcap_BL_data = read_mym_df('SSP2\\SSP2_BL\\GCap.out') # baseline scenario? TODO: what is the purpose of reading in the scneario + the baseline?
    gcap_BL_data = read_mym_df(path_image_output_BL / 'GCap.out')

    ###########################################################################################################
    # Prepare model specific variables #

    gcap_data = gcap_data.loc[~gcap_data['DIM_1'].isin([27,28])]    # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
    gcap = pd.pivot_table(gcap_data[gcap_data['time'].isin(list(range(YEAR_START,YEAR_END+1)))], index=['time','DIM_1'], values=list(range(1,TECH_GEN+1)))  #gcap as multi-index (index = years & regions (26); columns = technologies (28));  the last column in gcap_data (= totals) is now removed

    gcap_BL_data = gcap_BL_data.loc[~gcap_BL_data['DIM_1'].isin([27,28])]    # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
    gcap_BL = pd.pivot_table(gcap_BL_data[gcap_BL_data['time'].isin(list(range(YEAR_START,YEAR_END+1)))], index=['time','DIM_1'], values=list(range(1,TECH_GEN+1)))  #gcap as multi-index (index = years & regions (26); columns = technologies (28));  the last column in gcap_data (= totals) is now removed

    region_list = list(grid_length_Hv.columns.values)
    material_list = list(materials_grid.columns.values)

    # renaming multi-index dataframe: generation capacity, based on the regions in grid_length_Hv & technologies as given
    gcap_techlist = ['Solar PV', 'Solar Decentral', 'CSP', 'Wind onshore', 'Wind offshore', 'Wave', 'Hydro', 'Other Renewables', 'Geothermal', 'Hydrogen', 'Nuclear', '<EMPTY>', 'Conv. Coal', 'Conv. Oil', 'Conv. Natural Gas', 'Waste', 'IGCC', 'OGCC', 'NG CC', 'Biomass CC', 'Coal + CCS', 'Oil/Coal + CCS', 'Natural Gas + CCS', 'Biomass + CCS', 'CHP Coal', 'CHP Oil', 'CHP Natural Gas', 'CHP Biomass', 'CHP Coal + CCS', 'CHP Oil + CCS', 'CHP Natural Gas + CCS', 'CHP Biomass + CCS', 'CHP Geothermal', 'CHP Hydrogen']
    gcap.index = pd.MultiIndex.from_product([list(range(YEAR_START,YEAR_END+1)), region_list], names=['years', 'regions'])
    gcap_BL.index = pd.MultiIndex.from_product([list(range(YEAR_START,YEAR_END+1)), region_list], names=['years', 'regions'])
    gcap.columns = gcap_techlist
    gcap_BL.columns = gcap_techlist

    gdp_pc.columns = region_list
    # gdp_pc = gdp_pc.drop([1970]).drop(list(range(YEAR_END+1,YEAR_LAST+1)))


    # length calculations ----------------------------------------------------------------------------

    # only the regional total (peak) generation capacity is used as a proxy for the grid growth (BL to 2016, then BL or 450)
    gcap_BL_total = gcap_BL.sum(axis=1).unstack()
    gcap_BL_total = gcap_BL_total[region_list]               # re-order columns to the original TIMER order
    gcap_growth = gcap_BL_total / gcap_BL_total.loc[2016]    # define growth according to 2016 as base year
    gcap_total = gcap.sum(axis=1).unstack()
    gcap_total = gcap_total[region_list]                     # re-order columns to the original TIMER order
    gcap_growth.loc[2016:YEAR_END] = gcap_total.loc[2016:YEAR_END] / gcap_total.loc[2016]        # define growth according to 2016 as base year

    # in the sensitivity variant, additional growth is presumed after 2020 based on the fraction of variable renewable energy (vre) generation capacity (solar & wind)
    vre_fraction = gcap[['Solar PV', 'CSP', 'Wind onshore', 'Wind offshore']].sum(axis=1).unstack().divide(gcap.sum(axis=1).unstack())
    add_growth = vre_fraction * 1                  # 0.2 = 20% additional HV lines per doubling of vre gcap
    red_growth = (1-vre_fraction) * 0.7            # 0.2 = 20% less HV lines per doubling of baseline gcap
    add_growth.loc[list(range(1971,2020+1)),:] = 0  # pre 2020, additional HV grid growth is 0, afterwards the additional line length is gradually introduced (towards 2050)
    red_growth.loc[list(range(1971,2020+1)),:] = 0  # pre 2020, reduction of HV grid growth is 0, afterwards the line length reduction is gradually introduced (towards 2050)
    for year in range(2020,2050+1):
        add_growth.loc[year] = add_growth.loc[year] * (1/30*(year-2020)) 
        red_growth.loc[year] = red_growth.loc[year] * (1/30*(year-2020)) 

    # Hv length (in kms) is region-specific. However, we use a single ratio between the length of Hv and Mv networks, the same applies to Lv networks 
    grid_length_Mv = grid_length_Hv.mul(ratio_Hv['HV to MV'])
    grid_length_Lv = grid_length_Hv.mul(ratio_Hv['HV to LV'])

    # define grid length over time (fixed in 2016, growth according to gcap)
    grid_length_Hv_time = pd.DataFrame().reindex_like(gcap_total)
    grid_length_Mv_time = pd.DataFrame().reindex_like(gcap_total)
    grid_length_Lv_time = pd.DataFrame().reindex_like(gcap_total)

    #implement growth correction (sensitivity variant)
    if SENS_ANALYSIS == 'high_grid':
        gcap_growth_HV = gcap_growth.add(add_growth.reindex_like(gcap_growth)).subtract(red_growth.reindex_like(gcap_growth))
    else: 
        gcap_growth_HV = gcap_growth

    for year in range(YEAR_START, YEAR_END+1):
        grid_length_Hv_time.loc[year] = gcap_growth_HV.loc[year].mul(grid_length_Hv.loc['2016'])
        grid_length_Mv_time.loc[year] = gcap_growth.loc[year].mul(grid_length_Mv.loc['2016'])
        grid_length_Lv_time.loc[year] = gcap_growth.loc[year].mul(grid_length_Lv.loc['2016'])

    # define underground vs. aboveground fraction (%) based on static ratios (the Hv length is the aboveground fraction according to Open Street Maps, we add the underground fractions for 3 voltage networks)
    # Based on new insights from Kalt et al. 2021, we adjust the underground ratios downwards for non-European regions
    function_Hv_under = pd.DataFrame(index=gdp_pc.index, columns=gdp_pc.columns)
    function_Mv_under = pd.DataFrame(index=gdp_pc.index, columns=gdp_pc.columns)
    function_Lv_under = pd.DataFrame(index=gdp_pc.index, columns=gdp_pc.columns)

    for region in region_list:
        if region in ['W.Europe','C.Europe']:
            select_proxy = 'Europe'
        else:
            select_proxy = 'Other'
        #print(str(region) + ': ' + select_proxy)
        function_Hv_under[region] = gdp_pc[region] * underground_ratio.loc[idx[select_proxy,'mult'],'HV'] + underground_ratio.loc[idx[select_proxy,'add'],'HV']
        function_Mv_under[region] = gdp_pc[region] * underground_ratio.loc[idx[select_proxy,'mult'],'MV'] + underground_ratio.loc[idx[select_proxy,'add'],'MV']
        function_Lv_under[region] = gdp_pc[region] * underground_ratio.loc[idx[select_proxy,'mult'],'LV'] + underground_ratio.loc[idx[select_proxy,'add'],'LV']

    # maximize linear function at 100 & minimize at 0 (%)
    function_Hv_under = function_Hv_under.apply(lambda x: [y if y >= 0 else 0 for y in x])                     
    function_Hv_under = function_Hv_under.apply(lambda x: [y if y <= 100 else 100 for y in x])  
    function_Mv_under = function_Mv_under.apply(lambda x: [y if y >= 0 else 0 for y in x])  
    function_Mv_under = function_Mv_under.apply(lambda x: [y if y <= 100 else 100 for y in x])  
    function_Lv_under = function_Lv_under.apply(lambda x: [y if y >= 0 else 0 for y in x])  
    function_Lv_under = function_Lv_under.apply(lambda x: [y if y <= 100 else 100 for y in x])

    # MIND! the HV lines found in OSM (+national sources) are considered as the total of the aboveground line length + the underground line length
    grid_length_Hv_total = grid_length_Hv_time                                      # assuming the length from OSM IS the abovegrond fraction
    grid_length_Hv_above = grid_length_Hv_total * (1 - (function_Hv_under/100)) 
    grid_length_Hv_under = grid_length_Hv_total * function_Hv_under/100

    # out for main text figure 2
    grid_length_HV_out_a = pd.concat([grid_length_Hv_above], keys=['aboveground'], names=['type']) 
    grid_length_HV_out_u = pd.concat([grid_length_Hv_under], keys=['underground'], names=['type']) 
    grid_length_HV_out   = pd.concat([grid_length_HV_out_a, grid_length_HV_out_u])
    grid_length_HV_out.to_csv(path_base / 'imagematerials' / 'electricity' / 'out_test' / 'grid_length_HV_km.csv') # in km

    grid_length_Mv_above = grid_length_Mv_time * (1 - function_Mv_under/100)
    grid_length_Mv_under = grid_length_Mv_time * function_Mv_under/100
    grid_length_Mv_total = grid_length_Mv_above + grid_length_Mv_under

    grid_length_Lv_above = grid_length_Lv_time * (1 - function_Lv_under/100)
    grid_length_Lv_under = grid_length_Lv_time * function_Lv_under/100
    grid_length_Lv_total = grid_length_Lv_above + grid_length_Lv_under

    grid_subst_Hv = grid_length_Hv_total.mul(grid_additions.loc['Substations','HV'])        # number of substations on HV network
    grid_subst_Mv = grid_length_Mv_total.mul(grid_additions.loc['Substations','MV'])        # # of substations
    grid_subst_Lv = grid_length_Lv_total.mul(grid_additions.loc['Substations','LV'])        # # of substations
    grid_trans_Hv = grid_length_Hv_total.mul(grid_additions.loc['Transformers','HV'])       # number of transformers on the HV network
    grid_trans_Mv = grid_length_Mv_total.mul(grid_additions.loc['Transformers','MV'])       # # of transformers
    grid_trans_Lv = grid_length_Lv_total.mul(grid_additions.loc['Transformers','LV'])       # # of transformers


    ##################
    # Interpolations #
    ##################

    # Interpolate material intensities (dynamic content from 1926 to 2100, based on data files)
    index                             = pd.MultiIndex.from_product([list(range(YEAR_FIRST_GRID,YEAR_END+1)), list(materials_grid.index.levels[1])])
    materials_grid_interpol           = pd.DataFrame(index=index, columns=materials_grid.columns)
    materials_grid_additions_interpol = pd.DataFrame(index=pd.MultiIndex.from_product([list(range(YEAR_FIRST_GRID,YEAR_END+1)), list(materials_grid_additions.index.levels[1])]), columns=materials_grid_additions.columns)

    for cat in list(materials_grid.index.levels[1]):
        materials_grid_1st   = materials_grid.loc[idx[materials_grid.index[0][0], cat],:]
        materials_grid_interpol.loc[idx[YEAR_FIRST_GRID ,cat],:] = materials_grid_1st                # set the first year (1926) values to the first available values in the dataset (for the year 2000) 
        materials_grid_interpol.loc[idx[materials_grid.index.levels[0].min(),cat],:] = materials_grid.loc[idx[materials_grid.index.levels[0].min(),cat],:]                # set the middle year (2000) values to the first available values in the dataset (for the year 2000) 
        materials_grid_interpol.loc[idx[materials_grid.index.levels[0].max(),cat],:] = materials_grid.loc[idx[materials_grid.index.levels[0].max(),cat],:]                # set the last year (2100) values to the last available values in the dataset (for the year 2050) 
        materials_grid_interpol.loc[idx[:,cat],:] = materials_grid_interpol.loc[idx[:,cat],:].astype('float32').reindex(list(range(YEAR_FIRST_GRID,YEAR_END+1)), level=0).interpolate()

    for cat in list(materials_grid_additions.index.levels[1]):
        materials_grid_additions_1st   = materials_grid_additions.loc[idx[materials_grid_additions.index[0][0], cat],:]
        materials_grid_additions_interpol.loc[idx[YEAR_FIRST_GRID ,cat],:] = materials_grid_additions_1st          # set the first year (1926) values to the first available values in the dataset (for the year 2000) 
        materials_grid_additions_interpol.loc[idx[materials_grid_additions.index.levels[0].min(),cat],:] = materials_grid_additions.loc[idx[materials_grid_additions.index.levels[0].min(),cat],:]                # set the middle year (2000) values to the first available values in the dataset (for the year 2000) 
        materials_grid_additions_interpol.loc[idx[materials_grid_additions.index.levels[0].max(),cat],:] = materials_grid_additions.loc[idx[materials_grid_additions.index.levels[0].max(),cat],:]                # set the last year (2100) values to the last available values in the dataset (for the year 2050) 
        materials_grid_additions_interpol.loc[idx[:,cat],:] = materials_grid_additions_interpol.loc[idx[:,cat],:].astype('float32').reindex(list(range(YEAR_FIRST_GRID,YEAR_END+1)), level=0).interpolate()

    # call the stock_tail function on all lines, substations & transformers, to add historic stock tail between 1926 & 1971
    grid_length_Hv_above_new = stock_tail(grid_length_Hv_above) # km
    grid_length_Mv_above_new = stock_tail(grid_length_Mv_above) # km
    grid_length_Lv_above_new = stock_tail(grid_length_Lv_above) # km
    grid_length_Hv_under_new = stock_tail(grid_length_Hv_under) # km 
    grid_length_Mv_under_new = stock_tail(grid_length_Mv_under) # km
    grid_length_Lv_under_new = stock_tail(grid_length_Lv_under) # km
    grid_subst_Hv_new = stock_tail(grid_subst_Hv)               # units
    grid_subst_Mv_new = stock_tail(grid_subst_Mv)               # units
    grid_subst_Lv_new = stock_tail(grid_subst_Lv)               # units
    grid_trans_Hv_new = stock_tail(grid_trans_Hv)               # units
    grid_trans_Mv_new = stock_tail(grid_trans_Mv)               # units
    grid_trans_Lv_new = stock_tail(grid_trans_Lv)               # units


    #############
    # Lifetimes #
    #############

    # data only for lines, substations and transformer -> bring in knowledge_graph format: HV - Lines, MV - Lines, LV - Lines, HV - Transformers, etc.
    expanded_data = {}
    for typ in ["Lines", "Transformers", "Substations"]:
        for level in ["HV", "MV", "LV"]:
            new_col = f"{level} - {typ}"
            expanded_data[new_col] = lifetime_grid_elements[typ]
    lifetime_grid_elements = pd.DataFrame(expanded_data, index=lifetime_grid_elements.index)
    lifetime_grid_elements.rename_axis('Year', inplace=True)

    # no differentiation between HV, MV & LV lines as well as between aboveground and belowground
    # Types: lines, transformers, substations
    lifetime_grid_elements.loc[YEAR_FIRST_GRID,:]  = lifetime_grid_elements.loc[lifetime_grid_elements.first_valid_index(),:]
    lifetime_grid_elements.loc[YEAR_OUT,:]         = lifetime_grid_elements.loc[lifetime_grid_elements.last_valid_index(),:]
    lifetime_grid_elements                         = lifetime_grid_elements.reindex(list(range(YEAR_FIRST_GRID, YEAR_OUT+1))).interpolate()
    # TODO: check why lifetime for lines is interpoltaed from 2020 - 40yrs to 2050 - 48 yrs and then back to 2060 - 40 yrs -> should stay at 48 yrs?
    df_mean = lifetime_grid_elements.copy()
    df_stdev = df_mean * STD_LIFETIMES_ELECTR
    df_mean.columns = [(col, 'mean') for col in df_mean.columns] # Rename columns to multi-level tuples
    df_stdev.columns = [(col, 'stdev') for col in df_stdev.columns]
    lifetime_grid_distr = pd.concat([df_mean, df_stdev], axis=1) # Concatenate along columns


    # Materials

    materials_grid_additions_kgperunit            = materials_grid_additions_interpol.copy()
    # material intensities: (years, tech. type) index and materials as columns -> years as index and (tech. type, materials) as columns
    materials_grid_additions_kgperunit.index.names = ["Year", "Type"]
    materials_grid_additions_kgperunit             = materials_grid_additions_kgperunit.unstack(level='Type')   # bring tech. type from row index to column header
    materials_grid_additions_kgperunit.columns     = materials_grid_additions_kgperunit.columns.swaplevel(0, 1) # Swap the levels of the MultiIndex columns
    materials_grid_additions_kgperunit             = materials_grid_additions_kgperunit.sort_index(axis=1)
    # rename columns to match knowledge graph
    new_level_0 = [col.replace(' ', ' - ', 1) for col in materials_grid_additions_kgperunit.columns.get_level_values(0)]
    new_columns = pd.MultiIndex.from_arrays([
        new_level_0,
        materials_grid_additions_kgperunit.columns.get_level_values(1)
    ], names=materials_grid_additions_kgperunit.columns.names)
    materials_grid_additions_kgperunit.columns = new_columns

    # Grid MIs ---
    materials_grid_kgperkm              = materials_grid_interpol.copy() # copy the interpolated material intensities
    materials_grid_kgperkm.index.names  = ["Year", "Type"]
    materials_grid_kgperkm              = materials_grid_kgperkm.unstack(level='Type') # bring tech. type from row index to column header
    materials_grid_kgperkm.columns      = materials_grid_kgperkm.columns.swaplevel(0, 1) # Swap the levels of the MultiIndex columns
    materials_grid_kgperkm              = materials_grid_kgperkm.sort_index(axis=1)
    # rename columns to match knowledge graph
    new_level_0 = [col.replace(' ', ' - Lines - ', 1) for col in materials_grid_kgperkm.columns.get_level_values(0)]
    new_columns = pd.MultiIndex.from_arrays([
        new_level_0,
        materials_grid_kgperkm.columns.get_level_values(1)
    ], names=materials_grid_kgperkm.columns.names)
    materials_grid_kgperkm.columns = new_columns

    grid_dict_add = dict({
        'HV - Substations':         grid_subst_Hv_new,
        'HV - Transformers':        grid_trans_Hv_new,
        'MV - Substations':         grid_subst_Mv_new,
        'MV - Transformers':        grid_trans_Mv_new,
        'LV - Substations':         grid_subst_Lv_new,
        'LV - Transformers':        grid_trans_Lv_new
    })

    grid_dict_lines = dict({
            'HV - Lines - Overhead':    grid_length_Hv_above_new,
            'HV - Lines - Underground': grid_length_Hv_under_new,
            'MV - Lines - Overhead':    grid_length_Mv_above_new,
            'MV - Lines - Underground': grid_length_Mv_under_new,
            'LV - Lines - Overhead':    grid_length_Lv_above_new,
            'LV - Lines - Underground': grid_length_Lv_under_new,
        })
        
    grid_stock_lines = pd.concat(grid_dict_lines, axis=1) # Concatenate with keys to create MultiIndex ('Name', 'Region')
    grid_stock_lines = grid_stock_lines.sort_index(axis=1)

    grid_stock_add = pd.concat(grid_dict_add, axis=1) # Concatenate with keys to create MultiIndex ('Name', 'Region')
    grid_stock_add = grid_stock_add.sort_index(axis=1)


    ###########################################################################################################
    # Prep_data File #

    conversion_table = {
    "grid_stock_lines": (["Time"], ["Type", "Region"],),
    "materials_grid_kgperkm": (["Cohort"], ["Type", "material"],),
    "grid_stock_add": (["Time"], ["Type", "Region"],),
    "materials_grid_add_kgperunit": (["Cohort"], ["Type", "material"],),
    "grid_stock": (["Time"], ["Type", "Region"],), # TODO: delete
    "materials_grid_combined_kgperkm": (["Cohort"], ["Type", "material"],) # TODO: delete
    }

    results_dict_lines = {
            'grid_stock_lines': grid_stock_lines,
            'materials_grid_kgperkm': materials_grid_kgperkm,
            'lifetime_grid_distr': lifetime_grid_distr,
    }
    results_dict_add = {
            'grid_stock_add': grid_stock_add,
            'materials_grid_add_kgperunit': materials_grid_additions_kgperunit,
            'lifetime_grid_distr': lifetime_grid_distr,
    }

    prep_data_lines = create_prep_data(results_dict_lines, conversion_table, unit_mapping)
    prep_data_add = create_prep_data(results_dict_add, conversion_table, unit_mapping)

    prep_data_lines["stocks"] = prism.Q_(prep_data_lines["stocks"], "km")
    prep_data_lines["material_intensities"] = prism.Q_(prep_data_lines["material_intensities"], "kg/km")
    prep_data_lines["set_unit_flexible"] = prism.U_(prep_data_lines["stocks"]) # prism.U_ gives the unit back
    # set_unit_flexible is needed by the model to deal with the fact the in the beginning of the model it doesn't know th data yet and needs to work with a placeholder/flexible unit (see model.py) 

    prep_data_add["stocks"] = prism.Q_(prep_data_add["stocks"], "count")
    prep_data_add["material_intensities"] = prism.Q_(prep_data_add["material_intensities"], "kg/count")
    prep_data_add["set_unit_flexible"] = prism.U_(prep_data_add["stocks"]) # prism.U_ gives the unit back

    return prep_data_lines, prep_data_add





