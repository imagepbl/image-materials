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

import prism
from imagematerials.distribution import ALL_DISTRIBUTIONS, NAME_TO_DIST
from imagematerials.read_mym import read_mym_df
from imagematerials.util import dataset_to_array, pandas_to_xarray, convert_lifetime
from imagematerials.model import GenericMainModel, GenericStocks, SharesInflowStocks, Maintenance, GenericMaterials, MaterialIntensities
from imagematerials.factory import ModelFactory, Sector
from imagematerials.concepts import create_electricity_graph, create_region_graph
from imagematerials.electricity.utils import MNLogit, stock_tail, create_prep_data, stock_share_calc,add_historic_stock, interpolate_xr, flexible_plot_1panel

from imagematerials.constants import (
    IMAGE_REGIONS,
)

from imagematerials.electricity.constants import (
    YEAR_FIRST,
    YEAR_FIRST_GRID,
    YEAR_SWITCH,
    STANDARD_SCEN_EXTERNAL_DATA,
    SENS_ANALYSIS,
    REGIONS,
    TECH_GEN,
    EPG_TECHNOLOGIES,
    EPG_TECHNOLOGIES_VRE,
    STD_LIFETIMES_ELECTR,
    MEGA_TO_TERA,
    PKMS_TO_VKMS,
    TONNES_TO_KGS,
    LOAD_FACTOR,
    BEV_CAPACITY_CURRENT,
    PHEV_CAPACITY_CURRENT,
    unit_mapping,
    DICT_GENTECH_TO_CATEGORY,
    DICT_GENTECH_STYLES,
    DICT_STOR_STYLES,
    DICT_GEN_CATEGORY_COLORS,
    DICT_MATERIALS_COLORS,
    DICT_GRID_COLORS,
    DICT_GRID_STYLES_1,
    DICT_GRID_STYLES_2,
    DICT_STOR_CATEGORY_COLORS_SEBASTIAAN
)

from imagematerials.electricity.electr_external_data import (
    df_iea_cu_aps,
    df_iea_cu_nzs,
    df_iea_co_aps,
    df_iea_mn_aps,
    df_iea_ni_aps
)

SCEN = "SSP2"
VARIANT = "VLHO"
# VARIANT = "M_CP"
# VARIANT = "BL"
# VARIANT = "450"
# Define paths ----------------------------------------------------------------------
#YOUR_DIR = "C:\\Users\\Admin\\surfdrive\\Projects\\IRP\\GRO23\\Modelling\\2060\\ELMA"   # Change the running directory here
# os.chdir(YOUR_DIR)
scen_folder = SCEN + "_" + VARIANT
# path_base = Path().resolve() # TODO absolute path of file "preprocessing.py" ? current solution can differ depending on IDE used (?) 
path_current = Path().resolve()
path_base = path_current.parent.parent # base path of the project -> image-materials

path_image_output = Path(path_base, "data", "raw", "image", scen_folder, "EnergyServices")
# path_image_output = Path(path_base, "data", "raw", "image", scen_folder)

# TEST---
path_image_output_SSP2_BL = Path(path_base, "data", "raw", "image", "SSP2_BL")
path_image_output_SSP2_450 = Path(path_base, "data", "raw", "image", "SSP2_450")
path_image_output_SSP2_M_CP = Path(path_base, "data", "raw", "image", "SSP2_M_CP", "EnergyServices")
path_image_output_SSP2_VLHO = Path(path_base, "data", "raw", "image", "SSP2_VLHO", "EnergyServices")
#-----------
path_external_data_standard = Path(path_base, "data", "raw", "electricity", "standard_data")
path_external_data_scenario = Path(path_base, "data", "raw", "electricity", scen_folder)
# test if path_external_data_scenario exists and if not set to standard scenario
if not path_external_data_scenario.exists():
    path_external_data_scenario = Path(path_base, "data", "raw", "electricity", STANDARD_SCEN_EXTERNAL_DATA)
print(f"Path to image output: {path_image_output}")

assert path_image_output.is_dir()
assert path_external_data_standard.is_dir()
assert path_external_data_scenario.is_dir()

# create the folder out_test if it does not exist
if not (path_base / 'imagematerials' / 'electricity' / 'out_test').is_dir():
    (path_base / 'imagematerials' / 'electricity' / 'out_test').mkdir(parents=True)
    
    
idx = pd.IndexSlice             # needed for slicing multi-index

    

    

###########################################################################################################
###########################################################################################################
#%% 3) TRANSMISSION GRID 
###########################################################################################################
###########################################################################################################

def get_preprocessing_data_grid(base_dir: str, SCEN, VARIANT, YEAR_START, YEAR_END, YEAR_OUT): #, climate_policy_config: dict, circular_economy_config: dict

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
    ratio_hv = pd.read_csv(path_external_data_standard / 'Hv_ratio.csv', index_col=0)                                        # Ratio between the length of Medium-voltage (Mv) and Low-voltage (Lv) lines in relation to Hv lines (km Lv /km Hv) & (km Mv/ km Hv)
    underground_ratio = pd.read_csv(path_external_data_standard / 'underground_ratio.csv', index_col=[0,1])           # these contain the definition of the constants in the linear function used to determine the relation between income & the percentage of underground power-lines (% underground = mult * gdp/cap + add)
    grid_additions = pd.read_csv(path_external_data_standard / 'grid_additions.csv', index_col=0)                            # Transformers & substations per km of grid (Hv, Mv & Lv, in units/km)


    # dynamic or scenario-dependent data (lifetimes & material intensity)

    lifetime_grid_elements = pd.read_csv(path_external_data_scenario  / 'operational_lifetime_grid.csv', index_col=0)         # Average lifetime in years of grid elements

    # dynamic material intensity files (kg/km or kg/unit)
    materials_grid = pd.read_csv(path_external_data_scenario / 'Materials_grid_dynamic.csv', index_col=[0,1])                # Material intensity of grid lines specific material content for Hv, Mv & Lv lines, & specific for underground vs. aboveground lines. (kg/km)
    materials_grid_additions = pd.read_csv(path_external_data_scenario / 'Materials_grid_additions.csv', index_col=[0,1])    # (not part of the SA yet) Additional infrastructure required for grid connections, such as transformers & substations (material compositin in kg/unit)

    # IMAGE file: GDP per capita (US-dollar 2005, ppp), used to derive underground-aboveground ratio based on income levels
    gdp_pc = pd.read_csv(path_external_data_scenario / 'gdp_pc.csv', index_col=0)  # TODO: check why it says this is an IMAGE file (why is it .csv?)


    # 2. IMAGE/TIMER files ---------------------------------------------

    # Generation capacity (stock & inflow/new) in MW peak capacity, FILES from TIMER
    gcap_data = read_mym_df(path_image_output / 'Gcap.out')
    # gcap_BL_data = read_mym_df('SSP2\\SSP2_BL\\Gcap.out') # baseline scenario? TODO: what is the purpose of reading in the scneario + the baseline?
    gcap_BL_data = read_mym_df(path_image_output_BL / 'Gcap.out')

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
    grid_length_Mv = grid_length_Hv.mul(ratio_hv['HV to MV'])
    grid_length_Lv = grid_length_Hv.mul(ratio_hv['HV to LV'])

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
    # grid_length_HV_out.to_csv(path_base / 'imagematerials' / 'electricity' / 'out_test' / 'grid_length_HV_km.csv') # in km

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
    grid_length_Hv_above_new = stock_tail(grid_length_Hv_above, YEAR_OUT) # km
    grid_length_Mv_above_new = stock_tail(grid_length_Mv_above, YEAR_OUT) # km
    grid_length_Lv_above_new = stock_tail(grid_length_Lv_above, YEAR_OUT) # km
    grid_length_Hv_under_new = stock_tail(grid_length_Hv_under, YEAR_OUT) # km 
    grid_length_Mv_under_new = stock_tail(grid_length_Mv_under, YEAR_OUT) # km
    grid_length_Lv_under_new = stock_tail(grid_length_Lv_under, YEAR_OUT) # km
    grid_subst_Hv_new = stock_tail(grid_subst_Hv, YEAR_OUT)               # units
    grid_subst_Mv_new = stock_tail(grid_subst_Mv, YEAR_OUT)               # units
    grid_subst_Lv_new = stock_tail(grid_subst_Lv, YEAR_OUT)               # units
    grid_trans_Hv_new = stock_tail(grid_trans_Hv, YEAR_OUT)               # units
    grid_trans_Mv_new = stock_tail(grid_trans_Mv, YEAR_OUT)               # units
    grid_trans_Lv_new = stock_tail(grid_trans_Lv, YEAR_OUT)               # units


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

    # harmonize units of MI -> then materials_grid and materials_grid_additions can be merged in one dataframe
    # NOT POSSIBLE - separate stock modelling necessary

    # Grid Additions MIs ---
    # # for substations and transformer we multiply the MI (kg/unit of substation or transformer) with the number of substations or transformers per kilometer of grid length
    # materials_grid_additions_kgperkm             = materials_grid_additions_to_kgperkm(materials_grid_additions_interpol, grid_additions)
    # # material intensities: (years, tech. type) index and materials as columns -> years as index and (tech. type, materials) as columns
    # materials_grid_additions_kgperkm.index.names = ["Year", "Type"]
    # materials_grid_additions_kgperkm             = materials_grid_additions_kgperkm.unstack(level='Type')   # bring tech. type from row index to column header
    # materials_grid_additions_kgperkm.columns     = materials_grid_additions_kgperkm.columns.swaplevel(0, 1) # Swap the levels of the MultiIndex columns
    # materials_grid_additions_kgperkm             = materials_grid_additions_kgperkm.sort_index(axis=1)
    # # rename columns to match knowledge graph
    # new_level_0 = [col.replace(' ', ' - ', 1) for col in materials_grid_additions_kgperkm.columns.get_level_values(0)]
    # new_columns = pd.MultiIndex.from_arrays([
    #     new_level_0,
    #     materials_grid_additions_kgperkm.columns.get_level_values(1)
    # ], names=materials_grid_additions_kgperkm.columns.names)
    # materials_grid_additions_kgperkm.columns = new_columns

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

    return prep_data_lines, prep_data_add



#----------------------------------------------------------------------------------------------------------
###########################################################################################################
#%%% 3.1) Read in files
###########################################################################################################
#----------------------------------------------------------------------------------------------------------

# TODO: remove these numbers from preprocessing
YEAR_START = 1971   # start year of the simulation period
YEAR_END = 2100     # end year of the calculations
YEAR_OUT = 2100     # year of output generation = last year of reporting

# 1. External Data --------------------------------------------- 

# lenght of the High-voltage (Hv) lines in the grid, based on Open Street Map (OSM) analysis (km)
grid_length_hv_data    = pd.read_csv(path_external_data_standard /'grid_length_Hv.csv', index_col=0, names=None).transpose()    
# Ratio between the length of Medium-voltage (Mv) and Low-voltage (Lv) lines in relation to Hv lines (km Lv /km Hv) & (km Mv/ km Hv)
ratio_hv               = pd.read_csv(path_external_data_standard / 'grid_ratio_voltage_lines.csv', index_col=0)
# Regression constants in the linear function used to determine the relation between income & the percentage of underground power-lines (% underground = mult * gdp/cap + add)
ratio_underground      = pd.read_csv(path_external_data_standard / 'grid_ratio_underground.csv', index_col=[0,1])
# Transformers & substations per km of grid (Hv, Mv & Lv, in units/km)
ratio_grid_additions   = pd.read_csv(path_external_data_standard / 'grid_additions.csv', index_col=0)


# dynamic or scenario-dependent data (lifetimes & material intensity)

# Average lifetime in years of grid elements
grid_lifetime_data     = pd.read_csv(path_external_data_scenario  / 'operational_lifetime_grid.csv', index_col=0)

# Material intensity of grid lines (Hv, Mv & Lv; specific for underground vs. aboveground lines) (kg/km)
materials_lines_data        = pd.read_csv(path_external_data_scenario / 'Materials_grid_dynamic.csv', index_col=[0,1])
# Material Intensity for additional infrastructure required for grid connections, such as transformers & substations (kg/unit)
materials_additions_data    = pd.read_csv(path_external_data_scenario / 'Materials_grid_additions.csv', index_col=[0,1])


# 2. IMAGE/TIMER files ====================================================================================

# Generation capacity (stock & inflow/new) in MW peak capacity, FILES from TIMER
gcap_data: pd.DataFrame     = read_mym_df(path_image_output / 'Gcap.out')

# GDP per capita (US-dollar 2005, ppp), used to derive underground-aboveground ratio based on income levels
gdp_pc_data: pd.DataFrame   = read_mym_df(Path(path_base, "data", "raw", "image", scen_folder, "Socioeconomic", "gdp_pc.scn"))


###########################################################################################################
#%%% Transform to xarray #
###########################################################################################################

knowledge_graph_region = create_region_graph()
knowledge_graph_electr = create_electricity_graph()

# # Grid Lines -----------
grid_length_hv_data.columns.name = None # drop column name ('km')
# line_types = ["hv_lines_overhead","hv_lines_underground","mv_lines_overhead","mv_lines_underground", "lv_lines_overhead","lv_lines_underground"]  
line_types = ["HV - Lines - Overhead","HV - Lines - Underground","MV - Lines - Overhead","MV - Lines - Underground", "LV - Lines - Overhead","LV - Lines - Underground"]
years = np.arange(YEAR_START,YEAR_OUT+1,1)
grid_lines = xr.DataArray(
    np.full((len(years), len(grid_length_hv_data.columns), len(line_types)), np.nan),
    dims=("Time","Region","Type"),
    coords={
        "Time": years.astype(int),
        "Region": grid_length_hv_data.columns.astype(str),
        "Type": line_types
    },
    name="GridLength"
)
# fill all years with the data read in for HV in 2016, as they will be later scaled by multiplying with a growth factor relative to the year 2016
grid_lines.loc[{"Time":slice(YEAR_START, YEAR_END), "Type":"HV - Lines - Overhead"}] = grid_length_hv_data.values.reshape(-1)
grid_lines = prism.Q_(grid_lines, "km")
grid_lines = knowledge_graph_region.rebroadcast_xarray(grid_lines, output_coords=IMAGE_REGIONS, dim="Region") # convert region names to the standard names from IMAGE



# # Grid Additions -----------
additions_types = ["HV - Transformers","HV - Substations","MV - Transformers","MV - Substations", "LV - Transformers","LV - Substations"]
years = np.arange(YEAR_START,YEAR_OUT+1,1)
grid_additions = xr.DataArray(
    np.full((len(years), len(IMAGE_REGIONS), len(additions_types)), np.nan),
    dims=("Time","Region","Type"),
    coords={
        "Time": years.astype(int),
        "Region": IMAGE_REGIONS,
        "Type": additions_types
    },
    name="GridAdditions"
)
grid_additions = prism.Q_(grid_additions, "count")

# # Lifetimes -------
# data only for lines, substations and transformer -> bring in knowledge_graph format: HV - Lines, MV - Lines, LV - Lines, HV - Transformers, etc.
expanded_data = {}
tech_types = []
for tech in ["Lines", "Transformers", "Substations"]:
    for level in ["HV", "MV", "LV"]:
        new_col = f"{level} - {tech}"
        tech_types.append(new_col)
        expanded_data[new_col] = grid_lifetime_data[tech]
grid_lifetime_data = pd.DataFrame(expanded_data, index=grid_lifetime_data.index)

values = grid_lifetime_data.to_numpy(dtype=float)
# Create coordinates
times = grid_lifetime_data.index.to_numpy()
scipy_params = ["mean", "stdev"]
# Build full array: shape (ScipyParam, Time, Type)
data_array = np.stack([values, np.full_like(values, np.nan)], axis=0)
# Create DataArray
grid_lifetime = xr.DataArray(
    data_array,
    dims=["DistributionParams", "Cohort", "Type"],
    coords={
        "DistributionParams": scipy_params,
        "Cohort": times.astype(int),
        "Type": [str(r) for r in tech_types]
    },
    name="Lifetime"
)
grid_lifetime = prism.Q_(grid_lifetime, "year")


# # Material Intensities -------

# Lines ---
materials_lines = materials_lines_data.reset_index().rename(columns={'year': 'Time', 'kg/km': 'Type'})
# turn "HV Overhead" -> "HV - Lines - Overhead"
materials_lines['Type'] = materials_lines['Type'].str.split(n=1).apply(lambda x: f"{x[0]} - Lines - {x[1]}")
# convert to xarray: material columns become a new 'materials' dimension
materials_lines = materials_lines.set_index(['Time', 'Type']).to_xarray()
materials_lines = materials_lines.to_array(dim='materials').rename("GridMaterialsLines")
materials_lines = materials_lines.transpose('Time', 'Type', 'materials') # reorder dimensions
materials_lines = prism.Q_(materials_lines, "kg/km")

# Additions ---
materials_additions = materials_additions_data.reset_index().rename(columns={'year': 'Time', 'kg/unit': 'Type'})
# turn "HV Overhead" -> "HV - Lines - Overhead"
materials_additions['Type'] = materials_additions['Type'].str.split(n=1).apply(lambda x: f"{x[0]} - {x[1]}")
# convert to xarray: material columns become a new 'materials' dimension
materials_additions = materials_additions.set_index(['Time', 'Type']).to_xarray()
materials_additions = materials_additions.to_array(dim='materials').rename("GridMaterialsAdditions")
materials_additions = materials_additions.transpose('Time', 'Type', 'materials') # reorder dimensions
materials_additions = prism.Q_(materials_additions, "kg/count")


# # Gcap ------
gcap_data = gcap_data.loc[~gcap_data['DIM_1'].isin([27,28])]  # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
gcap_data = gcap_data.loc[gcap_data['time'].isin(range(YEAR_START, YEAR_END + 1)), ['time', 'DIM_1', *range(1, len(EPG_TECHNOLOGIES) + 1)]]  # only keep relevant years and technology columns
# Extract coordinate labels
years = sorted(gcap_data['time'].unique())
regions = sorted(gcap_data['DIM_1'].unique())
techs = list(range(1, len(EPG_TECHNOLOGIES)+1))
# Convert to 3D array: (Year, Region, Tech)
data_array = gcap_data[techs].to_numpy().reshape(len(years), len(regions), len(techs))
# Build xarray DataArray
gcap = xr.DataArray(
    data_array,
    dims=('Time', 'Region', 'Type'),
    coords={
        'Time': years,
        'Region': [str(r) for r in regions],
        'Type': [str(r) for r in techs]
    },
    name='GCap'
)
gcap = prism.Q_(gcap, "MW")
gcap = knowledge_graph_region.rebroadcast_xarray(gcap, output_coords=IMAGE_REGIONS, dim="Region") 
gcap = knowledge_graph_electr.rebroadcast_xarray(gcap, output_coords=EPG_TECHNOLOGIES, dim="Type")
gcap = gcap.assign_coords(Type=np.array(gcap.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

# # GDP ------
gdp_pc_data = gdp_pc_data.iloc[:, :len(IMAGE_REGIONS)] # exclude empty column (27) and totals (28)
# Extract coordinate labels
years = sorted(gdp_pc_data.index)
regions = sorted(gdp_pc_data.columns)
# Convert to DD array: (Year, Region, Tech)
data_array = gdp_pc_data.to_numpy() #.reshape(len(years), len(regions))
# Build xarray DataArray
gdp_pc = xr.DataArray(
    data_array,
    dims=('Time', 'Region'),
    coords={
        'Time': years,
        'Region': [str(r) for r in regions]
    },
    name='GDPPerCapita'
)
gdp_pc = prism.Q_(gdp_pc, "USD/person")
gdp_pc = knowledge_graph_region.rebroadcast_xarray(gdp_pc, output_coords=IMAGE_REGIONS, dim="Region") 


# # line ratios ------
# region specific ratio between the length of Hv and Mv and LV networks
ds_ratio_hv = ratio_hv.to_xarray()
ds_ratio_hv = ds_ratio_hv.swap_dims({'km/km': 'Region'}) # change the dimension
ds_ratio_hv = ds_ratio_hv.rename({'km/km': 'Region'}) # rename the coordinate
ds_ratio_hv = ds_ratio_hv.set_index(Region='Region')
ds_ratio_hv["HV to MV"] = prism.Q_(ds_ratio_hv["HV to MV"], "km/km")
ds_ratio_hv["HV to LV"] = prism.Q_(ds_ratio_hv["HV to LV"], "km/km")
ds_ratio_hv = xr.Dataset({ # rebroadcast both xarrays in dataset to IMAGE regions
    name: knowledge_graph_region.rebroadcast_xarray(da, output_coords=IMAGE_REGIONS, dim="Region")
    for name, da in ds_ratio_hv.data_vars.items()
}, attrs=ds_ratio_hv.attrs)


###########################################################################################################
#%%% Calculate variables #
###########################################################################################################

# 1. calculate growth factors for the grid lines -------------------------------------------------------------

# regional total (peak) generation capacity is used as a proxy for the grid growth
gcap_total = gcap.sum(dim='Type')
gcap_growth = gcap_total / gcap_total.loc[2016]        # define growth according to 2016 as base year

# copy growth factor for all voltage levels (overhead)
grid_growth = gcap_growth.expand_dims(Type=["HV - Lines - Overhead","MV - Lines - Overhead","LV - Lines - Overhead"]).copy()
# add coordinates for underground lines to match grid length dataarray (set to NaN, as underground lines are later calculated based on aboveground lines & fixed ratios)
grid_growth_expanded = grid_growth.broadcast_like(grid_lines).copy().rename("GridGrowthFactor")

# for HV lines: additional growth is presumed after 2020 based on the fraction of variable renewable energy (vre) generation capacity (solar & wind) (used to be only in the sensitivity variant, but now used in the base case as well)
vre_fraction = gcap.sel(Type=EPG_TECHNOLOGIES_VRE).sum(dim='Type') / gcap_total
# Compute additional/reduced growth
add_growth = vre_fraction * 1             # if value is e.g. 0.2 = 20% additional HV lines per doubling of vre gcap
red_growth = (1 - vre_fraction) * 0.7     
add_growth = add_growth.where(add_growth.Time >= 2020, 0)  # Set pre-2020 values to 0
red_growth = red_growth.where(red_growth.Time >= 2020, 0)
# Create a ramp factor (line length addition/reduction is gradually introduced from 2020 towards 2050)
ramp_factor = xr.DataArray(
    np.clip((add_growth.Time - 2020) / 30, 0, 1),
    coords={"Time": add_growth.Time},
    dims=["Time"]
)
add_growth = add_growth * ramp_factor # Apply ramp factor
red_growth = red_growth * ramp_factor
grid_growth_expanded.loc[{"Type": "HV - Lines - Overhead"}] = grid_growth_expanded.loc[{"Type": "HV - Lines - Overhead"}] + add_growth - red_growth


# 2. calculate grid lengths (overhead) with ratios & growth factors -------------------------------------------------------------

# calculate extend of MV and LV networks in 2016 based on Hv network and fixed ratios
grid_lines.loc[{"Type": "MV - Lines - Overhead"}] = grid_lines.loc[{"Type": "HV - Lines - Overhead"}] * ds_ratio_hv["HV to MV"]
grid_lines.loc[{"Type": "LV - Lines - Overhead"}] = grid_lines.loc[{"Type": "HV - Lines - Overhead"}] * ds_ratio_hv["HV to LV"]
# scale line lengths based on growth factors
grid_lines = (grid_lines * grid_growth_expanded).rename(grid_lines.name).assign_attrs(grid_lines.attrs) # calculate line lengths over time based on growth factors; restore name and attributes (lost during multiplication)


# 3. calculate underground lines -------------------------------------------------
# determine length of underground and aboveground grid lines based on GDP/capita: fraction_underground = mult * gdp_pc + add
# Based on insights from Kalt et al. 2021, we adjust the underground ratios downwards for non-European regions

fraction_lines_above_below = xr.full_like(grid_lines, np.nan).rename("FractionUndergroundAboveground")

gdp_pc_unitless = gdp_pc.pint.dequantify() #work-around for now
for region in IMAGE_REGIONS:
    if region in ['WEU','CEU']:
        select_proxy = 'Europe'
    else:
        select_proxy = 'Other'

    fraction_lines_above_below.loc[{"Region": region, "Type": "HV - Lines - Underground"}] = (gdp_pc_unitless.sel(Region=region) * ratio_underground.loc[idx[select_proxy,'mult'],'HV'] + ratio_underground.loc[idx[select_proxy,'add'],'HV'])/100 # /100 to convert from % to fraction
    fraction_lines_above_below.loc[{"Region": region, "Type": "MV - Lines - Underground"}] = (gdp_pc_unitless.sel(Region=region) * ratio_underground.loc[idx[select_proxy,'mult'],'MV'] + ratio_underground.loc[idx[select_proxy,'add'],'MV'])/100
    fraction_lines_above_below.loc[{"Region": region, "Type": "LV - Lines - Underground"}] = (gdp_pc_unitless.sel(Region=region) * ratio_underground.loc[idx[select_proxy,'mult'],'LV'] + ratio_underground.loc[idx[select_proxy,'add'],'LV'])/100
# fraction must be between 0 and 1
fraction_lines_above_below = fraction_lines_above_below.clip(min=0, max=1)

# MIND! the HV lines found in OSM (+national sources) are considered as the total of the aboveground line length + the underground line length
# currently the total is saved in xarray grid_lines as the aboveground fraction only -> need to split into aboveground & underground based on the calculated ratios
# for this, we copy the aboveground length into the underground length & then multiply both with the coresponding fraction to get the correct lengths
for level in ["HV", "MV", "LV"]:
    over = f"{level} - Lines - Overhead"
    under = f"{level} - Lines - Underground"
    grid_lines.loc[dict(Type=under)] = grid_lines.sel(Type=over) # under = over
    fraction_lines_above_below.loc[dict(Type=over)] = 1-fraction_lines_above_below.loc[dict(Type=under)] # above = 1 - under

grid_lines = (grid_lines * fraction_lines_above_below).rename(grid_lines.name).assign_attrs(grid_lines.attrs) # calculate line lengths over time based on growth factors; restore name and attributes (lost during multiplication)


# 4. calculate number of substations & transformers based on line lengths & fixed ratios -------------------------------------------------------------
ureg = grid_lines.data._REGISTRY # extract unit registry from pint xarray dataarray to be able to define units for the calculations below
for level in ["HV", "MV", "LV"]:
    grid_additions.loc[dict(Type=f"{level} - Transformers")] = (grid_lines.sel(Type=f"{level} - Lines - Overhead") + grid_lines.sel(Type=f"{level} - Lines - Underground")) * (ratio_grid_additions.loc['Transformers',level] * (ureg.count/ureg.kilometer))  # pandas dataframe does not support units itself -> workaround: manually multiply with units count/km
    grid_additions.loc[dict(Type=f"{level} - Substations")]  = (grid_lines.sel(Type=f"{level} - Lines - Overhead") + grid_lines.sel(Type=f"{level} - Lines - Underground")) * (ratio_grid_additions.loc['Substations',level] * (ureg.count/ureg.kilometer))  



###########################################################################################################
#%%% Interpolate #
###########################################################################################################


grid_lifetime_interp = interpolate_xr(grid_lifetime, YEAR_FIRST_GRID, YEAR_OUT)

grid_lines_interp = add_historic_stock(grid_lines, YEAR_FIRST_GRID)
grid_additions_interp = add_historic_stock(grid_additions, YEAR_FIRST_GRID)





flexible_plot_1panel(
    da=grid_additions_interp,
    x_dim="Time",
    varying_dims=["Type", "Region"],
    fixed={"Type": [1, 2, 3, 4,5], "Region": [3]}, #
    plot_type='scatter'
)

# above: done, below: in progress -----------------------------------------------------------------------------------------







# below: old ====================================================================================================

# gcap_data = gcap_data.loc[~gcap_data['DIM_1'].isin([27,28])]    # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
# gcap = pd.pivot_table(gcap_data[gcap_data['time'].isin(list(range(YEAR_START,YEAR_END+1)))], index=['time','DIM_1'], values=list(range(1,TECH_GEN+1)))  #gcap as multi-index (index = years & regions (26); columns = technologies (28));  the last column in gcap_data (= totals) is now removed


# region_list = list(grid_length_Hv.columns.values)
# material_list = list(materials_grid.columns.values)

# renaming multi-index dataframe: generation capacity, based on the regions in grid_length_Hv & technologies as given
# gcap_techlist = ['Solar PV', 'Solar Decentral', 'CSP', 'Wind onshore', 'Wind offshore', 'Wave', 'Hydro', 'Other Renewables', 'Geothermal', 'Hydrogen', 'Nuclear', '<EMPTY>', 'Conv. Coal', 'Conv. Oil', 'Conv. Natural Gas', 'Waste', 'IGCC', 'OGCC', 'NG CC', 'Biomass CC', 'Coal + CCS', 'Oil/Coal + CCS', 'Natural Gas + CCS', 'Biomass + CCS', 'CHP Coal', 'CHP Oil', 'CHP Natural Gas', 'CHP Biomass', 'CHP Coal + CCS', 'CHP Oil + CCS', 'CHP Natural Gas + CCS', 'CHP Biomass + CCS', 'CHP Geothermal', 'CHP Hydrogen']
# gcap.index = pd.MultiIndex.from_product([list(range(YEAR_START,YEAR_END+1)), region_list], names=['years', 'regions'])
# gcap.columns = gcap_techlist

# gdp_pc = gdp_pc.iloc[:, :len(IMAGE_REGIONS)] # exclude empty column (27) and totals (28)
# gdp_pc.columns = region_list


# length calculations ----------------------------------------------------------------------------

# regional total (peak) generation capacity is used as a proxy for the grid growth 
# gcap_total = gcap.sum(axis=1).unstack()
# gcap_total = gcap_total[region_list]                     # re-order columns to the original TIMER order
# gcap_growth = gcap_total / gcap_total.loc[2016]        # define growth according to 2016 as base year


# Hv length (in kms) is region-specific. However, we use a single ratio between the length of Hv and Mv networks, the same applies to Lv networks 
# grid_length_Mv = grid_length_Hv.mul(ratio_hv['HV to MV'])
# grid_length_Lv = grid_length_Hv.mul(ratio_hv['HV to LV'])

# define grid length over time (fixed in 2016, growth according to gcap)
# grid_length_Hv_time = pd.DataFrame().reindex_like(gcap_total)
# grid_length_Mv_time = pd.DataFrame().reindex_like(gcap_total)
# grid_length_Lv_time = pd.DataFrame().reindex_like(gcap_total)

#implement growth correction (sensitivity variant)
# if SENS_ANALYSIS == 'high_grid':
#     # in the sensitivity variant, additional growth is presumed after 2020 based on the fraction of variable renewable energy (vre) generation capacity (solar & wind)
#     vre_fraction = gcap[['Solar PV', 'CSP', 'Wind onshore', 'Wind offshore']].sum(axis=1).unstack().divide(gcap.sum(axis=1).unstack())
#     add_growth = vre_fraction * 1                  # 0.2 = 20% additional HV lines per doubling of vre gcap
#     red_growth = (1-vre_fraction) * 0.7            # 0.2 = 20% less HV lines per doubling of baseline gcap
#     add_growth.loc[list(range(1971,2020+1)),:] = 0  # pre 2020, additional HV grid growth is 0, afterwards the additional line length is gradually introduced (towards 2050)
#     red_growth.loc[list(range(1971,2020+1)),:] = 0  # pre 2020, reduction of HV grid growth is 0, afterwards the line length reduction is gradually introduced (towards 2050)
#     for year in range(2020,2050+1):
#         add_growth.loc[year] = add_growth.loc[year] * (1/30*(year-2020)) 
#         red_growth.loc[year] = red_growth.loc[year] * (1/30*(year-2020)) 

#     gcap_growth_HV = gcap_growth.add(add_growth.reindex_like(gcap_growth)).subtract(red_growth.reindex_like(gcap_growth))
# else: 
#     gcap_growth_HV = gcap_growth

# for year in range(YEAR_START, YEAR_END+1):
#    grid_length_Hv_time.loc[year] = gcap_growth_HV.loc[year].mul(grid_length_Hv.loc['2016'])
#    grid_length_Mv_time.loc[year] = gcap_growth.loc[year].mul(grid_length_Mv.loc['2016'])
#    grid_length_Lv_time.loc[year] = gcap_growth.loc[year].mul(grid_length_Lv.loc['2016'])

# # define underground vs. aboveground fraction (%) based on static ratios (the Hv length is the aboveground fraction according to Open Street Maps, we add the underground fractions for 3 voltage networks)
# # Based on new insights from Kalt et al. 2021, we adjust the underground ratios downwards for non-European regions
# ratio_hv_under = pd.DataFrame(index=gdp_pc.index, columns=gdp_pc.columns)
# ratio_Mv_under = pd.DataFrame(index=gdp_pc.index, columns=gdp_pc.columns)
# ratio_Lv_under = pd.DataFrame(index=gdp_pc.index, columns=gdp_pc.columns)

# for region in region_list:
#     if region in ['W.Europe','C.Europe']:
#         select_proxy = 'Europe'
#     else:
#         select_proxy = 'Other'
#     #print(str(region) + ': ' + select_proxy)
#     ratio_hv_under[region] = gdp_pc[region] * underground_ratio.loc[idx[select_proxy,'mult'],'HV'] + underground_ratio.loc[idx[select_proxy,'add'],'HV']
#     ratio_Mv_under[region] = gdp_pc[region] * underground_ratio.loc[idx[select_proxy,'mult'],'MV'] + underground_ratio.loc[idx[select_proxy,'add'],'MV']
#     ratio_Lv_under[region] = gdp_pc[region] * underground_ratio.loc[idx[select_proxy,'mult'],'LV'] + underground_ratio.loc[idx[select_proxy,'add'],'LV']

# # maximize linear ratio at 100 & minimize at 0 (%)
# ratio_hv_under = ratio_hv_under.apply(lambda x: [y if y >= 0 else 0 for y in x])                     
# ratio_hv_under = ratio_hv_under.apply(lambda x: [y if y <= 100 else 100 for y in x])  
# ratio_Mv_under = ratio_Mv_under.apply(lambda x: [y if y >= 0 else 0 for y in x])  
# ratio_Mv_under = ratio_Mv_under.apply(lambda x: [y if y <= 100 else 100 for y in x])  
# ratio_Lv_under = ratio_Lv_under.apply(lambda x: [y if y >= 0 else 0 for y in x])  
# ratio_Lv_under = ratio_Lv_under.apply(lambda x: [y if y <= 100 else 100 for y in x])

# MIND! the HV lines found in OSM (+national sources) are considered as the total of the aboveground line length + the underground line length
# grid_length_Hv_total = grid_length_Hv_time   # assuming the length from OSM IS the abovegrond fraction
# grid_length_Hv_above = grid_length_Hv_total * (1 - (ratio_hv_under/100)) 
# grid_length_Hv_under = grid_length_Hv_total * ratio_hv_under/100

# out for main text figure 2
# grid_length_HV_out_a = pd.concat([grid_length_Hv_above], keys=['aboveground'], names=['type']) 
# grid_length_HV_out_u = pd.concat([grid_length_Hv_under], keys=['underground'], names=['type']) 
# grid_length_HV_out   = pd.concat([grid_length_HV_out_a, grid_length_HV_out_u])
# grid_length_HV_out.to_csv(path_base / 'imagematerials' / 'electricity' / 'out_test' / 'grid_length_HV_km.csv') # in km

# grid_length_Mv_above = grid_length_Mv_time * (1 - ratio_Mv_under/100)
# grid_length_Mv_under = grid_length_Mv_time * ratio_Mv_under/100
# grid_length_Mv_total = grid_length_Mv_above + grid_length_Mv_under

# grid_length_Lv_above = grid_length_Lv_time * (1 - ratio_Lv_under/100)
# grid_length_Lv_under = grid_length_Lv_time * ratio_Lv_under/100
# grid_length_Lv_total = grid_length_Lv_above + grid_length_Lv_under

# grid_subst_Hv = grid_length_Hv_total.mul(grid_additions.loc['Substations','HV'])        # number of substations on HV network
# grid_subst_Mv = grid_length_Mv_total.mul(grid_additions.loc['Substations','MV'])        # # of substations
# grid_subst_Lv = grid_length_Lv_total.mul(grid_additions.loc['Substations','LV'])        # # of substations
# grid_trans_Hv = grid_length_Hv_total.mul(grid_additions.loc['Transformers','HV'])       # number of transformers on the HV network
# grid_trans_Mv = grid_length_Mv_total.mul(grid_additions.loc['Transformers','MV'])       # # of transformers
# grid_trans_Lv = grid_length_Lv_total.mul(grid_additions.loc['Transformers','LV'])       # # of transformers




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
# grid_length_Hv_above_new = stock_tail(grid_length_Hv_above, YEAR_OUT) # km
# grid_length_Mv_above_new = stock_tail(grid_length_Mv_above, YEAR_OUT) # km
# grid_length_Lv_above_new = stock_tail(grid_length_Lv_above, YEAR_OUT) # km
# grid_length_Hv_under_new = stock_tail(grid_length_Hv_under, YEAR_OUT) # km 
# grid_length_Mv_under_new = stock_tail(grid_length_Mv_under, YEAR_OUT) # km
# grid_length_Lv_under_new = stock_tail(grid_length_Lv_under, YEAR_OUT) # km
# grid_subst_Hv_new = stock_tail(grid_subst_Hv, YEAR_OUT)               # units
# grid_subst_Mv_new = stock_tail(grid_subst_Mv, YEAR_OUT)               # units
# grid_subst_Lv_new = stock_tail(grid_subst_Lv, YEAR_OUT)               # units
# grid_trans_Hv_new = stock_tail(grid_trans_Hv, YEAR_OUT)               # units
# grid_trans_Mv_new = stock_tail(grid_trans_Mv, YEAR_OUT)               # units
# grid_trans_Lv_new = stock_tail(grid_trans_Lv, YEAR_OUT)               # units


#############
# Lifetimes #
#############

# data only for lines, substations and transformer -> bring in knowledge_graph format: HV - Lines, MV - Lines, LV - Lines, HV - Transformers, etc.
# expanded_data = {}
# for typ in ["Lines", "Transformers", "Substations"]:
#     for level in ["HV", "MV", "LV"]:
#         new_col = f"{level} - {typ}"
#         expanded_data[new_col] = lifetime_grid_elements[typ]
# lifetime_grid_elements = pd.DataFrame(expanded_data, index=lifetime_grid_elements.index)
# lifetime_grid_elements.rename_axis('Year', inplace=True)

# # no differentiation between HV, MV & LV lines as well as between aboveground and belowground
# # Types: lines, transformers, substations
# lifetime_grid_elements.loc[YEAR_FIRST_GRID,:]  = lifetime_grid_elements.loc[lifetime_grid_elements.first_valid_index(),:]
# lifetime_grid_elements.loc[YEAR_OUT,:]         = lifetime_grid_elements.loc[lifetime_grid_elements.last_valid_index(),:]
# lifetime_grid_elements                         = lifetime_grid_elements.reindex(list(range(YEAR_FIRST_GRID, YEAR_OUT+1))).interpolate()
# TODO: check why lifetime for lines is interpoltaed from 2020 - 40yrs to 2050 - 48 yrs and then back to 2060 - 40 yrs -> should stay at 48 yrs?


#######################################################################################################
#%%%% NEW

# lifetimes
# df_mean = lifetime_grid_elements.copy()
# df_stdev = df_mean * STD_LIFETIMES_ELECTR
# df_mean.columns = [(col, 'mean') for col in df_mean.columns] # Rename columns to multi-level tuples
# df_stdev.columns = [(col, 'stdev') for col in df_stdev.columns]
# lifetime_grid_distr = pd.concat([df_mean, df_stdev], axis=1) # Concatenate along columns


# Materials

# harmonize units of MI -> then materials_grid and materials_grid_additions can be merged in one dataframe
# NOT POSSIBLE - separate stock modelling necessary

# Grid Additions MIs ---
# # for substations and transformer we multiply the MI (kg/unit of substation or transformer) with the number of substations or transformers per kilometer of grid length
# materials_grid_additions_kgperkm             = materials_grid_additions_to_kgperkm(materials_grid_additions_interpol, grid_additions)
# # material intensities: (years, tech. type) index and materials as columns -> years as index and (tech. type, materials) as columns
# materials_grid_additions_kgperkm.index.names = ["Year", "Type"]
# materials_grid_additions_kgperkm             = materials_grid_additions_kgperkm.unstack(level='Type')   # bring tech. type from row index to column header
# materials_grid_additions_kgperkm.columns     = materials_grid_additions_kgperkm.columns.swaplevel(0, 1) # Swap the levels of the MultiIndex columns
# materials_grid_additions_kgperkm             = materials_grid_additions_kgperkm.sort_index(axis=1)
# # rename columns to match knowledge graph
# new_level_0 = [col.replace(' ', ' - ', 1) for col in materials_grid_additions_kgperkm.columns.get_level_values(0)]
# new_columns = pd.MultiIndex.from_arrays([
#     new_level_0,
#     materials_grid_additions_kgperkm.columns.get_level_values(1)
# ], names=materials_grid_additions_kgperkm.columns.names)
# materials_grid_additions_kgperkm.columns = new_columns

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

# # print_df_info(materials_grid_kgperkm, "Materials Grid kg/km")
# # print_df_info(materials_grid_additions_kgperkm, "Materials Grid Additions kg/km")

# materials_grid_combined_kgperkm = pd.concat([materials_grid_kgperkm, materials_grid_additions_kgperkm], axis=1) # 


# # Stocks
# grid_dict = dict({
#         'HV - Lines - Overhead':    grid_length_Hv_above_new,
#         'HV - Lines - Underground': grid_length_Hv_under_new,
#         'HV - Substations':         grid_subst_Hv_new,
#         'HV - Transformers':        grid_trans_Hv_new,
#         'MV - Lines - Overhead':    grid_length_Mv_above_new,
#         'MV - Lines - Underground': grid_length_Mv_under_new,
#         'MV - Substations':         grid_subst_Mv_new,
#         'MV - Transformers':        grid_trans_Mv_new,
#         'LV - Lines - Overhead':    grid_length_Lv_above_new,
#         'LV - Lines - Underground': grid_length_Lv_under_new,
#         'LV - Substations':         grid_subst_Lv_new,
#         'LV - Transformers':        grid_trans_Lv_new
#     })
# grid_stock = pd.concat(grid_dict, axis=1) # Concatenate with keys to create MultiIndex ('Name', 'Region')
# grid_stock = grid_stock.sort_index(axis=1)

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

#----------------------------------------------------------------------------------------------------------
###########################################################################################################
#%%% 3.3) Prep_data File
###########################################################################################################
#----------------------------------------------------------------------------------------------------------


ureg = pint.UnitRegistry(force_ndarray_like=True)
# Define the units for each dimension
unit_mapping = { # TODO: move to constants.py
    'time': ureg.year,
    'year': ureg.year,
    'Year': ureg.year,
    'kg': ureg.kilogram,
    'yr': ureg.year,
    '%': ureg.percent,
    't': ureg.tonne,
    'MW': ureg.megawatt, #added
    'GW': ureg.gigawatt, #added
}

# Conversion table for all coordinates, to be removed/adapted after input tables are fixed.
# conversion_table = {
#     "grid_stock": (["Time"], ["Type", "Region"],),
#     "materials_grid_combined_kgperkm": (["Cohort"], ["Type", "material"],)
#     # "gcap_materials_interpol": (["Cohort"], ["Type", "SubType", "material"], {"Type": ["Type", "SubType"]})
# }

conversion_table = {
    "grid_stock_lines": (["Time"], ["Type", "Region"],),
    "materials_grid_kgperkm": (["Cohort"], ["Type", "material"],),
    "grid_stock_add": (["Time"], ["Type", "Region"],),
    "materials_grid_add_kgperunit": (["Cohort"], ["Type", "material"],),
    "grid_stock": (["Time"], ["Type", "Region"],), # TODO: delete
    "materials_grid_combined_kgperkm": (["Cohort"], ["Type", "material"],) # TODO: delete
}



# results_dict = {
#         'grid_stock': grid_stock,
#         'materials_grid_combined_kgperkm': materials_grid_combined_kgperkm,
#         'lifetime_grid_distr': lifetime_grid_distr,
# }
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


# change region names to IMAGE_REGIONS # TODO: this should be done in hte beginning of preprocessing
knowledge_graph_region =    create_region_graph()
prep_data_lines["stocks"] = knowledge_graph_region.rebroadcast_xarray(prep_data_lines["stocks"], output_coords=IMAGE_REGIONS, dim="Region")
prep_data_add["stocks"] =   knowledge_graph_region.rebroadcast_xarray(prep_data_add["stocks"], output_coords=IMAGE_REGIONS, dim="Region")

#----------------------------------------------------------------------------------------------------------
###########################################################################################################
#%%% 3.4) Run Stock Model New
###########################################################################################################
# TODO: move this to electricity.py
#----------------------------------------------------------------------------------------------------------


prep_data_lines, prep_data_add = get_preprocessing_data_grid(path_base, SCEN, VARIANT, YEAR_START, YEAR_END, YEAR_OUT)

# LINES ----------------------------------------------------
# prep_data = create_prep_data(results_dict_lines, conversion_table, unit_mapping)

# # Define the complete timeline, including historic tail
time_start = prep_data["stocks"].coords["Time"].min().values
complete_timeline = prism.Timeline(time_start, YEAR_END, 1)
simulation_timeline = prism.Timeline(YEAR_START, YEAR_END, 1) #1970


sec_electr_grid_lines = Sector("electr_grid_lines", prep_data_lines)

main_model_factory_lines = ModelFactory(
    sec_electr_grid_lines, complete_timeline
    ).add(GenericStocks
    ).add(MaterialIntensities
    ).finish()

main_model_factory_lines.simulate(simulation_timeline)
list(main_model_factory_lines.electr_grid_lines)

# ADDITIONS -------------------------------------------------------------------------------------
# prep_data = create_prep_data(results_dict_add, conversion_table, unit_mapping)

# # Define the complete timeline, including historic tail
time_start = prep_data["stocks"].coords["Time"].min().values
complete_timeline = prism.Timeline(time_start, YEAR_END, 1)
simulation_timeline = prism.Timeline(YEAR_START, YEAR_END, 1) #1970


sec_electr_grid_add = Sector("electr_grid_add", prep_data_add)

main_model_factory_add = ModelFactory(
    sec_electr_grid_add, complete_timeline
    ).add(GenericStocks
    ).add(MaterialIntensities
    ).finish()

main_model_factory_add.simulate(simulation_timeline)
list(main_model_factory_add.electr_grid_add)



# Define the coordinates of all dimensions.
# Region = list(prep_data["stocks"].coords["Region"].values)
# Time = [t for t in complete_timeline]
# Cohort = Time
# Type = list(prep_data["stocks"].coords["Type"].values)
# material = list(prep_data["material_intensities"].coords["material"].values)

# Create
# main_model_normal = GenericMainModel(
#     complete_timeline, Region=Region, Time=Time, Cohort=Cohort, Type=Type, prep_data=prep_data,
#     compute_materials=True, compute_battery_materials=False, compute_maintenance_materials=False, 
#     material=material)

path_test_plots = Path(path_base, "imagematerials", "electricity", "out_test", "Test", "Figures") #scen_folder




###########################################################################################################
#%%% Stocks 
###########################################################################################################

#%%%% 2 models ---------------------------------------------------


data_lines  = main_model_factory_lines.stocks.copy().sum(dim="Region")
data_add    = main_model_factory_add.stocks.copy().sum(dim="Region")

# data        = xr.concat([data_lines, data_add], dim='Type')
# data_plot   = data.sum(dim="Region")

types_top    = ['HV - Lines - Overhead', 'HV - Lines - Underground', 'MV - Lines - Overhead', 'MV - Lines - Underground', 
                'LV - Lines - Overhead', 'LV - Lines - Underground'] #, 'LV - Transformers', 'LV - Substations',
types_bottom = ['HV - Substations', 'HV - Transformers', 'MV - Substations', 'MV - Transformers']



fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 8))
linewidth = 2
s_legend = 12
s_label = 14

# Top row:
for t in types_top:
    line, = axes[0].plot(data_lines.Time, data_lines.sel(Type=t), label=t, color=DICT_GRID_STYLES_2[t][0], linestyle=DICT_GRID_STYLES_2[t][1], linewidth=linewidth)

axes[0].set_ylabel("Stocks (# counts/km)", fontsize=s_label)
axes[0].grid(alpha=0.3, linestyle='--')
axes[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
axes[0].tick_params(axis='both', which='major', labelsize=s_legend) # set font size of axis ticks
axes[0].legend(loc='upper left', fontsize=s_legend) #handles=handles, labels=labels, 

# Bottom row:
for t in types_bottom:
    axes[1].plot(data_add.Time, data_add.sel(Type=t), label=t, color=DICT_GRID_STYLES_2[t][0], linestyle=DICT_GRID_STYLES_2[t][1], linewidth=linewidth)

axes[1].set_xlabel("Time", fontsize=s_label)
axes[1].set_ylabel("Stocks (# counts)", fontsize=s_label)
axes[1].grid(alpha=0.3, linestyle='--')
axes[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
axes[1].tick_params(axis='both', which='major', labelsize=s_legend) # set font size of axis ticks
axes[1].legend(loc='upper left', fontsize=s_legend)

plt.suptitle(f"{scen_folder}: Electricity Grid - Stocks", fontsize=16)

plt.tight_layout()
# fig.savefig(path_test_plots / "Grid_stocks_world.png", dpi=300)
plt.show()


###########################################################################################################
#%%% Stocks Materials
###########################################################################################################



#%%%% 2 model ---------------------------------------------------

materials = ["steel", "concrete", "aluminium", "copper"]

data_lines  = main_model_factory_lines.stock_by_cohort_materials.copy().sum(dim="Region").pint.to("t")  # Convert kg -> tonnes
data_add    = main_model_factory_add.stock_by_cohort_materials.copy().sum(dim="Region").pint.to("t")  # Convert kg -> tonnes

data        = xr.concat([data_lines, data_add], dim='Type')
data        = data.sel(Time=slice(1971, None))

data_sum    = data.sum(dim='Type')

lines_sum           = data.sel(Type=[t for t in data.Type.values if 'Lines' in t]).sum(dim='Type') # Get group sums by keyword and sum over types (sum over HV, MV and LV (and overground/underground for lines))
transformers_sum    = data.sel(Type=[t for t in data.Type.values if 'Transformers' in t]).sum(dim='Type')
substations_sum     = data.sel(Type=[t for t in data.Type.values if 'Substations' in t]).sum(dim='Type')


fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 10))
linewidth = 2
s_legend = 12
s_label = 14

for i, mat in enumerate(materials):
    lines_sum.sel(material=mat).plot(ax=axes[i], color = DICT_GRID_COLORS['Lines'], label="Lines")
    transformers_sum.sel(material=mat).plot(ax=axes[i], color = DICT_GRID_COLORS['Transformers'], label="Transformers")
    substations_sum.sel(material=mat).plot(ax=axes[i], color = DICT_GRID_COLORS['Substations'], label="Substations")
    data_sum.sel(material=mat).plot(ax=axes[i], label="Total", color='red', alpha=0.8, linestyle='--', linewidth=3)

    # if mat == "Cu":
    #     # Add IEA data for copper
    #     axes[i].plot(df_iea_lt_cu.index, df_iea_lt_cu['Cu'], label="IEA L&T", color='#00a5cf', linestyle=':', linewidth=4)

    # if mat == "Aluminium":
    #     # Add IEA data for aluminium
    #     axes[i].plot(df_iea_lt_alu.index, df_iea_lt_alu['Aluminium'], label="IEA L&T", color='#00a5cf', linestyle=':', linewidth=4)

    # if mat == "Steel":
    #     # Add IEA data for steel
    #     axes[i].plot(df_iea_t_steel.index, df_iea_t_steel['Steel'], label="IEA T", color='#00a5cf', linestyle=':', linewidth=4)
    
    axes[i].grid(alpha=0.3, linestyle='--')
    axes[i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
    axes[i].tick_params(axis='both', which='major', labelsize=s_legend) # set font size of axis ticks
    axes[i].set_title(f"{mat}")
    axes[i].set_xlabel(" ")
    axes[i].set_ylabel("Inflow [t]", fontsize=s_label)
    axes[i].legend()

axes[-1].set_xlabel("Time", fontsize=s_label)

plt.suptitle(f"{scen_folder}: Electricity Grid Stocks Materials", fontsize=16)
plt.tight_layout()
# fig.savefig(path_test_plots / "Grid_stocks-materials_world.svg")
# fig.savefig(path_test_plots / "Grid_stocks-materials_world_1971.pdf")
fig.savefig(path_test_plots / "Grid_stocks-materials_world_1971.png")
# fig.savefig(path_test_plots / "Grid_stocks-materials_world_1971.svg")
plt.show()


###########################################################################################################
#%% INFLOW Materials
###########################################################################################################

#================================================================================
#%%%% Per TECH - World


regions = ['Brazil', 'C.Europe', 'China'] 
materials = ["steel", "concrete", "aluminium", "copper"]

#%%%% 2 model ---------------------------------------------------

data_lines  = main_model_factory_lines.inflow_materials.to_array().sum(dim="Region").pint.to("t")  # Convert kg -> tonnes
data_add    = main_model_factory_add.inflow_materials.to_array().sum(dim="Region").pint.to("t")  # Convert kg -> tonnes

data_lines  = data_lines.sel(time=slice(1971, None))
data_add    = data_add.sel(time=slice(1971, None))
data        = xr.concat([data_lines, data_add], dim='Type')
data_sum    = data.sum(dim='Type')

lines_sum           = data_lines.sel(Type=[t for t in data_lines.Type.values if 'Lines' in t]).sum(dim='Type') # Get group sums by keyword and sum over types (sum over HV, MV and LV (and overground/underground for lines))
transformers_sum    = data_add.sel(Type=[t for t in data_add.Type.values if 'Transformers' in t]).sum(dim='Type')
substations_sum     = data_add.sel(Type=[t for t in data_add.Type.values if 'Substations' in t]).sum(dim='Type')


fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 10))
linewidth = 2
s_legend = 12
s_label = 14

for i, mat in enumerate(materials):
    lines_sum.sel(material=mat).plot(ax=axes[i], color = DICT_GRID_COLORS['Lines'], label="Lines")
    transformers_sum.sel(material=mat).plot(ax=axes[i], color = DICT_GRID_COLORS['Transformers'], label="Transformers")
    substations_sum.sel(material=mat).plot(ax=axes[i], color = DICT_GRID_COLORS['Substations'], label="Substations")
    data_sum.sel(material=mat).plot(ax=axes[i], label="Total", color='red', alpha=0.8, linestyle='--', linewidth=3)

    # if mat == "Cu":
    #     # Add IEA data for copper
    #     axes[i].plot(df_iea_lt_cu.index, df_iea_lt_cu['Cu'], label="IEA L&T", color='#00a5cf', linestyle=':', linewidth=4)

    # if mat == "Aluminium":
    #     # Add IEA data for aluminium
    #     axes[i].plot(df_iea_lt_alu.index, df_iea_lt_alu['Aluminium'], label="IEA L&T", color='#00a5cf', linestyle=':', linewidth=4)

    # if mat == "Steel":
    #     # Add IEA data for steel
    #     axes[i].plot(df_iea_t_steel.index, df_iea_t_steel['Steel'], label="IEA T", color='#00a5cf', linestyle=':', linewidth=4)
    
    axes[i].grid(alpha=0.3, linestyle='--')
    axes[i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
    axes[i].tick_params(axis='both', which='major', labelsize=s_legend) # set font size of axis ticks
    axes[i].set_title(f"{mat}")
    axes[i].set_xlabel(" ")
    axes[i].set_ylabel("Inflow [t]", fontsize=s_label)
    axes[i].legend()

axes[-1].set_xlabel("Time", fontsize=s_label)

plt.suptitle(f"{scen_folder}: Electricity Grid Inflow Materials", fontsize=16)
plt.tight_layout()
# fig.savefig(path_test_plots / "Grid_inflow-materials_world.svg")
# fig.savefig(path_test_plots / "Grid_inflow-materials_world_1971.pdf")
fig.savefig(path_test_plots / "Grid_inflow-materials_world_1971.png")
# fig.savefig(path_test_plots / "Grid_inflow-materials_world_1971.svg")
plt.show()


