#%% Import modules and constants
import pandas as pd
import numpy as np
from pathlib import Path
import pint
import xarray as xr


import prism
from imagematerials.read_mym import read_mym_df
from imagematerials.util import dataset_to_array, pandas_to_xarray, convert_lifetime
from imagematerials.concepts import create_electricity_graph, create_region_graph
from imagematerials.electricity.utils import MNLogit, stock_tail, create_prep_data, interpolate_xr, add_historic_stock

from imagematerials.constants import IMAGE_REGIONS

from imagematerials.electricity.constants import (
    STANDARD_SCEN_EXTERNAL_DATA,
    YEAR_FIRST,
    YEAR_FIRST_GRID,
    YEAR_SWITCH,
    SENS_ANALYSIS,
    EPG_TECHNOLOGIES,
    STD_LIFETIMES_ELECTR,
    LOAD_FACTOR,
    BEV_CAPACITY_CURRENT,
    PHEV_CAPACITY_CURRENT,
    unit_mapping
)



###########################################################################################################
###########################################################################################################
#%% 1) Generation 
###########################################################################################################
###########################################################################################################


def get_preprocessing_data_gen(path_base: str, SCEN, VARIANT, YEAR_START, YEAR_END, YEAR_OUT): #, climate_policy_config: dict, circular_economy_config: dict

    scen_folder = SCEN + "_" + VARIANT
    path_image_output = Path(path_base, "data", "raw", "image", scen_folder, "EnergyServices")
    path_external_data_standard = Path(path_base, "data", "raw", "electricity", "standard_data")
    path_external_data_scenario = Path(path_base, "data", "raw", "electricity", scen_folder)
    # test if path_external_data_scenario exists and if not set to standard scenario
    if not path_external_data_scenario.exists():
        path_external_data_scenario = Path(path_base, "data", "raw", "electricity", STANDARD_SCEN_EXTERNAL_DATA)
    print(f"Path to image output: {path_image_output}")
    assert path_image_output.is_dir()
    assert path_external_data_standard.is_dir()
    assert path_external_data_scenario.is_dir()


    ###########################################################################################################
    # Read in files #

    # 1. External Data --------------------------------------------- 

    # lifetimes of Gcap tech (original data according to van Vuuren 2006, PhD Thesis)
    gcap_lifetime_data = pd.read_csv(path_external_data_scenario / 'LTTechnical_dynamic.csv', index_col=['Year','DIM_1'])        
    
    # material compositions of electricity generation tecnologies (g/MW)
    gcap_materials_data = pd.read_csv(path_external_data_scenario / 'composition_generation.csv',index_col=[0,1]).transpose()

    # 2. IMAGE/TIMER files -----------------------------------------

    # Generation capacity (stock demand per generation technology) in MW peak capacity
    gcap_data = read_mym_df(path_image_output / 'Gcap.out')



    ###########################################################################################################
    # Transform to xarray #

    knowledge_graph_region = create_region_graph()
    knowledge_graph_electr = create_electricity_graph()
    

    # Lifetimes -------
    values = gcap_lifetime_data["TechnicalLT"].unstack().to_numpy(dtype=float)
    # Create coordinates
    times = gcap_lifetime_data.index.levels[0].to_numpy()
    types = gcap_lifetime_data.index.levels[1].to_numpy()
    scipy_params = ["mean", "stdev"]
    # Build full array: shape (ScipyParam, Time, Type)
    data_array = np.stack([values, np.full_like(values, np.nan)], axis=0)
    # Create DataArray
    gcap_lifetime_xr = xr.DataArray(
        data_array,
        dims=["DistributionParams", "Cohort", "Type"],
        coords={
            "DistributionParams": scipy_params,
            "Cohort": times,
            "Type": [str(r) for r in types]
        },
        name="Lifetime"
    )
    gcap_lifetime_xr = prism.Q_(gcap_lifetime_xr, "year")
    gcap_lifetime_xr = knowledge_graph_electr.rebroadcast_xarray(gcap_lifetime_xr, output_coords=EPG_TECHNOLOGIES, dim="Type") # convert technology names to the standard names from TIMER
    gcap_lifetime_xr = gcap_lifetime_xr.assign_coords(Type=np.array(gcap_lifetime_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)


    # Material Intensities -------
    # Extract coordinate labels
    gcap_materials_data.columns = gcap_materials_data.columns.rename([None, None]) # remove column MultiIndex name "g/MW" as it causes issues when converting to xarray
    years = sorted(gcap_materials_data.columns.get_level_values(0).unique())
    techs = gcap_materials_data.columns.get_level_values(1).unique()
    materials = gcap_materials_data.index
    # Convert to 3D array: (Material, Year, Tech)
    data_array = gcap_materials_data.to_numpy().reshape(len(materials), len(years), len(techs))
    # Build xarray DataArray
    gcap_materials_xr = xr.DataArray(
        data_array,
        dims=('material', 'Cohort', 'Type'),
        coords={
            'material': materials,
            'Cohort': years,
            'Type': techs
        },
        name='MaterialIntensities'
    )
    gcap_materials_xr = prism.Q_(gcap_materials_xr, "g/MW")
    gcap_materials_xr = knowledge_graph_electr.rebroadcast_xarray(gcap_materials_xr, output_coords=EPG_TECHNOLOGIES, dim="Type")
    gcap_materials_xr = gcap_materials_xr.assign_coords(Type=np.array(gcap_materials_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)


    # Gcap ------
    gcap_data = gcap_data.loc[~gcap_data['DIM_1'].isin([27,28])]  # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
    gcap_data = gcap_data.loc[gcap_data['time'].isin(range(YEAR_START, YEAR_END + 1)), ['time', 'DIM_1', *range(1, len(EPG_TECHNOLOGIES) + 1)]]  # only keep relevant years and technology columns
    # Extract coordinate labels
    years = sorted(gcap_data['time'].unique())
    regions = sorted(gcap_data['DIM_1'].unique())
    techs = list(range(1, len(EPG_TECHNOLOGIES)+1))
    # Convert to 3D array: (Year, Region, Tech)
    data_array = gcap_data[techs].to_numpy().reshape(len(years), len(regions), len(techs))
    # Build xarray DataArray
    gcap_xr = xr.DataArray(
        data_array,
        dims=('Time', 'Region', 'Type'),
        coords={
            'Time': years,
            'Region': [str(r) for r in regions],
            'Type': [str(r) for r in techs]
        },
        name='GCap'
    )
    gcap_xr = prism.Q_(gcap_xr, "MW")
    gcap_xr = knowledge_graph_region.rebroadcast_xarray(gcap_xr, output_coords=IMAGE_REGIONS, dim="Region") 
    gcap_xr = knowledge_graph_electr.rebroadcast_xarray(gcap_xr, output_coords=EPG_TECHNOLOGIES, dim="Type")
    gcap_xr = gcap_xr.assign_coords(Type=np.array(gcap_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)


    ###########################################################################################################
    # Interpolate #

    # interpolate_xr: The lifetimes & material intensities are only given for specific years (2020 and 2050), so we linearly interpolate to get values for the years 2020-2050.
    # The values before 2020 are kept constant at the 2020 level, and the values after 2050 are kept constant at the 2050 level.
    gcap_lifetime_xr_interp = interpolate_xr(gcap_lifetime_xr, YEAR_FIRST_GRID, YEAR_OUT)
    gcap_lifetime_xr_interp.loc[dict(DistributionParams="stdev")] = gcap_lifetime_xr_interp.loc[dict(DistributionParams="mean")] * STD_LIFETIMES_ELECTR
    gcap_materials_xr_interp = interpolate_xr(gcap_materials_xr, YEAR_FIRST_GRID, YEAR_OUT)

    # The lifetimes are converted to the proper format for the model (dictionary with keys:distribution name, values:datarrays containing distribution parameters)
    gcap_lifetime_xr_interp = convert_lifetime(gcap_lifetime_xr_interp)

    # TIMER data only start in 1971, so we add a historic tail back to YEAR_FIRST_GRID=1921 #TODO to be adjusted
    gcap_xr_interp = add_historic_stock(gcap_xr, YEAR_FIRST_GRID)


    ###########################################################################################################
    # Prep_data File #
    
    # bring preprocessing data into a generic format for the model
    prep_data = {}
    prep_data["lifetimes"] = gcap_lifetime_xr_interp
    prep_data["stocks"] = gcap_xr_interp
    prep_data["material_intensities"] = gcap_materials_xr_interp
    prep_data["knowledge_graph"] = create_electricity_graph()
    # add units
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

def get_preprocessing_data_grid(path_base: str, SCEN, VARIANT, YEAR_START, YEAR_END, YEAR_OUT): #, climate_policy_config: dict, circular_economy_config: dict

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
    gcap = pd.pivot_table(gcap_data[gcap_data['time'].isin(list(range(YEAR_START,YEAR_END+1)))], index=['time','DIM_1'], values=list(range(1,len(EPG_TECHNOLOGIES)+1)))  #gcap as multi-index (index = years & regions (26); columns = technologies (28));  the last column in gcap_data (= totals) is now removed

    gcap_BL_data = gcap_BL_data.loc[~gcap_BL_data['DIM_1'].isin([27,28])]    # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
    gcap_BL = pd.pivot_table(gcap_BL_data[gcap_BL_data['time'].isin(list(range(YEAR_START,YEAR_END+1)))], index=['time','DIM_1'], values=list(range(1,len(EPG_TECHNOLOGIES)+1)))  #gcap as multi-index (index = years & regions (26); columns = technologies (28));  the last column in gcap_data (= totals) is now removed

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
    lifetime_grid_elements.rename_axis('year', inplace=True)

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
    materials_grid_additions_kgperunit.index.names = ["year", "Type"]
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
    materials_grid_kgperkm.index.names  = ["year", "Type"]
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





###########################################################################################################
###########################################################################################################
#%% 1) Storage 
###########################################################################################################
###########################################################################################################



def get_preprocessing_data_stor(path_base: str, SCEN, VARIANT, YEAR_START, YEAR_END, YEAR_OUT): #, climate_policy_config: dict, circular_economy_config: dict
    """Preprocess electricity storage data and creates "prep_data" suitable for stock modelling using IMAGE-Materials.

    It does so for the two sub-sectors:
    1) Pumped Hydro Storage (PHS)
    2) Other Electricity Storage Technologies (e.g., batteries, compressed air, flywheels)
    (in the future also vehicle batteries will be added here)


    The function performs the following steps:
    1. Reads input data on storage stocks scenario projections (IMAGE/TIMER) and storage technologies, including costs, lifetimes, energy densities, and material compositions (different sources, e.g. IRENA, IHA)
    2. Interpolates data over time to fill missing years.
    3. Calculates market shares of different storage technologies using a multinomial logit model.
    4. Infers pumped hydro storage demand from IHA projections and IMAGE hydro power projections.
    5. Computes shares of total storage capacity from PHS, (electric vehicles - not yet), and other storage systems.

    Parameters
    ----------
    path_base : str
        Base directory of the project where the data structure is located.
    SCEN : str
        Scenario name (e.g., "SSP2").
    VARIANT : str
        Variant of the scenario (e.g., "M_CP", "VLHO").
    YEAR_START : int
        First simulation year for data processing (e.g., 1971).
    YEAR_END : int
        Last year for data interpolation (used for intermediate computations).
    YEAR_OUT : int
        Final output year for the results (e.g., 2100).

    Returns
    -------
    prep_data_phs: 
        Prepared dataset for pumped hydro storage, including:
            * "stocks": Time series of storage capacity (MWh)
            * "material_intensities": Material intensities (kg/MWh)
            * "lifetimes": Lifetime mean and standard deviation
            * "set_unit_flexible": Unit mapping for Prism integration
    prep_data_oth_storage: 
        Prepared dataset for other storage technologies, including:
            * "stocks": Time series of storage capacity (MWh)
            * "material_intensities": Material intensities (kg/kWh)
            * "lifetimes": Lifetime mean and standard deviation
            * "shares": Market shares of storage technologies
            * "set_unit_flexible": Unit mapping for Prism integration


    Notes
    -----
    - Assumes that the IMAGE/TIMER model outputs and other technology reference data are available in the expected
      directory structure under `path_base/data/raw/`.
    - Sensitivity variants (e.g., `high_stor`) affect storage demand scaling and pumped hydro projections. # TODO: move to parameters?

    """

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

    idx = pd.IndexSlice   

    #----------------------------------------------------------------------------------------------------------
    ###########################################################################################################
    #%%% 2.1) Read in files
    ###########################################################################################################
    #----------------------------------------------------------------------------------------------------------


    # 1. External Data ======================================================================================== 


    # read in the storage share in 2016 according to IEA (Technology perspectives 2017)
    storage_IEA = pd.read_csv(path_external_data_standard / 'storage_IEA2016.csv', index_col=0)

    # read in the storage costs according to IRENA storage report & other sources in the SI
    storage_costs = pd.read_csv(path_external_data_standard / 'storage_cost.csv', index_col=0).transpose()

    # read in the assumed malus & bonus of storage costs (malus for advanced technologies, still under development; bonus for batteries currently used in EVs, we assume that a large volume of used EV batteries will be available and used for dedicated electricity storage, thus lowering costs), only the bonus remains by 2030
    storage_malus = pd.read_csv(path_external_data_standard / 'storage_malus.csv', index_col=0).transpose()

    #read in the assumptions on the long-term price decline after 2050. Prices are in $ct / kWh electricity cycled (the fraction of the annual growth rate (determined based on 2018-2030) that will be applied after 2030, ranging from 0.25 to 1 - 0.25 means the price decline is not expected to continue strongly, while 1 means that the same (2018-2030) annual price decline is also applied between 2030 and 2050)
    storage_ltdecline = pd.Series(pd.read_csv(path_external_data_standard / 'storage_ltdecline.csv',index_col=0,  header=None).transpose().iloc[0])

    #read in the energy density assumptions (kg/kWh storage capacity - mass required to store one unit of energy — more mass per energy = worse performance)
    storage_density = pd.read_csv(path_external_data_standard / 'storage_density_kg_per_kwh.csv',index_col=0).transpose()

    #read in the lifetime of storage technologies (in yrs). The lifetime is assumed to be 1.5* the number of cycles divided by the number of days in a year (assuming diurnal use, and 50% extra cycles before replacement, representing continued use below 80% remaining capacity) OR the maximum lifetime in years, which-ever comes first 
    storage_lifetime = pd.read_csv(path_external_data_standard / 'storage_lifetime.csv',index_col=0).transpose()

    kilometrage = pd.read_csv(path_external_data_scenario / 'kilometrage.csv', index_col='t')   #annual car mileage in kms/yr, based  mostly  on  Pauliuk  et  al.  (2012a)

    # material compositions (storage) in wt%
    storage_materials = pd.read_csv(path_external_data_standard / 'storage_materials_dynamic.csv',index_col=[0,1]).transpose()  # wt% of total battery weight for various materials, total battery weight is given by the density file above

    # Hydro-dam power capacity (also MW) within 5 regions reported by the IHA (international Hydropwer Association)
    phs_projections = pd.read_csv(path_external_data_standard / 'PHS.csv', index_col='t')   # pumped hydro storage capacity (MW)


    # 2. IMAGE/TIMER files ====================================================================================

    # read TIMER installed storage capacity (MWh, reservoir)
    storage = read_mym_df(path_image_output.joinpath("StorResTot.out"))   #storage capacity in MWh (reservoir, so energy capacity, not power capacity, the latter is used later on in the pumped hydro storage calculations)
        
    #storage capacity in MW (power capacity), to compare it to Pumped hydro storage projections (also given in MW, power capacity)
    storage_power = read_mym_df(path_image_output / 'StorCapTot.out')  

    # Generation capacity (stock & inflow/new) in MW peak capacity, FILES from TIMER
    gcap_data = read_mym_df(path_image_output / 'Gcap.out') # needed to get hydro power for storage
    gcap_data = gcap_data.iloc[:, :26]


    # ----------------------------------------------------------------------------------------------------------
    # ##########################################################################################################
    # %% 2.2) Prepare general variables
    # ##########################################################################################################
    # ----------------------------------------------------------------------------------------------------------

    # Calculations used in both 'vehicle battery storage' and 'other storage'

    ##################
    # Interpolations #
    ##################

    storage = storage.iloc[:, :26]    # drop global total column and empty (27) column

    # J: in high storage scenario the storage demand linearly increases between 2021 and 2050 compared to its original value until it is double by 2050, and then remains constant
    if SENS_ANALYSIS == 'high_stor':
        storage_multiplier = storage
        for year in range(2021,2051):
            storage_multiplier.loc[year] = storage.loc[year] * (1 + (1/30*(year-2020)))
        for year in range(2051,YEAR_END+1):
            storage_multiplier.loc[year] = storage.loc[year] * 2


    # turn index to integer for sorting during the next step
    storage_costs.index = storage_costs.index.astype('int64')
    storage_malus.index = storage_malus.index.astype('int64')
    storage_density.index = storage_density.index.astype('int64')
    storage_lifetime.index = storage_lifetime.index.astype('int64')

    # to interpolate between 2018 and 2030, first create empty rows (NaN values) 
    storage_start = storage_costs.first_valid_index()
    storage_end =   storage_costs.last_valid_index()
    for i in range(storage_start+1,storage_end):
        storage_costs = pd.concat([storage_costs, pd.DataFrame(index=[i])]) #, ignore_index=True
        storage_malus = pd.concat([storage_malus, pd.DataFrame(index=[i])])         # mind: the malus needs to be defined for the same years as the cost indications
        storage_density = pd.concat([storage_density, pd.DataFrame(index=[i])])     # mind: the density needs to be defined for the same years as the cost indications
        storage_lifetime = pd.concat([storage_lifetime, pd.DataFrame(index=[i])])   # mind: the lifetime needs to be defined for the same years as the cost indications
        
    # then, do the actual interpolation on the sorted dataframes                                                    
    storage_costs_interpol = storage_costs.sort_index(axis=0).interpolate(axis=0)#.index.astype('int64')
    storage_malus_interpol = storage_malus.sort_index(axis=0).interpolate(axis=0)
    storage_density_interpol = storage_density.sort_index(axis=0).interpolate(axis=0)  # density calculation continue with the material calculations
    storage_lifetime_interpol = storage_lifetime.sort_index(axis=0).interpolate(axis=0)  # lifetime calculation continue with the material calculations

    # energy density ---
    # fix the energy density (kg/kwh) of storage technologies after 2030
    for year in range(2030+1,YEAR_OUT+1):
        # storage_density_interpol = storage_density_interpol.append(pd.Series(storage_density_interpol.loc[storage_density_interpol.last_valid_index()], name=year))
        row = storage_density_interpol.loc[[storage_density_interpol.last_valid_index()]]
        row.index = [year]
        storage_density_interpol = pd.concat([storage_density_interpol, row])
    # assumed fixed energy densities before 2018
    for year in reversed(range(YEAR_FIRST_GRID,storage_start)): # was YEAR_SWITCH, storage_start
        # storage_density_interpol = storage_density_interpol.append(pd.Series(storage_density_interpol.loc[storage_density_interpol.first_valid_index()], name=year)).sort_index(axis=0)
        row = storage_density_interpol.loc[[storage_density_interpol.first_valid_index()]]
        row.index = [year]
        storage_density_interpol = pd.concat([storage_density_interpol, row]).sort_index(axis=0)

    # storage material intensity ---
    # Interpolate material intensities (dynamic content for gcap & storage technologies between 1926 to 2100, based on data files)
    index = pd.MultiIndex.from_product([list(range(YEAR_FIRST_GRID, YEAR_OUT+1)), list(storage_materials.index)])
    stor_materials_interpol = pd.DataFrame(index=index, columns=storage_materials.columns.levels[1])
    # material intensities for storage
    for cat in list(storage_materials.columns.levels[1]):
        stor_materials_1st   = storage_materials.loc[:,idx[storage_materials.columns[0][0],cat]]
        stor_materials_interpol.loc[idx[YEAR_FIRST_GRID ,:],cat] = stor_materials_1st.to_numpy()  # set the first year (1926) values to the first available values in the dataset (for the year 2000) 
        stor_materials_interpol.loc[idx[storage_materials.columns.levels[0].min(),:],cat] = storage_materials.loc[:, idx[storage_materials.columns.levels[0].min(),cat]].to_numpy() # set the middle year (2000) values to the first available values in the dataset (for the year 2000) 
        stor_materials_interpol.loc[idx[storage_materials.columns.levels[0].max(),:],cat] = storage_materials.loc[:, idx[storage_materials.columns.levels[0].max(),cat]].to_numpy() # set the last year (2100) values to the last available values in the dataset (for the year 2050) 
        stor_materials_interpol.loc[idx[:,:],cat] = stor_materials_interpol.loc[idx[:,:],cat].unstack().astype('float64').interpolate().stack()

    #############
    # Lifetimes #
    #############

    # First the lifetime of storage technologies needs to be defined over time, before running the dynamic stock function
    # before 2018
    for year in reversed(range(YEAR_FIRST_GRID,storage_start)):
        # storage_lifetime_interpol = pd.concat([storage_lifetime_interpol, pd.Series(storage_lifetime_interpol.loc[storage_lifetime_interpol.first_valid_index()], name=year)])
        row = pd.DataFrame([storage_lifetime_interpol.loc[storage_lifetime_interpol.first_valid_index()]])
        storage_lifetime_interpol.loc[year] = row.iloc[0]
    # after 2030
    for year in range(2030+1,YEAR_OUT+1):
        # storage_lifetime_interpol = pd.concat([storage_lifetime_interpol, pd.Series(storage_lifetime_interpol.loc[storage_lifetime_interpol.last_valid_index()], name=year)])
        row = pd.DataFrame([storage_lifetime_interpol.loc[storage_lifetime_interpol.last_valid_index()]])
        storage_lifetime_interpol.loc[year] = row.iloc[0]

    storage_lifetime_interpol = storage_lifetime_interpol.sort_index(axis=0)
    # drop the PHS from the interpolated lifetime frame, as the PHS is calculated separately
    storage_lifetime_interpol = storage_lifetime_interpol.drop(columns=['PHS'])

    #################
    # Market Shares #
    #################

    # Determine MARKET SHARE of the storage capacity using a multi-nomial logit function

    # storage costs ---
    # determine the annual % decline of the costs based on the 2018-2030 data (original, before applying the malus)
    decline = ((storage_costs_interpol.loc[storage_start,:]-storage_costs_interpol.loc[storage_end,:])/(storage_end-storage_start))/storage_costs_interpol.loc[storage_start,:]
    decline_used = decline*storage_ltdecline #TODO: what is happening here? Why?
    # storage_ltdecline is a single number and should describe the long-term decline after 2030 relative to the 2018-2030 decline

    storage_costs_new = storage_costs_interpol * storage_malus_interpol
    # calculate the development from 2030 to 2050 (using annual price decline)
    for year in range(storage_end+1,2050+1):
        # print(year)
        # storage_costs_new = storage_costs_new.append(pd.Series(storage_costs_new.loc[storage_costs_new.last_valid_index()]*(1-decline_used), name=year))
        # storage_costs_new = pd.concat([storage_costs_new, pd.Series(storage_costs_new.loc[storage_costs_new.last_valid_index()] * (1 - decline_used), name=year)])
        row = pd.DataFrame([storage_costs_new.loc[storage_costs_new.last_valid_index()] * (1 - decline_used)])
        storage_costs_new.loc[year] = row.iloc[0]
    # for historic price development, assume 2x AVERAGE annual price decline on all technologies, except lead-acid (so that lead-acid gets a relative price advantage from 1970-2018)
    for year in reversed(range(YEAR_FIRST_GRID,storage_start)):
        # storage_costs_new = storage_costs_new.append(pd.Series(storage_costs_new.loc[storage_costs_new.first_valid_index()]*(1+(2*decline_used.mean())), name=year)).sort_index(axis=0)
        row = pd.DataFrame([storage_costs_new.loc[storage_costs_new.first_valid_index()]*(1+(2*decline_used.mean()))])
        storage_costs_new.loc[year] = row.iloc[0]

    storage_costs_new.sort_index(axis=0, inplace=True) 
    storage_costs_new.loc[1971:2017,'Deep-cycle Lead-Acid'] = storage_costs_new.loc[2018,'Deep-cycle Lead-Acid'] # restore the exception (set to constant 2018 values)


    # market shares ---
    # use the storage price development in the logit model to get market shares
    storage_market_share = MNLogit(storage_costs_new, -0.2) #assumes input of an ordered dataframe with rows as years and columns as technologies, values as prices. Logitpar is the calibrated Logit parameter (usually a nagetive number between 0 and 1)

    # fix the market share of storage technologies before YEAR_START
    for year in range(YEAR_FIRST_GRID,storage_start):
        # storage_market_share = storage_market_share.append(pd.Series(storage_market_share.loc[storage_market_share.last_valid_index()], name=year))
        row = pd.DataFrame([storage_market_share.loc[storage_market_share.first_valid_index()]])
        storage_market_share.loc[year] = row.iloc[0]
    # fix the market share of storage technologies after 2050
    for year in range(2050+1,YEAR_OUT+1):
        # storage_market_share = storage_market_share.append(pd.Series(storage_market_share.loc[storage_market_share.last_valid_index()], name=year))
        row = pd.DataFrame([storage_market_share.loc[storage_market_share.last_valid_index()]])
        storage_market_share.loc[year] = row.iloc[0]
    storage_market_share = storage_market_share.sort_index(axis=0)
    
    # total = storage_market_share.sum(axis=1)
    region_list = list(kilometrage.columns.values)   
    storage.columns = region_list

    #%% 2.4) Hydro Power & Other Storage
    ###########################################################################################################

    # OPTION: NO V2G ---------------------------------------------------------------
    # TODO: this is a temporary solution, until the coupling with the vehicle sector and the battery module calculations are implemented/integrated
    storage_vehicles = pd.DataFrame(0, index=storage.index, columns=storage.columns)  # set vehicle storage to zero when not using V2G
    storage_vehicles = storage_vehicles.loc[:YEAR_OUT]
    #-------------------------------------------------------------------------------

    # Take the TIMER Hydro-dam capacity (MW) & compare it to Pumped hydro capacity (MW) projections from the International Hydropower Association

    Gcap_hydro = gcap_data[['time','DIM_1', 7]].pivot_table(index='time', columns='DIM_1')   # IMAGE-TIMER Hydro dam capacity (power, in MW)
    Gcap_hydro = Gcap_hydro.iloc[:, :26]
    region_list = list(kilometrage.columns.values)  # get a list with region names
    Gcap_hydro.columns = region_list
    Gcap_hydro = Gcap_hydro.loc[:YEAR_OUT]

    # storage capacity in MW (power capacity), to compare it to Pumped hydro storage projections (also given in MW, power capacity)              
    # storage_power.drop(storage_power.iloc[:, -2:], inplace = True, axis = 1) # error prone
    storage_power = storage_power.iloc[:, :26]  
    storage_power.columns = region_list
    storage_power = storage_power.loc[:YEAR_OUT]

    #Disaggregate the Pumped hydro-storgae projections to 26 IMAGE regions according to the relative Hydro-dam power capacity (also MW) within 5 regions reported by the IHA (international Hydropwer Association)
    phs_regions = [[10,11],[19],[1],[22],[0,2,3,4,5,6,7,8,9,12,13,14,15,16,17,18,20,21,23,24,25]]   # subregions in IHA data for Europe, China, US, Japan, RoW, MIND: region refers to IMAGE region MINUS 1
    phs_projections_IMAGE = pd.DataFrame(index=Gcap_hydro.index, columns=Gcap_hydro.columns)        # empty dataframe

    for column in range(0,len(phs_regions)):
        sum_data = Gcap_hydro.iloc[:,phs_regions[column]].sum(axis=1) # first, get the sum of all hydropower in the IHA regions (to divide over in second step)
        for region in range(0,len(IMAGE_REGIONS)):
            if region in phs_regions[column]:
                phs_projections_IMAGE.iloc[:,region] = phs_projections.iloc[:,column] * (Gcap_hydro.iloc[:,region]/sum_data) 
                # J: allocate share of the phs_projections to each IMAGE region based on the share of that region on the generation capacity of the IHA region it is part of
                # J: (Gcap_hydro.iloc[:,region]/sum_data) is between 0 and 1, so the phs_projections are disaggregated to IMAGE regions

    # Then fill the years after 2030 (end of IHA projections) according to the Gcap annual growth rate (assuming a fixed percentage of Hydro dams will be built with Pumped hydro capabilities after )
    if SENS_ANALYSIS == 'high_stor':
        phs_projections_IMAGE.loc[2030:YEAR_OUT] =  phs_projections_IMAGE.loc[2030] * (Gcap_hydro.loc[2030:YEAR_OUT]/Gcap_hydro.loc[2030:YEAR_OUT])  # no growth after 2030 in the high_stor sensitivity variant
    else:
        phs_projections_IMAGE.loc[2030:YEAR_OUT] =  phs_projections_IMAGE.loc[2030] * (Gcap_hydro.loc[2030:YEAR_OUT]/Gcap_hydro.loc[2030])

    # Calculate the fractions of the storage capacity that is provided through pumped hydro-storage, electric vehicles or other storage (larger than 1 means the capacity superseeds the demand for energy storage, in terms of power in MW or enery in MWh) 
    phs_storage_fraction = phs_projections_IMAGE.divide(storage_power.loc[:YEAR_OUT]).clip(upper=1) # the phs storage fraction deployed to fulfill storage demand, both phs & storage_power here are expressed in MW
    storage_remaining = storage.loc[:YEAR_OUT] * (1 - phs_storage_fraction)

    if SENS_ANALYSIS == 'high_stor':
        oth_storage_fraction = 0.5 * storage_remaining 
        oth_storage_fraction += ((storage_remaining * 0.5) - storage_vehicles).clip(lower=0)    
        oth_storage_fraction = oth_storage_fraction.divide(storage).where(oth_storage_fraction > 0, 0).clip(lower=0) 
        evs_storage_fraction = 1 - (phs_storage_fraction + oth_storage_fraction)     # electric vehicle storage (BEV + PHEV) capacity and total storage demand are expressed as MWh
    else: 
        oth_storage_fraction = (storage_remaining - storage_vehicles).clip(lower=0)    
        oth_storage_fraction = oth_storage_fraction.divide(storage.loc[:YEAR_OUT]).where(oth_storage_fraction > 0, 0).clip(lower=0)      
        evs_storage_fraction = 1 - (phs_storage_fraction + oth_storage_fraction)     # electric vehicle storage (BEV + PHEV) capacity and total storage demand are expressed as MWh
    
    checksum = phs_storage_fraction + evs_storage_fraction + oth_storage_fraction   # should be 1 for all fields

    # absolute storage capacity (MWh)
    phs_storage_theoretical = phs_projections_IMAGE.divide(storage_power) * storage.loc[:YEAR_OUT] # ??? theoretically available PHS storage (MWh; fraction * total) only used in the graphs that show surplus capacity
    phs_storage = phs_storage_fraction * storage.loc[:YEAR_OUT]
    evs_storage = evs_storage_fraction * storage.loc[:YEAR_OUT]
    oth_storage = oth_storage_fraction * storage.loc[:YEAR_OUT]

    #output for Main text figure 2 (storage reservoir, in MWh for 3 storage types)
    storage_out_phs = pd.concat([phs_storage], keys=['phs'], names=['type']) 
    storage_out_evs = pd.concat([evs_storage], keys=['evs'], names=['type']) 
    storage_out_oth = pd.concat([oth_storage], keys=['oth'], names=['type']) 
    storage_out = pd.concat([storage_out_phs, storage_out_evs, storage_out_oth])
    # storage_out.to_csv(path_base / 'imagematerials' / 'electricity' / 'out_test'  / 'storage_by_type_MWh.csv')        # in MWh

    # derive inflow & outflow (in MWh) for PHS, for later use in the material calculations 
    PHS_kg_perkWh = 26.8   # kg per kWh storage capacity (as weight addition to existing hydro plants to make them pumped) 
    phs_storage_stock_tail = stock_tail(phs_storage.astype(float), YEAR_OUT)
    storage_lifetime_PHS = storage_lifetime['PHS'].reindex(list(range(YEAR_FIRST_GRID,YEAR_OUT+1)), axis=0).interpolate(limit_direction='both')

    # For now: assume now other storage before 1971 -> TODO: check this
    for year in range(YEAR_FIRST_GRID,YEAR_START):
        oth_storage.loc[year] = 0
    oth_storage = oth_storage.sort_index(axis=0)

    ###########################################################################################################
    #%%% 2.4.1) Prep_data File
    ###########################################################################################################


    # PHS -----------------------------------------------------------------------------------------------------

    phs_stock = phs_storage_stock_tail.copy()

    # Bring dataframes into correct shape for the results_dict

    # stocks: years as index and regions as columns -> years as index and (technology, region) as columns
    # Current columns are regions
    regions = phs_stock.columns.tolist()

    # Create a MultiIndex with technology "PHS" for all columns
    multi_cols = pd.MultiIndex.from_tuples([("PHS", r) for r in regions], names=["technology", "region"])
    # Assign to the DataFrame
    phs_stock.columns = multi_cols

    # lifetimes
    df_mean = storage_lifetime_PHS.copy().to_frame()
    df_stdev = df_mean * STD_LIFETIMES_ELECTR
    df_mean.columns = [(col, 'mean') for col in df_mean.columns] # Rename columns to multi-level tuples
    df_stdev.columns = [(col, 'stdev') for col in df_stdev.columns]
    phs_lifetime_distr = pd.concat([df_mean, df_stdev], axis=1) # Concatenate along columns
    phs_lifetime_distr.index.name = 'year'

    # MIs: (years, material) index and technologies as columns -> years as index and (technology, Material) as columns
    phs_materials = stor_materials_interpol.loc[idx[:,:],'PHS'].unstack() * PHS_kg_perkWh * 1000 # wt% * kg/kWh * 1000 kWh/MWh = kg/MWh
    # Current columns are materials
    materials = phs_materials.columns.tolist()
    # Create a MultiIndex with technology "PHS" for all columns
    multi_cols = pd.MultiIndex.from_tuples([("PHS", m) for m in materials], names=["technology", "material"])
    # Assign to the DataFrame
    phs_materials.columns = multi_cols

    # Conversion table for all coordinates, to be removed/adapted after input tables are fixed.
    conversion_table = {
        "phs_stock": (["Time"], ["Type", "Region"],),
        "phs_materials": (["Cohort"], ["Type", "material"],)
        # "gcap_materials_interpol": (["Cohort"], ["Type", "SubType", "material"], {"Type": ["Type", "SubType"]})
    }

    results_dict = {
            'phs_stock': phs_stock,
            'phs_materials': phs_materials,
            'phs_lifetime_distr': phs_lifetime_distr,
    }

    prep_data_phs = create_prep_data(results_dict, conversion_table, unit_mapping)
    prep_data_phs["stocks"] = prism.Q_(prep_data_phs["stocks"], "MWh")
    prep_data_phs["material_intensities"] = prism.Q_(prep_data_phs["material_intensities"], "kg/MWh")
    prep_data_phs["set_unit_flexible"] = prism.U_(prep_data_phs["stocks"])


    # Other storage--------------------------------------------------------------------------------------------

    oth_storage_stock = oth_storage.copy()

    # Bring dataframes into correct shape for the results_dict

    # stocks: (years, regions) index and technologies as columns -> years as index and (technology, region) as columns
    # Current columns are regions
    regions = oth_storage.columns.tolist()
    # stor_tech = list(storage_lifetime_interpol.columns)
    # Create a MultiIndex with technology "Other Storage" for all columns
    multi_cols = pd.MultiIndex.from_tuples([("Other Storage", r) for r in regions], names=["technology", "region"])
    # multi_cols = pd.MultiIndex.from_product([stor_tech, regions], names=["technology", "region"])
    # Assign to the DataFrame
    oth_storage_stock.columns = multi_cols

    # lifetimes
    df_mean = storage_lifetime_interpol.copy() #.to_frame()
    df_stdev = df_mean * STD_LIFETIMES_ELECTR
    df_mean.columns = [(col, 'mean') for col in df_mean.columns] # Rename columns to multi-level tuples
    df_stdev.columns = [(col, 'stdev') for col in df_stdev.columns]
    oth_storage_lifetime_distr = pd.concat([df_mean, df_stdev], axis=1) # Concatenate along columns
    oth_storage_lifetime_distr.index.name = 'year'

    # MIs: (years, material) index and technologies as columns -> years as index and (technology, Material) as columns
    stor_tech = list(storage_lifetime_interpol.columns)
    # 2nd level of the MultiIndex are materials (1971,'Aluminium')
    oth_storage_materials = stor_materials_interpol.copy()
    oth_storage_materials.index.names = ["year", "material"]  # assign names
    oth_storage_materials = oth_storage_materials.reset_index(level=["year", "material"])
    # Pivot so we get (technology, material) as columns, years as index
    oth_storage_materials = oth_storage_materials.melt(id_vars=["year", "material"], var_name="technology", value_name="value")
    oth_storage_materials = oth_storage_materials.pivot_table(index="year", columns=["technology", "material"], values="value")
    # Ensure proper MultiIndex column names
    oth_storage_materials.columns = pd.MultiIndex.from_tuples(oth_storage_materials.columns, names=["technology", "material"])
    xr_oth_storage_materials = dataset_to_array(pandas_to_xarray(oth_storage_materials, unit_mapping), *(["Cohort"], ["Type", "material"],))
    xr_storage_density_interpol = dataset_to_array(pandas_to_xarray(storage_density_interpol, unit_mapping), *(["Cohort"], ["Type"],))
    oth_storage_materialintens = xr_oth_storage_materials * xr_storage_density_interpol

    # Conversion table for all coordinates, to be removed/adapted after input tables are fixed.
    conversion_table = {
        "oth_storage_stock": (["Time"], ["SuperType", "Region"],),
        "oth_storage_materials": (["Cohort"], ["Type", "material"],), #SubType
        "oth_storage_shares": (["Cohort"], ["Type",]) #SubType
    }
    ## "gcap_materials_interpol": (["Cohort"], ["Type", "SubType", "material"], {"Type": ["Type", "SubType"]})

    results_dict = {
            'oth_storage_stock': oth_storage_stock,
            'oth_storage_materials': oth_storage_materialintens,
            'oth_storage_lifetime_distr': oth_storage_lifetime_distr,
            'oth_storage_shares': storage_market_share
    }

    prep_data_oth_storage = create_prep_data(results_dict, conversion_table, unit_mapping)
    prep_data_oth_storage["stocks"] = prism.Q_(prep_data_oth_storage["stocks"], "MWh")
    prep_data_oth_storage["material_intensities"] = prism.Q_(prep_data_oth_storage["material_intensities"], "kg/kWh")
    prep_data_oth_storage["shares"] = prism.Q_(prep_data_oth_storage["shares"], "share")
    prep_data_oth_storage["set_unit_flexible"] = prism.U_(prep_data_oth_storage["stocks"]) # prism.U_ gives the unit back


    return prep_data_phs, prep_data_oth_storage
