#%% Import modules and constants
import pandas as pd
import numpy as np
from pathlib import Path
import pint
import xarray as xr
from importlib.resources import files

import prism
from imagematerials.read_mym import read_mym_df
from imagematerials.util import dataset_to_array, pandas_to_xarray, convert_lifetime
from imagematerials.concepts import create_electricity_graph, create_region_graph
from imagematerials.electricity.utils import (
    MNLogit, 
    stock_tail, 
    create_prep_data, 
    logistic, 
    quadratic,
    interpolate_xr, 
    add_historic_stock, 
    calculate_grid_growth, 
    calculate_fraction_underground, 
    apply_ce_measures_to_elc, 
    normalize_selected_techs,
    calculate_storage_market_shares
)

from imagematerials.constants import IMAGE_REGIONS

from imagematerials.electricity.constants import (
    STANDARD_SCEN_EXTERNAL_DATA,
    YEAR_FIRST_GRID,
    SENS_ANALYSIS,
    EPG_TECHNOLOGIES,
    STD_LIFETIMES_ELECTR,
    EV_BATTERIES,
    unit_mapping
)

# for ttesting, remove later:
END_YEAR = 2100
INTERMEDIATE_YEAR = 2080


#############################################################################################################
#############################################################################################################

def get_preprocessing_data_stor(path_base: str, climate_policy_config: dict, circular_economy_config: dict, scenario: str, year_start: int, year_end: int, year_out: int):
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
        Base directory of the project where the data structure is located (image-materials/data/raw/).
    scenario : str
        Scenario name (e.g., "SSP2_baseline").
    climate_policy_config : dict
        Dictionary created from a scenario-specific config.toml file.
        Contains all TOML entries ("data_files" mapping) plus a path:
        - "config_file_path": pathlib.Path to the scenario directory (e.g. SSP2_baseline) containing config.toml and all referenced
        output subfolders. 
        Used to construct full paths to scenario output files, e.g. read_mym_df(climate_policy_config["config_file_path"]/climate_policy_config["data_files"]["variable_x"]).
    year_start : int
        First simulation year for data processing (e.g., 1971).
    year_end : int
        Last year for data interpolation (used for intermediate computations).
    year_out : int
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

    # path_image_output = Path(path_base, "data", "raw", "image", scen_folder, "EnergyServices")
    path_external_data_standard = Path(path_base, "electricity", "standard_data")
    path_external_data_scenario = Path(path_base, "electricity", scenario) #test

    # test if path_external_data_scenario exists and if not set to standard scenario
    if not path_external_data_scenario.exists():
        path_external_data_scenario = Path(path_base, "electricity", STANDARD_SCEN_EXTERNAL_DATA)

    # print(f"Path to image output: {path_image_output}")
    # assert path_image_output.is_dir()
    # assert path_external_data_standard.is_dir()
    # assert path_external_data_scenario.is_dir()

    idx = pd.IndexSlice   

    #----------------------------------------------------------------------------------------------------------
    ###########################################################################################################
    #%%% 2.1) Read in files
    ###########################################################################################################
    #----------------------------------------------------------------------------------------------------------


    # 1. External Data ======================================================================================== 

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
    storage_materials = pd.read_csv(path_external_data_standard / 'storage_materials_dynamic.csv',index_col=[0,1],usecols=lambda col: col != "unit").transpose()  # wt% of total battery weight for various materials, total battery weight is given by the density file above

    # Hydro-dam power capacity (also MW) within 5 regions reported by the IHA (international Hydropwer Association)
    phs_projections = pd.read_csv(path_external_data_standard / 'PHS.csv', index_col='t')   # pumped hydro storage capacity (MW)


    # 2. IMAGE/TIMER files ====================================================================================

    # read TIMER installed storage capacity (MWh, reservoir)
    # storage = read_mym_df(path_image_output.joinpath("StorResTot.out"))   #storage capacity in MWh (reservoir, so energy capacity, not power capacity, the latter is used later on in the pumped hydro storage calculations)
    storage = read_mym_df(climate_policy_config["config_file_path"] / climate_policy_config["data_files"]['StorResTot'])

    #storage capacity in MW (power capacity), to compare it to Pumped hydro storage projections (also given in MW, power capacity)
    # storage_power = read_mym_df(path_image_output / 'StorCapTot.out')  
    storage_power = read_mym_df(climate_policy_config["config_file_path"] / climate_policy_config["data_files"]['StorCapTot'])

    # Generation capacity (stock & inflow/new) in MW peak capacity, FILES from TIMER
    gcap_data = read_mym_df(climate_policy_config["config_file_path"] / climate_policy_config["data_files"]['GCap']) # needed to get hydro power for storage
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
        for year in range(2051,year_end+1):
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
    for year in range(2030+1,year_out+1):
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
    index = pd.MultiIndex.from_product([list(range(YEAR_FIRST_GRID, year_out+1)), list(storage_materials.index)])
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
    for year in reversed(range(year_start,storage_start)): # TODO: use YEAR_FIRST_GRID instead? -> hen possible to run from earlier on?
        # storage_lifetime_interpol = pd.concat([storage_lifetime_interpol, pd.Series(storage_lifetime_interpol.loc[storage_lifetime_interpol.first_valid_index()], name=year)])
        row = pd.DataFrame([storage_lifetime_interpol.loc[storage_lifetime_interpol.first_valid_index()]])
        storage_lifetime_interpol.loc[year] = row.iloc[0]
    # after 2030
    for year in range(2030+1,year_out+1):
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
    for year in reversed(range(year_start,storage_start)):
        # storage_costs_new = storage_costs_new.append(pd.Series(storage_costs_new.loc[storage_costs_new.first_valid_index()]*(1+(2*decline_used.mean())), name=year)).sort_index(axis=0)
        row = pd.DataFrame([storage_costs_new.loc[storage_costs_new.first_valid_index()]*(1+(2*decline_used.mean()))])
        storage_costs_new.loc[year] = row.iloc[0]
        storage_costs_new.sort_index(axis=0, inplace=True) 

    storage_costs_new.sort_index(axis=0, inplace=True) 
    storage_costs_new.loc[1971:2017,'Deep-cycle Lead-Acid'] = storage_costs_new.loc[2018,'Deep-cycle Lead-Acid'] # restore the exception (set to constant 2018 values)


    # market shares ---
    # use the storage price development in the logit model to get market shares
    storage_market_share = MNLogit(storage_costs_new, -0.2) #assumes input of an ordered dataframe with rows as years and columns as technologies, values as prices. Logitpar is the calibrated Logit parameter (usually a nagetive number between 0 and 1)

    # fix the market share of storage technologies after 2050
    for year in range(2050+1,year_out+1):
        # storage_market_share = storage_market_share.append(pd.Series(storage_market_share.loc[storage_market_share.last_valid_index()], name=year))
        row = pd.DataFrame([storage_market_share.loc[storage_market_share.last_valid_index()]])
        storage_market_share.loc[year] = row.iloc[0]
    # fix the market share of storage technologies before YEAR_START
    for year in range(YEAR_FIRST_GRID,year_start):
        # storage_market_share = storage_market_share.append(pd.Series(storage_market_share.loc[storage_market_share.last_valid_index()], name=year))
        row = pd.DataFrame([storage_market_share.loc[storage_market_share.first_valid_index()]])
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
    storage_vehicles = storage_vehicles.loc[:year_out]
    #-------------------------------------------------------------------------------

    # Take the TIMER Hydro-dam capacity (MW) & compare it to Pumped hydro capacity (MW) projections from the International Hydropower Association

    Gcap_hydro = gcap_data[['time','DIM_1', 7]].pivot_table(index='time', columns='DIM_1')   # IMAGE-TIMER Hydro dam capacity (power, in MW)
    Gcap_hydro = Gcap_hydro.iloc[:, :26]
    region_list = list(kilometrage.columns.values)  # get a list with region names
    Gcap_hydro.columns = region_list
    Gcap_hydro = Gcap_hydro.loc[:year_out]

    # storage capacity in MW (power capacity), to compare it to Pumped hydro storage projections (also given in MW, power capacity)              
    # storage_power.drop(storage_power.iloc[:, -2:], inplace = True, axis = 1) # error prone
    storage_power = storage_power.iloc[:, :26]  
    storage_power.columns = region_list
    storage_power = storage_power.loc[:year_out]

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
        phs_projections_IMAGE.loc[2030:year_out] =  phs_projections_IMAGE.loc[2030] * (Gcap_hydro.loc[2030:year_out]/Gcap_hydro.loc[2030:year_out])  # no growth after 2030 in the high_stor sensitivity variant
    else:
        phs_projections_IMAGE.loc[2030:year_out] =  phs_projections_IMAGE.loc[2030] * (Gcap_hydro.loc[2030:year_out]/Gcap_hydro.loc[2030])

    # Calculate the fractions of the storage capacity that is provided through pumped hydro-storage, electric vehicles or other storage (larger than 1 means the capacity superseeds the demand for energy storage, in terms of power in MW or enery in MWh) 
    phs_storage_fraction = phs_projections_IMAGE.divide(storage_power.loc[:year_out]).clip(upper=1) # the phs storage fraction deployed to fulfill storage demand, both phs & storage_power here are expressed in MW
    storage_remaining = storage.loc[:year_out] * (1 - phs_storage_fraction) # asumption here (?): share in MW = share in GHh (not really true, though since both are very high for PHS for early years, this might be okey)

    phs_storage = storage.loc[:year_out] * phs_storage_fraction
    oth_storage = storage_remaining 


    # if SENS_ANALYSIS == 'high_stor':
    #     oth_storage_fraction = 0.5 * storage_remaining 
    #     oth_storage_fraction += ((storage_remaining * 0.5) - storage_vehicles).clip(lower=0)    
    #     oth_storage_fraction = oth_storage_fraction.divide(storage).where(oth_storage_fraction > 0, 0).clip(lower=0) 
    #     evs_storage_fraction = 1 - (phs_storage_fraction + oth_storage_fraction)     # electric vehicle storage (BEV + PHEV) capacity and total storage demand are expressed as MWh
    # else: 
    #     oth_storage_fraction = (storage_remaining - storage_vehicles).clip(lower=0)    
    #     oth_storage_fraction = oth_storage_fraction.divide(storage.loc[:year_out]).where(oth_storage_fraction > 0, 0).clip(lower=0)      
    #     evs_storage_fraction = 1 - (phs_storage_fraction + oth_storage_fraction)     # electric vehicle storage (BEV + PHEV) capacity and total storage demand are expressed as MWh
    
    # checksum = phs_storage_fraction + evs_storage_fraction + oth_storage_fraction   # should be 1 for all fields

    # # absolute storage capacity (MWh)
    # phs_storage_theoretical = phs_projections_IMAGE.divide(storage_power) * storage.loc[:year_out] # ??? theoretically available PHS storage (MWh; fraction * total) only used in the graphs that show surplus capacity
    # phs_storage = phs_storage_fraction * storage.loc[:year_out]
    # evs_storage = evs_storage_fraction * storage.loc[:year_out]
    # oth_storage = oth_storage_fraction * storage.loc[:year_out]

    #output for Main text figure 2 (storage reservoir, in MWh for 3 storage types)
    # storage_out_phs = pd.concat([phs_storage], keys=['phs'], names=['type']) 
    # storage_out_evs = pd.concat([evs_storage], keys=['evs'], names=['type']) 
    # storage_out_oth = pd.concat([oth_storage], keys=['oth'], names=['type']) 
    # storage_out = pd.concat([storage_out_phs, storage_out_evs, storage_out_oth])
    # storage_out.to_csv(path_base /  'electricity' / 'test'  / 'storage_by_type_MWh.csv')        # in MWh
    # storage_frac = pd.concat([phs_storage_fraction, evs_storage_fraction, oth_storage_fraction])
    # storage_frac.to_csv(path_base /  'electricity' / 'test'  / 'storage_by_type_fraction.csv')

    # derive inflow & outflow (in MWh) for PHS, for later use in the material calculations 
    PHS_kg_perkWh = 26.8   # kg per kWh storage capacity (as weight addition to existing hydro plants to make them pumped) 
    phs_storage_stock_tail = stock_tail(phs_storage.astype(float), year_out)
    storage_lifetime_PHS = storage_lifetime['PHS'].reindex(list(range(YEAR_FIRST_GRID,year_out+1)), axis=0).interpolate(limit_direction='both')

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
    prep_data_oth_storage["knowledge_graph_elc"] = create_electricity_graph()
    prep_data_oth_storage["set_unit_flexible"] = prism.U_(prep_data_oth_storage["stocks"]) # prism.U_ gives the unit back


    # change region names to IMAGE_REGIONS # TODO: this should be done in hte beginning of preprocessing
    knowledge_graph_region =            create_region_graph()
    prep_data_phs["stocks"] =           knowledge_graph_region.rebroadcast_xarray(prep_data_phs["stocks"], output_coords=IMAGE_REGIONS, dim="Region")
    prep_data_oth_storage["stocks"] =   knowledge_graph_region.rebroadcast_xarray(prep_data_oth_storage["stocks"], output_coords=IMAGE_REGIONS, dim="Region")

    # Have both stocks and stocks_non_phs in the prep_data_oth_storage. In case vehicle-to-grid (V2G) is considered and the ev battery + Link module is added
    # to the joined model run, stocks_non_phs is used and stocks is replaced with the remaining storage demand after subtracting the EV battery storage. In 
    # case V2G is not considered, stocks is used directly and represents the storage demand fulfilled by dedicated grid storage technologies (non-PHS).
    prep_data_oth_storage["stocks_non_phs"] = prep_data_oth_storage["stocks"].copy()

    return prep_data_phs, prep_data_oth_storage

