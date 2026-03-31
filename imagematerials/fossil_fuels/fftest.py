 
#%% Outline of the pre-processing steps for fossil fuels (coal, oil, gas) ---------------------------------------------------------------------------------------------------------------------------------

# 1: Import FUMA 
# a Run to create capacity files

# 2: Import module and constants
# a General imports/packages
# b Image material imports
# c Set scenario 
# d Make variable for pre-processing data 
# e Make variable for path to external data for the scenario, and test if it exists, if not set to standard scenario

# 3: Read in files for the extraction stage for each fuel (coal, oil, gas)
# a lifetimes 
# b material intensities (steel, alumnium, copper?, I could add other ones?)
# C Import capacity files that are made from the stock calculation (FUMA) output (for each fuel: coal, oil, gas) which is labelled as fuma.py and are found under stock_calculation/output/SSP1_ML (or the relevant scenario)
# # d Transform each to xarray
# e add units using the knowledge graph
# f expand materials and lifteimes to all years (2020-2100)

# 4: Read in files for the processing stage for each fuel
# a lifetimes 
# b material intensities
# C Import capacity files that are made from the stock calculation (FUMA) output (for each fuel: coal, oil, gas) which is labelled as fuma.py and are found under stock_calculation/output/SSP1_ML (or the relevant scenario)
# d Transform each to xarray and add units using the knowledge graph and expand materials and lifteimes to all years (2020-2100)

# 5: Read in files for the transport stage for each fuel
# a lifetimes 
# b material intensities
# C Import capacity files that are made from the stock calculation (FUMA) output (for each fuel: coal, oil, gas) which is labelled as fuma.py and are found under stock_calculation/output/SSP1_ML (or the relevant scenario)
# d Transform each to xarray
# e add units using the knowledge graph
# f expand materials and lifteimes to all years (2020-2100)

# 6: Read in files for the pipelines stage for each fuel
# a lifetimes 
# b material intensities
# C Import capacity files that are made from the stock calculation (FUMA) output (for each fuel: coal, oil, gas) which is labelled as fuma.py and are found under stock_calculation/output/SSP1_ML (or the relevant scenario)
# d Transform each to xarray
# e add units using the knowledge graph
# f expand materials and lifteimes to all years (2020-2100)

# 8 Make prep_data dictionary with all the pre-processed data, and return it

# 9 Adjust as necessary to fit into the IMAGE framework (so its runs with ELMA)

# 10 Make knowledge graph for fossil fuel so that figure can combine many types of infrastructure for each fuel type

# 11 Make output figures 


#%% Import FUMA model 
# A1) FUMA MODEL 

import pandas as pd
import numpy as np
import os
from pathlib import Path

# SET SCENARIO HERE (Select from SSP1_delayed_action, SSP1_climate_policy, SSP2, SSP2_2D, SSP2_delayed_action, SSP2_climate_policy, SSP2_climate_policy_resource_efficiency SSP3_no_policy)
scenario = "SSP3_no_policy"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "stock_calculation" / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "fossil_fuels" / "Scenario_data"

(OUTPUT_DIR / scenario).mkdir(parents=True, exist_ok=True)
print(f"Output directory ready: {OUTPUT_DIR / scenario}")

print(f"Base directory: {BASE_DIR}")
print(f"Data directory: {DATA_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

from stock_calculation.attributes import constants
from stock_calculation.calculations.materials import cohorts_to_materials_typical_np
from stock_calculation.calculations.pipelines import pipelines_indexed_stock_growth
from stock_calculation.calculations.transport import transport
from stock_calculation.calculations.extraction import extraction_stocks
from stock_calculation.calculations.storage import storage_indexed_stocks_growth
from stock_calculation.calculations.refinery import refinery_stocks
from stock_calculation.prepare_output_files import merge_material, merge_total
from stock_calculation.mym.read_mym import read_mym_df
from stock_calculation.stock_model.inflow_outflow_models import (
    inflow_outflow_typical_np,
    inflow_outflow_dynamic_np,
    inflow_outflow_surplus,
    inflow_outflow_surplus_typical_np,
    inflow_outflow_surplus_int,
)

idx = pd.IndexSlice
# Automatically set working directory to the script's location
# script_dir = Path(__file__).resolve().parent
# os.chdir(script_dir)
# DATA_DIR = Path("data")
# print(f" Working directory set to: {script_dir}")

ref_frac = 0.8  #  refineryRefining capacity used

# A2) Load input-data files

# IMAGE drivers
import os

primpersec = read_mym_df(
    os.path.join(
        "stock_calculation",
        "data",
        "T2ReMo_may23",
        scenario,
        "PrimPerSec.out"
    )
)
  # primary enery consumption per sector in GJ
final_energy = read_mym_df(
    os.path.join(
        "stock_calculation",
        "data",
        "T2ReMo_may23",
        scenario,
        "final_energy_rt.out"
    )
)  

# final enery consumption per sector in PJ
mining_type_share_coal = pd.read_csv(
    DATA_DIR / "coal_mine_type_share.csv",
    index_col=[0],
)

# % of open-cast vs. underground mining
extraction_type_share_gas = pd.read_csv(
    DATA_DIR /"gas_extraction_type_share.csv", index_col=[0]
)  # % of offshore vs. onshore extraction
extraction_type_share_oil = pd.read_csv(
    DATA_DIR /"oil_extraction_type_share.csv", index_col=[0]
)  # % of offshore vs. onshore extraction

# Pipeline data
gas_pipeline_length = pd.read_csv(
    DATA_DIR /"natural_gas_pipeline_length_km_new.csv", index_col=[0]
)  # Regional Gas Pipeline length (km) for the year 2019
oil_pipeline_length = pd.read_csv(
    DATA_DIR /"oil_pipeline_length_km_new.csv", index_col=[0]
)  # Regional Oil Pipeline length (km) for the year 2019

on_offshore_oil_pipeline = pd.read_csv(
    DATA_DIR /"onshore_offshore_oil_pipeline.csv", index_col=[0]
)  # Share of 2 different pipeline types (onshore & offshore) global (%)

# Storage data
oil_storage_capacity = pd.read_csv(
    DATA_DIR /"oil_storage_capacity_million_barrels_new.csv", index_col=[0]
)  # Regional Oil storage capacity (million barrels, mb)
oil_storage_materials = pd.read_csv(
    DATA_DIR /"material_intensity/oil_strorage_material_intensity_kg_per_m3.csv",
    index_col=[0],
)  # Material intensity of oil regional oil distribution (kg material per m3 of storage capacity)
# transport vehicle data
transport_demand_domestic = pd.read_csv(
    DATA_DIR /"Transport_demand_domestic_tkm_per_kg.csv",
    index_col=[0],
    sep=";",
).drop(
    columns="Pipeline"
)  # Domestic transport demand per fuel type (tkm/kg, MIND: in the case of gas it is given in tkm/m3)
transport_demand_internat = pd.read_csv(
    DATA_DIR /"Transport_demand_international_tkm_per_kg.csv",
    index_col=[0],
    sep=";",
).drop(
    columns="Pipeline"
)  # International transport demand per fuel type(tkm/kg, MIND: in the case of gas it is given in tkm/m3)
transport_demand_total = (
    transport_demand_domestic + transport_demand_internat
)

transport_materials_per_vehicle = pd.read_csv(
    DATA_DIR /"material_intensity/fossil_fuel_freight_vehicle_material_fraction.csv",
    index_col=[0],
).T  # % of the total weight by material in fossil fuel freight transport vehicles
transport_vehicle_weight_per_tkm = pd.read_csv(
    DATA_DIR /"material_intensity/fossil_fuel_freight_vehicle_weight_intensity_g_per_tkm.csv",
    index_col=[0],
    header=None,
    sep=";",
).squeeze()  # average intensity of fossil fuel freight transport vehicles weight (g) in stock per tkm (g/tkm) per year so the stock required to continuously provide the service of transportation

# Mining data & material intensities (in kg material per unit of annual production capacity)
production_intensity_coal = pd.read_csv(
    DATA_DIR /"material_intensity/coal_production_intensity.csv", index_col=[0]
)  # material use per unit of production capacity (kg/kg yr-1)
production_intensity_gas = pd.read_csv(
    DATA_DIR /"material_intensity/Natural_gas_production_intensity.csv",
    index_col=[0],
)  # material use per unit of production capacity (kg/m3 yr-1)
production_intensity_oil = pd.read_csv(
    DATA_DIR /"material_intensity/oil_production_intensity.csv", index_col=[0]
)

# Processing & refinery material intensities (in kg material per unit of annual production capacity)
processing_intensity_gas = pd.read_csv(
    DATA_DIR /"material_intensity/gas_processing_plant_intensity_kg_per_m3.csv",
    index_col=[0],
)
refinery_intensity_oil = pd.read_csv(
    DATA_DIR /"material_intensity/oil_refinery_material_intensity_kg_per_kg.csv",
    index_col=[0],
)
preparation_intensity_coal = pd.read_csv(
    DATA_DIR /"material_intensity/coal_preparation_material_intensity_kg_per_kg.csv",
    index_col=[0],
)
# Pipeline material intensity data (kg/km)
gas_pipeline_composition_trans = pd.read_csv(
    DATA_DIR /"material_intensity/gas_pipeline_composition_transmission_critical_kg_per_km.csv",
    index_col=[0],
)  # material use (kg) per km of transmission pipeline (kg/km)
gas_pipeline_composition_distr = pd.read_csv(
    DATA_DIR /"material_intensity/gas_pipeline_composition_distribution_kg_per_km.csv",
    index_col=[0],
)  # material use (kg) per km of distribution pipeline (kg/km)
oil_pipeline_composition = pd.read_csv(
    DATA_DIR /"material_intensity/oil_pipeline_composition_kg_per_km.csv",
    index_col=[0],
)

# infrastructure lifetime assumptions (average lifetime in years)
lifetimes = pd.read_csv(DATA_DIR /"lifetime.csv", index_col=[0])

# A3) Prepare data files & formatting

# adding index names
primpersec.set_index(
    ["time", "DIM_1", "DIM_2"], inplace=True
)  # index: time, 11 primary fuels, 8 secondary fuels
final_energy.set_index(
    ["time", "DIM_1", "DIM_2"], inplace=True
)  # index: time, 28 regions, 6 sectors, columns: 10 secondary fuel types (region 27 is ignored)


# interpolation of data files
def interpolate_pipeline_data(
    original,
):  # original file has 2 level multi-columns (type & region) & time as a single index
    new = original.reindex(
        list(range(constants.START_YEAR, constants.END_YEAR + 1))
    ).interpolate(direction="both")
    first_known_index = original.loc[original.first_valid_index(), :].name
    for year in range(constants.START_YEAR, first_known_index):
        new.loc[year, :] = original.loc[
            first_known_index, :
        ]  # extend first known value to startyear
    return new


production_intensity_weight_coal = production_intensity_coal.sum(axis=1)
preparation_intensity_weight_coal = preparation_intensity_coal.sum(axis=1)
production_intensity_weight_gas = production_intensity_gas.sum(axis=1)
production_intensity_weight_oil = production_intensity_oil.sum(axis=1)
processing_intensity_weight_gas = processing_intensity_gas.sum(axis=1)
refinery_intensity_weight_oil = refinery_intensity_oil.sum(axis=1)

# A4) Model drivers (fossil fuel use scenarios from IMAGE, by region towards 2100)
#   For now, total regional consumption (as primary or final energy) is the driver of infrastructure requirements
#   In the future, this may be expanded

time_dimension = [
    x for x in range(constants.START_YEAR, constants.END_YEAR + 1, 1)
]
regions_counter = [x for x in range(1, constants.NUM_REGIONS + 1, 1)]

# Add labels
primary_fuel_types_ff = ["coal", "conv oil", "unconv oil", "gas"]  # 1st 4

final_fuel_types1 = [
    "Solids",
    "Liquids1",
    "Sec gas",
    "Hydrogen",
    "Sec mod biomass",
    "Sec heat",
    "Sec trad biomass",
    "Electricity",
    "Liquids2",
]
final_fuel_types_ff1 = [
    "Solids",
    "Liquids1",
    "Sec gas",
    "Hydrogen",
    "Sec heat",
    "Electricity",
    "Liquids2",
]

final_fuel_types2 = [
    "Solids",
    "Liquids1",
    "Liquids2",
    "Sec gas",
    "Sec mod biomass",
    "Sec trad biomass",
    "Hydrogen",
    "Sec heat",
    "Electricity",
]  # 1st 9, last one is total
final_fuel_types_ff2 = [
    "Solids",
    "Liquids1",
    "Liquids2",
    "Sec gas",
]  # 1st 9, last one is total

sectors_ff = [
    "industry",
    "transport",
    "residential",
    "services",
    "other",
    "non-energy",
    "bunkers",
]  # 1st 7, last one is total

index_pps = pd.MultiIndex.from_product(
    [time_dimension, primary_fuel_types_ff, final_fuel_types_ff1],
    names=["time", "primary fuels", "secondary fuels"],
)
index_fin = pd.MultiIndex.from_product(
    [time_dimension, regions_counter, sectors_ff],
    names=["time", "regions", "sectors"],
)

# select fossil fuel flows only
fossil_flows_primary = pd.DataFrame(
    primpersec.loc[
        idx[time_dimension, [1, 2, 3, 4], [1, 2, 3, 4, 6, 8, 9]],
        regions_counter,
    ].to_numpy(),
    index=index_pps,
    columns=regions_counter,
)
fossil_flows_final = pd.DataFrame(
    final_energy.loc[
        idx[time_dimension, regions_counter, [1, 2, 3, 4, 5, 6, 7]],
        [1, 2, 3, 4],
    ].to_numpy(),
    index=index_fin,
    columns=final_fuel_types_ff2,
)

# sum 2 liquid types as a single "liquids" category (both in primary and final energy)
fossil_flows_final["oil"] = fossil_flows_final[
    ["Liquids1", "Liquids2"]
].sum(
    axis=1
)  # new column as sum of 2 liquid types
fossil_flows_final = fossil_flows_final[
    ["Solids", "Sec gas", "oil"]
].rename(
    columns={"Solids": "coal", "Sec gas": "gas"}
)  # Remove original data for liquids & Rename to oil/gas/coal

liquid_flows_primary = (
    fossil_flows_primary.loc[idx[:, :, ["Liquids1", "Liquids2"]], :]
    .groupby(level=[0, 1])
    .sum()
)
liquid_flows_primary = pd.concat(
    {"Liquids": liquid_flows_primary}, names=["secondary fuels"]
).reorder_levels(
    ["time", "primary fuels", "secondary fuels"]
)  # add multi-index level for concatenation

fossil_flows_primary = pd.concat(
    [fossil_flows_primary, liquid_flows_primary]
)
fossil_flows_primary = fossil_flows_primary.drop(
    ["Liquids1", "Liquids2"], level="secondary fuels"
).sort_index(
    level=[0, 1, 2]
)  # and remove the 'old' conv & unconv oil multi-index level

# sum unconventional & conventional oil as oil (only for primary energy)
oil_flows_primary = (
    fossil_flows_primary.loc[idx[:, ["conv oil", "unconv oil"], :], :]
    .groupby(level=[0, 2])
    .sum()
)  # sum of conv & unconv oil
oil_flows_primary = pd.concat(
    {"oil": oil_flows_primary}, names=["primary fuels"]
).reorder_levels(
    ["time", "primary fuels", "secondary fuels"]
)  # add multi-index level for concatenation

fossil_flows_primary = pd.concat(
    [fossil_flows_primary, oil_flows_primary]
)  # then, append to all fossil flows (primary)
fossil_flows_primary = fossil_flows_primary.drop(
    ["conv oil", "unconv oil"], level="primary fuels"
).sort_index(
    level=[0, 1]
)  # and remove the 'old' conv & unconv oil multi-index level

# Select only the primary flows that are consumed as primary fuels (mind: X to Liquid is also consumed but added later, for the sankey)
consumed_primary = ["Electricity", "Hydrogen", "Sec heat"]
used_as_primary = fossil_flows_primary.loc[idx[:, :, consumed_primary], :]
used_as_final = (
    fossil_flows_final.stack()
    .unstack(level=1)
    .reorder_levels([0, 2, 1])
    .sort_index()
    * constants.PJ_TO_GJ
)  # stack same as primary

# Then, combine two dataframes and use a combined index name
all_fossil_flows = pd.concat([used_as_final, used_as_primary]).sort_index(
    level=[0, 1, 2]
)


def select_fuel(fuel):
    fuel_use_total = (
        all_fossil_flows.loc[idx[:, fuel, :], :].groupby(level=[0]).sum()
    )  # GJ
    return fuel_use_total

# Some gas/coal to liquids happens in TIMER - In the sankey we keep track of the size of the flow
# HOWEVER! For now we work from the assumption that the liquids consumed are supplied fully by the oil supply chain (ignoring some of the exchange/substitution between supply chains)
# Therefore, we include it as one of the consumption categories of primary energy only in the sankey, not in the calculations (for now)
primary_x_to_liquid = fossil_flows_primary.loc[
    idx[:, ["gas", "coal"], "Liquids"], :
]
all_fossil_flows_sankey = pd.concat(
    [all_fossil_flows, primary_x_to_liquid]
).sort_index(level=[0, 1, 2])

# (Intermediate results) Fuel use data

all_fossil_flows_sankey.to_csv(
    OUTPUT_DIR / scenario / "fuel_use_GJ.csv", index=True
)

select_year = 2019

# A5) Generate/prepend historic fossil fuel consumption profile (to avoid inflow shocks in the first year of 'stock-driven' dynamic stock calculations)
first_production_year = {"coal": 1880, "oil": 1900, "gas": 1920}
stabilisation_year = {"coal": 1920, "oil": 1970, "gas": 1970}


# add historic timeseries based on initial/first year of production, with the possibility to include a stabilisation period before the last know year of data
def generate_historic_profile(fuel_use_total, fuel):

    year_list = list(
        range(first_production_year[fuel], constants.START_YEAR)
    )
    prepend = pd.DataFrame(
        np.nan,
        index=year_list,
        columns=list(range(1, constants.NUM_REGIONS + 1)),
    )

    # set first year to 0 & latest unknown year to latest known year
    prepend.loc[year_list[0]] = float(0)
    prepend.loc[year_list[-1]] = fuel_use_total.loc[constants.START_YEAR]

    # if the stabilisation year is within the time series, then extend the last know year first
    if stabilisation_year[fuel] < year_list[-1]:
        for year in range(stabilisation_year[fuel], year_list[-1]):
            prepend.loc[year] = fuel_use_total.loc[constants.START_YEAR]

    prepend = prepend.interpolate(method="index")
    combine = pd.concat([prepend, fuel_use_total])

    return combine


# ###################################################################################################
#   B) SUPPLY CHAIN STOCK CALCULATIONS:
#   Supply Chain Elements:          Coal    Oil     Gas
#   B1) Extraction infrastructure     X       X       X
#   B2) transport by vehicles         X       X       X
#   B3) transport by pipelines                X       X
#   B4) Storage                               X       x
#   B5) Processing & Refining         x       X       X
#   B1 - B5 are seperate files in the calculation folder
#   B6) Run all supply chain steps
########################################################################################################
# B6) Run supply chain & infrastructure stock models for three fossil fuel types


def coal_infra():
    fuel = "coal"
    coal_use_total = generate_historic_profile(
        select_fuel(fuel), fuel
    )  # in GJ
    coal_use_total = coal_use_total.rolling(
        window=10, min_periods=2, center=True
    ).mean()  # coal_use_total_ma moving average of coal_use_total
    extraction_stock_coal = extraction_stocks(
        fuel,
        (coal_use_total * constants.GJ_TO_KG_COAL),
        production_intensity_weight_coal,
        mining_type_share_coal,
    )  # in kg (total weight of infrastructure)
    preparation_stock_coal = refinery_stocks(
        fuel,
        (coal_use_total * constants.GJ_TO_KG_COAL),
        preparation_intensity_weight_coal,
    )  # in kg (total weight of infrastructure)
    transport_total_coal = transport(
        fuel,
        coal_use_total * constants.GJ_TO_KG_COAL,
        transport_demand_total,
        transport_vehicle_weight_per_tkm,
    )  # in kg
    return (
        extraction_stock_coal,
        preparation_stock_coal,
        transport_total_coal,
    )

#Comment back in if running for a new scenario, otherwise it will overwrite the existing files with the same name (since the input data is the same)
# coalresults = coal_infra()
# coalresults[0].to_csv(OUTPUT_DIR / scenario / "coal_extraction_stock_kg.csv")
# coalresults[1].to_csv(OUTPUT_DIR / scenario / "coal_preparation_stock_kg.csv")
# coalresults[2].to_csv(OUTPUT_DIR / scenario / "coal_transport_stock_kg.csv")


def gas_infra():
    fuel = "gas"
    gas_use_total = generate_historic_profile(
        select_fuel(fuel), fuel
    )  # in GJ
    gas_use_total = gas_use_total.rolling(
        window=10, min_periods=2, center=True
    ).mean()
    baseyear = (
        constants.BASE_YEAR
    )  # year for which current stock is determined
    extraction_stock_gas = extraction_stocks(
        fuel,
        (gas_use_total * constants.GJ_TO_M3_GAS),
        production_intensity_weight_gas,
        extraction_type_share_gas,
    )  # in kg (total weight of infrastructure)

    transport_total_gas = transport(
        fuel,
        gas_use_total * constants.GJ_TO_M3_GAS,
        transport_demand_total,
        transport_vehicle_weight_per_tkm,
    )  # in kg
    gas_pipelines = pipelines_indexed_stock_growth(
        fuel, gas_use_total, gas_pipeline_length, baseyear
    )  # in km
    processing_stock_gas = refinery_stocks(
        fuel,
        (gas_use_total * constants.GJ_TO_M3_GAS),
        processing_intensity_weight_gas,
    )
    return (
        extraction_stock_gas,
        transport_total_gas,
        gas_pipelines,
        processing_stock_gas,
    )
#Comment back in if running for a new scenario, otherwise it will overwrite the existing files with the same name (since the input data is the same)
# gasresults = gas_infra()
# gasresults[0].to_csv(OUTPUT_DIR / scenario / "gas_extraction_stock_kg.csv")
# gasresults[1].to_csv(OUTPUT_DIR / scenario / "gas_transport_stock_kg.csv")
# gasresults[2].to_csv(OUTPUT_DIR / scenario / "gas_pipelines_length_km.csv")
# gasresults[3].to_csv(OUTPUT_DIR / scenario / "gas_processing_stock_kg.csv")


def oil_infra():
    fuel = "oil"
    oil_use_total = generate_historic_profile(
        select_fuel(fuel), fuel
    )  # in GJ
    oil_use_total = oil_use_total.rolling(
        window=10, min_periods=2, center=True
    ).mean()
    baseyear = (
        constants.BASE_YEAR
    )  # year for which current stock is determined
    extraction_stock_oil = extraction_stocks(
        fuel,
        (oil_use_total * constants.GJ_TO_KG_OIL),
        production_intensity_weight_oil,
        extraction_type_share_oil,
    )  # in kg (total weight of infrastructure)
    transport_total_oil = transport(
        fuel,
        oil_use_total * constants.GJ_TO_KG_OIL,
        transport_demand_total,
        transport_vehicle_weight_per_tkm,
    )  # in kg
    oil_pipelines = pipelines_indexed_stock_growth(
        fuel, oil_use_total, oil_pipeline_length, baseyear
    )  # in km
    oil_storage = storage_indexed_stocks_growth(
        fuel,
        oil_use_total,
        oil_storage_capacity * constants.MB_TO_M3,
        baseyear,
    )  # in m3
    refinery_stock_oil = refinery_stocks(
        fuel,
        (oil_use_total * constants.GJ_TO_KG_OIL / ref_frac),
        refinery_intensity_weight_oil,
    )  # IN KG
    return (
        extraction_stock_oil,
        transport_total_oil,
        oil_pipelines,
        oil_storage,
        refinery_stock_oil,
    )

#Comment back in if running for a new scenario, otherwise it will overwrite the existing files with the same name (since the input data is the same)
# oilresults = oil_infra()
# oilresults[0].to_csv(OUTPUT_DIR / scenario / "oil_extraction_stock_kg.csv")
# oilresults[1].to_csv(OUTPUT_DIR / scenario / "oil_transport_stock_kg.csv")
# oilresults[2].to_csv(OUTPUT_DIR / scenario / "oil_pipelines_length_km.csv")
# oilresults[3].to_csv(OUTPUT_DIR / scenario / "oil_storage_volume_m3.csv")
# oilresults[4].to_csv(OUTPUT_DIR / scenario / "oil_refinery_stock_kg.csv")

print("=== FUMA SCRIPT FINISHED SUCCESSFULLY ===")



#%% Import general modules and constants

import pint
import xarray as xr
import prism


from ffconstants import FF_TECHNOLOGIES, STANDARD_SCEN_EXTERNAL_DATA, YEAR_FIRST_GRID, Standard_deviation_lifetime

from imagematerials.read_mym import read_mym_df
from imagematerials.util import dataset_to_array, pandas_to_xarray, convert_lifetime
from imagematerials.concepts import create_electricity_graph, create_region_graph
from imagematerials.electricity.utils import (
   MNLogit, 
   stock_tail, 
   create_prep_data, 
   interpolate_xr, 
   add_historic_stock, 
   calculate_grid_growth, 
   calculate_fraction_underground, 
   apply_ce_measures_to_elc
)

#from prism.prism.examples.fuel import scenario

path_current = Path().resolve()
path_base = path_current.parents [1]
print("current:", path_current)
print("base:", path_base)

#NOt sure what to do with these 
scen_folder = "SSP3_no_policy"
climate_policy_scenario_dir = Path(path_base, "data", "raw", "image", scen_folder)
STANDARD_SCEN_EXTERNAL_DATA = "SSP3_no_policy"  # if the data for the selected scenario is not available, we will use the data for this scenario instead (which is the same as the selected scenario for now, but can be changed in the future when we have more scenarios available)
SCENARIO_DEFAULT = "SSP3_no_policy"  # default scenario to use if the selected scenario is not available (same as standard scen external data for now, but can be changed in the future when we have more scenarios available)

#Not 100% on what these do yet 
year_start = 1880
year_end = 2100
year_out = 2100


#%% Extraction stage (coal, oil, gas) ---------------------------------------------------------------------------------------------------------------------------------

###########################################################################################################
#def get_preprocessing_data_extraction(path_base: str, climate_policy_config: dict, scenario: str, year_start: int, year_end: int, year_out: int):

   # path_external_data_scenario = Path(path_base, "fossil", scenario)

    # If scenario folder does not exist, fall back to standard scenario
   # if not path_external_data_scenario.exists():
    #    path_external_data_scenario = Path(path_base, "fossil", STANDARD_SCEN_EXTERNAL_DATA)

    #assert path_external_data_scenario.is_dir()

    #return path_external_data_scenario

###########################################################################################################
    # Read in files #

    # 1. External Data --------------------------------------------- 

    # lifetimes of extraction tech in years
extraction_lifetime_data = pd.read_csv(path_base / "data" / "raw" / "fossil_fuels" / "Extraction" / "Extraction_lifetimes.csv",
    index_col=["Year", "TechType"]
) 

    # material compositions of coal infrastructure (processing, extraction, transport) in kg/kg/year
extraction_materials_data = pd.read_csv(path_base / "data" / "raw" / "fossil_fuels" / "Extraction" / "Extraction_materials.csv")
#print(extraction_materials_data.index.is_unique)
#print(extraction_materials_data.columns)
#print(extraction_materials_data.head())
    # 2.FUMA files -----------------------------------------
#Stock of each type of extraction infrastructure (stock demand per generation technology) per region per year
#files are sourced from the output of the stock calculation (FUMA) for the relevant scenario, which is found under stock_calculation/output/SSP1_ML (or the relevant scenario)
extraction_coal = Path(path_base, "data", "raw", "fossil_fuels", "Scenario_data",scenario, "coal_extraction_stock_kg.csv")
df_extraction_coal = pd.read_csv(extraction_coal)
extraction_oil = Path(path_base, "data", "raw", "fossil_fuels", "Scenario_data",scenario, "oil_extraction_stock_kg.csv")
df_extraction_oil = pd.read_csv(extraction_oil)
extraction_gas = Path(path_base, "data", "raw", "fossil_fuels", "Scenario_data",scenario, "gas_extraction_stock_kg.csv")
df_extraction_gas = pd.read_csv(extraction_gas)

#Add fuel column to each df so that we can combine them and then split them again into the different technologies (coal, oil, gas) after melting the df to long format
df_extraction_coal["fuel"] = "coal"
df_extraction_oil["fuel"] = "oil"
df_extraction_gas["fuel"] = "gas"

# Combine the dataframes for coal, oil, and gas into one dataframe and then melt it to long format so that we have one row per combination of time, region, fuel, technology, and value (stock)
df_extraction_all = pd.concat(
    [df_extraction_coal, df_extraction_oil, df_extraction_gas],
    ignore_index=True
)

df_extraction_all = df_extraction_all.melt(
    id_vars=['type', 'time', 'fuel', 'stage', 'unit'],  
    var_name='DIM_1',
    value_name='value'
)

#Combine fuel and type columns to one column "Tech Type"
df_extraction_all['Tech Type'] = df_extraction_all['fuel'].str.strip() + ' ' + df_extraction_all['type'].str.strip()

#Drop columns that are not needed for the analysis (stage, unit, type, fuel) since we have already combined the relevant information into the "Tech Type" column and we know that all rows are for the extraction stage and in kg units
df_extraction_all = df_extraction_all.drop(columns=['stage', 'unit', 'type', 'fuel'])  

    ###########################################################################################################
    # Transform to xarray #

#still need to make knowledge graphs  
#knowledge_graph_region = create_region_graph()
#knowledge_graph_coal = create_ff_graph()
    

# Lifetimes -------
types = ["Coal Underground","Coal open cast","Oil off shore","Oil on shore","Gas off shore","Gas on shore"]
values = extraction_lifetime_data["Lifetime"].unstack().reindex(columns=types).to_numpy(dtype=float)
times = extraction_lifetime_data.index.get_level_values("Year").unique().to_numpy()
scipy_params = ["mean", "stdev"]

# Build xarray with shape (ScipyParam, Time, Type)
data_array = np.stack([values, np.full_like(values, Standard_deviation_lifetime)], axis=0)

extraction_lifetime_xr = xr.DataArray(
    data_array,
    dims=["DistributionParams", "Cohort", "Type"],
    coords={
        "DistributionParams": scipy_params,
        "Cohort": times,
        "Type": types
    },
    name="ExtractionLifetime"
)

# Expand 2020 across 1880-2100
year_range = np.arange(1880, 2101)
value_2020 = extraction_lifetime_xr.sel(Cohort=2020)
extraction_lifetime_xr = value_2020.expand_dims(Cohort=year_range).transpose("DistributionParams", "Cohort", "Type")

#Add units 
extraction_lifetime_xr = prism.Q_(extraction_lifetime_xr, "year")

#Uncomment to check if the xarray looks correct
#print(extraction_lifetime_xr.head())
#print(types)
#print("First year:", extraction_lifetime_xr.Cohort.values[0])
#print("Last year:", extraction_lifetime_xr.Cohort.values[-1])
#print("Number of years:", len(extraction_lifetime_xr.Cohort))
#print("Shape of DataArray:", extraction_lifetime_xr.shape)

#TODO: check about rebroadcasting to knowledge graphs (we also dont have the knowledge graphs yet)

#extraction_lifetime_xr = knowledge_graph_electr.rebroadcast_xarray(extraction_lifetime_xr, output_coords=FF_TECHNOLOGIES, dim="Type") # convert technology names to the standard names from TIMER
#extraction_lifetime_xr = extraction_lifetime_xr.assign_coords(Type=np.array(extraction_lifetime_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

 # Material Intensities -------

# Set index
extraction_materials_data = extraction_materials_data.set_index(['Year', 'Tech Type'])

# Convert to 3D array: (Material, Year, Tech)
extraction_materials_xr = (
    extraction_materials_data
        .to_xarray()            
        .to_array("Material") 
        .rename({"Year": "Cohort", "Tech Type": "Type"})
)

extraction_materials_xr.name = "ExtractionMaterialIntensities"

#Expand 2020 values across 1880-2100
value_2020 = extraction_materials_xr.sel(Cohort=2020)
year_range = np.arange(1880, 2101)
extraction_materials_xr = value_2020.expand_dims(Cohort=year_range).transpose("Material", "Cohort", "Type")

#Add units
extraction_materials_xr = prism.Q_(extraction_materials_xr, "kg/kg/year")

#Uncomment to check if it worked
#print(extraction_materials_xr.head())
#print(types)
#print("First year:", extraction_materials_xr.Cohort.values[0])
#print("Last year:", extraction_materials_xr.Cohort.values[-1])
#print("Number of years:", len(extraction_materials_xr.Cohort))
#print("Shape of DataArray:", extraction_materials_xr.shape)

#TODO: check about rebroadcasting to knowledge graphs (we also dont have the knowledge graphs yet)
#extraction_materials_xr = knowledge_graph_coal.rebroadcast_xarray(extraction_materials_xr, output_coords=EPG_TECHNOLOGIES, dim="Type")
#extraction_materials_xr = extraction_materials_xr.assign_coords(Type=np.array(extraction_materials_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

# Starting stock (capacity) ------
df_extraction_all = df_extraction_all.loc[~df_extraction_all['DIM_1'].isin([27,28])]  # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
df_extraction_all = df_extraction_all.loc[df_extraction_all['time'].isin(range(year_start, year_end + 1)), ['time', 'DIM_1', 'value', 'Tech Type']]  # only keep relevant years and technology columns
df_extraction_all['DIM_1'] = df_extraction_all['DIM_1'].astype(int)   

#Extract coordinate labels for years, regions, technologies
years = sorted(df_extraction_all['time'].unique())
regions = sorted(df_extraction_all['DIM_1'].unique())
techtypes = sorted(df_extraction_all['Tech Type'].unique())

 # Convert to 3D array: (Year, Region, Fuel, Type)
extractioncap_xr = (
    df_extraction_all
    .set_index(['time', 'DIM_1', 'Tech Type'])
    .to_xarray()['value']
    .rename({'time': 'Time', 'DIM_1': 'Region', 'Tech Type': 'Type'})
)

#extractioncap_xr = prism.Q_(extractioncap_xr, "MW")
#extractioncap_xr = knowledge_graph_region.rebroadcast_xarray(extractioncap_xr, output_coords=IMAGE_REGIONS, dim="Region") 
#extractioncap_xr = knowledge_graph_coal.rebroadcast_xarray(extractioncap_xr, output_coords=FF_TECHNOLOGIES, dim="Type")
extractioncap_xr = extractioncap_xr.assign_coords(Type=np.array(extractioncap_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

 # This is from original model and i am not sure if something like this is needed in the new model?
# TIMER data only start in 1971, so we add a historic tail back to YEAR_FIRST_GRID=1921 
#extractioncap_xr_interp = add_historic_stock(extractioncap_xr, YEAR_FIRST_GRID)

print("=== extractioncap_xr ===")
print(extractioncap_xr)
print("\nDims:", extractioncap_xr.dims)
print("Sizes:", extractioncap_xr.sizes)
print("\nTime labels:", extractioncap_xr['Time'].values)
print("Region labels:", extractioncap_xr['Region'].values)
print("Type labels:", extractioncap_xr['Type'].values)
extractioncap_xr.isnull().sum()

print("=== extractionlifetimes_xr ===")
print(extraction_lifetime_xr)
print("Dims:", extraction_lifetime_xr.dims)
print("Sizes:", extraction_lifetime_xr.sizes)
print("Cohort:", extraction_lifetime_xr['Cohort'].values)
print("Type:", extraction_lifetime_xr['Type'].values)
print("Params:", extraction_lifetime_xr['DistributionParams'].values)
extraction_lifetime_xr.isnull().sum()

print("=== extraction_materials_xr ===")
print(extraction_materials_xr)
print("Dims:", extraction_materials_xr.dims)
print("Sizes:", extraction_materials_xr.sizes)
print("Materials:", extraction_materials_xr['Material'].values)
print("Cohort:", extraction_materials_xr['Cohort'].values)
print("Type:", extraction_materials_xr['Type'].values)
extraction_materials_xr.isnull().sum()


#%% Processing stage (coal, oil, gas) ---------------------------------------------------------------------------------------------------------------------------------

#def get_preprocessing_data_processing(path_base: str, climate_policy_config: dict, scenario: str, year_start: int, year_end: int, year_out: int):

    #path_external_data_scenario = Path(path_base, "fossil", scenario)

    # test if path_external_data_scenario exists and if not set to standard scenario
    #if not path_external_data_scenario.exists():
        #path_external_data_scenario = Path(path_base, "fossil", STANDARD_SCEN_EXTERNAL_DATA)

    #assert path_external_data_scenario.is_dir()

    #return path_external_data_scenario
###########################################################################################################
    # Read in files #

    # 1. External Data --------------------------------------------- 

    # lifetimes of processing tech in years
processing_lifetime_data = pd.read_csv(
    path_base / "data" / "raw" / "fossil_fuels" / "Processing" / "Processing_lifetimes.csv",
    index_col=["Year", "Tech Type"]
) 
    # material compositions of processing infrastructure in kg/kg/year
processing_materials_data = pd.read_csv(path_base / "data" / "raw" / "fossil_fuels" / "Processing" / "Processing_materials.csv")

    # 2. FUMA files -----------------------------------------
#Stock of each type of extraction infrastructure (stock demand per generation technology) per region per year

processing_coal = Path(path_base, "data", "raw", "fossil_fuels", "Scenario_data", scenario, "coal_preparation_stock_kg.csv")
df_processing_coal = pd.read_csv(processing_coal)
processing_storage_oil = Path(path_base, "data", "raw", "fossil_fuels", "Scenario_data", scenario, "oil_storage_volume_m3.csv")
df_processing_oil = pd.read_csv(processing_storage_oil)
processing_refinery_oil = Path(path_base, "data", "raw", "fossil_fuels", "Scenario_data", scenario, "oil_refinery_stock_kg.csv")
df_processing_refinery_oil = pd.read_csv(processing_refinery_oil)
processing_gas = Path(path_base, "data", "raw", "fossil_fuels", "Scenario_data", scenario, "gas_processing_stock_kg.csv")
df_processing_gas = pd.read_csv(processing_gas)

#Combine the dataframes for coal, oil, and gas into one dataframe and then melt it to long format so that we have one row per combination of time, region, fuel, technology, and value (stock)
df_processing_all = pd.concat(
    [df_processing_coal, df_processing_oil, df_processing_refinery_oil, df_processing_gas],
    ignore_index=True
)

df_processing_all = df_processing_all.melt(
    id_vars=['type', 'time', 'fuel', 'stage', 'unit'],  
    var_name='DIM_1',
    value_name='value'
)

#Combine fuel and type columns to one column "Tech Type"
df_processing_all['Tech Type'] = df_processing_all['fuel'].str.strip() + ' ' + df_processing_all['type'].str.strip()

# Drop columns that are not needed for the analysis
df_processing_all = df_processing_all.drop(columns=['stage', 'unit', 'type', 'fuel'])  

    ###########################################################################################################
    # Transform to xarray #
#TODO: need to work on knowledge graphs for the processing stage    
#knowledge_graph_region = create_region_graph()
#knowledge_graph_coal = create_ff_graph()
    
    # Lifetimes -------
values = processing_lifetime_data["Lifetime"].unstack().to_numpy(dtype=float)
#Create coordinates
times = processing_lifetime_data.index.levels[0].to_numpy()
types = processing_lifetime_data.index.levels[1].to_numpy()
scipy_params = ["mean", "stdev"]
    # Build full array: shape (ScipyParam, Time, Type)
data_array = np.stack([values, np.full_like(values, Standard_deviation_lifetime)], axis=0)
    # Create DataArray
processing_lifetime_xr = xr.DataArray(
        data_array,
        dims=["DistributionParams", "Cohort", "Type"],
        coords={
            "DistributionParams": scipy_params,
            "Cohort": times,
            "Type": [str(r) for r in types]
        },
        name="ProcessingLifetime"
    )
# Expand 2020 across 1880-2100
year_range = np.arange(1880, 2101)
value_2020 = processing_lifetime_xr.sel(Cohort=2020)
processing_lifetime_xr = value_2020.expand_dims(Cohort=year_range).transpose("DistributionParams", "Cohort", "Type")
processing_lifetime_xr = prism.Q_(processing_lifetime_xr, "year")

#Uncomment to check if the xarray looks correct
#print(processing_lifetime_xr.head())
#print(types)
#print("First year:", processing_lifetime_xr.Cohort.values[0])
#print("Last year:", processing_lifetime_xr.Cohort.values[-1])
#print("Number of years:", len(processing_lifetime_xr.Cohort))
#print("Shape of DataArray:", processing_lifetime_xr.shape)

 # Material Intensities -------

    # Material Intensities -------
   # Set index
processing_materials_data = processing_materials_data.set_index(['Year', 'Tech Type'])

   # Convert to 3D array: (Material, Year, Tech)
processing_materials_xr = (
    processing_materials_data
        .to_xarray()           # keeps Year & Tech Type as coords
        .to_array("Material")  # converts columns into a 'Material' dimension
        .rename({"Year": "Cohort", "Tech Type": "Type"})
)

# Name the DataArray
processing_materials_xr.name = "ProcessingMaterialIntensities"

#Expand 2020 values across 1880-2100
value_2020 = processing_materials_xr.sel(Cohort=2020)
year_range = np.arange(1880, 2101)
processing_materials_xr = value_2020.expand_dims(Cohort=year_range).transpose("Material", "Cohort", "Type")

#Add units
processing_materials_xr = prism.Q_(processing_materials_xr, "kg/kg/year")

#Uncomment to check if it worked
#print(processing_materials_xr.head())
# print(types)
# print("First year:", processing_materials_xr.Cohort.values[0])
# print("Last year:", processing_materials_xr.Cohort.values[-1])
# print("Number of years:", len(processing_materials_xr.Cohort))
# print("Shape of DataArray:", processing_materials_xr.shape)

#TODO: check about rebroadcasting to knowledge graphs (we also dont have the knowledge graphs yet)
#processing_materials_xr = knowledge_graph_coal.rebroadcast_xarray(processing_materials_xr, output_coords=EPG_TECHNOLOGIES, dim="Type")
#processing_materials_xr = processing_materials_xr.assign_coords(Type=np.array(processing_materials_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

# Gcap ------
# combine the processing data for coal, oil, gas into one df and then put it into this 
df_processing_all = df_processing_all.loc[~df_processing_all['DIM_1'].isin([27,28])]  # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
df_processsing_all = df_processing_all.loc[df_processing_all['time'].isin(range(year_start, year_end + 1)), ['time', 'DIM_1', 'value', 'Tech Type']]  # only keep relevant years and technology columns
   
# Extract coordinate labels for years, regions, technologies
regions = sorted(df_processing_all['DIM_1'].unique())
techtypes = sorted(df_processing_all['Tech Type'].unique())

 # Convert to 3D array: (Year, Region, Fuel, Type)
processingcap_xr = (
    df_processing_all
    .set_index(['time', 'DIM_1', 'Tech Type'])
    .to_xarray()['value']
    .rename({'time': 'Time', 'DIM_1': 'Region', 'Tech Type': 'Type'})
)
#Add units
processingcap_xr = prism.Q_(processingcap_xr, "kg?")
#processing_capacities_xr = knowledge_graph_region.rebroadcast_xarray(processing_capacities_xr, output_coords=IMAGE_REGIONS, dim="Region") 
#processing_capacities_xr = knowledge_graph_coal.rebroadcast_xarray(processing_capacities_xr, output_coords=FF_TECHNOLOGIES, dim="Type")
processingcap_xr = processingcap_xr.assign_coords(Type=np.array(processingcap_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

#Not sure if this is needed 
# TIMER data only start in 1971, so we add a historic tail back to YEAR_FIRST_GRID=1921 #TODO to be adjusted
#processingcap_xr_interp = add_historic_stock(processingcap_xr, YEAR_FIRST_GRID)

# print("=== processingcap_xr ===")
# print(processingcap_xr)
# print("\nDims:", processingcap_xr.dims)
# print("Sizes:", processingcap_xr.sizes)
# print("\nTime labels:", processingcap_xr['Time'].values)
# print("Region labels:", processingcap_xr['Region'].values)
# print("Type labels:", processingcap_xr['Type'].values)
# processingcap_xr.isnull().sum()

# print("=== processinglifetimes_xr ===")
# print(processing_lifetime_xr)
# print("Dims:", processing_lifetime_xr.dims)
# print("Sizes:", processing_lifetime_xr.sizes)
# print("Cohort:", processing_lifetime_xr['Cohort'].values)
# print("Type:", processing_lifetime_xr['Type'].values)
# print("Params:", processing_lifetime_xr['DistributionParams'].values)
# processing_lifetime_xr.isnull().sum()

# print("=== processing_materials_xr ===")
# print(processing_materials_xr)
# print("Dims:", processing_materials_xr.dims)
# print("Sizes:", processing_materials_xr.sizes)
# print("Materials:", processing_materials_xr['Material'].values)
# print("Cohort:", processing_materials_xr['Cohort'].values)
# print("Type:", processing_materials_xr['Type'].values)
# processing_materials_xr.isnull().sum()


#%% Transport/Vehicles stage (coal, oil, gas) ---------------------------------------------------------------------------------------------------------------------------------

###########################################################################################################
#def get_preprocessing_data_processing(path_base: str, climate_policy_config: dict, scenario: str, year_start: int, year_end: int, year_out: int):

    #path_external_data_scenario = Path(path_base, "fossil", scenario)

    # test if path_external_data_scenario exists and if not set to standard scenario
# if not path_external_data_scenario.exists():
#    path_external_data_scenario = Path(path_base, "fossil", STANDARD_SCEN_EXTERNAL_DATA)

#assert path_external_data_scenario.is_dir()
###########################################################################################################
# Read in files #

# 1. External Data --------------------------------------------- 

# lifetimes of transport tech in years
transport_lifetime_data = pd.read_csv(path_base / "data" / "raw" / "fossil_fuels" / "Transport" / "Transport_lifetimes.csv", index_col=["Year", "Tech Type"])

# material compositions of transport infrastructure in kg/kg/year
transport_materials_data = pd.read_csv(path_base / "data" / "raw" / "fossil_fuels" / "Transport" / "Transport_materials.csv")

    # 2. IMAGE/TIMER files -----------------------------------------
#Transport capacity (stock demand per generation technology) in MW peak capacity
transport_coal = Path(path_base, "data", "raw", "fossil_fuels", "Scenario_data", scenario, "coal_transport_stock_kg.csv")
df_transport_coal = pd.read_csv(transport_coal)
transport_oil = Path(path_base, "data", "raw", "fossil_fuels", "Scenario_data", scenario, "oil_transport_stock_kg.csv")
df_transport_oil = pd.read_csv(transport_oil)
transport_gas = Path(path_base, "data", "raw", "fossil_fuels", "Scenario_data", scenario, "gas_transport_stock_kg.csv")
df_transport_gas = pd.read_csv(transport_gas)

df_transport_coal["fuel"] = "coal"
df_transport_oil["fuel"] = "oil"
df_transport_gas["fuel"] = "gas"

#Combine the dataframes for coal, oil, and gas into one dataframe and then melt it to long format so that we have one row per combination of time, region, fuel, technology, and value (stock)
df_transport_all = pd.concat(
    [df_transport_coal, df_transport_oil, df_transport_gas],
    ignore_index=True
)

df_transport_all = df_transport_all.melt(
    id_vars=['type', 'time', 'fuel', 'stage', 'unit'],  
    var_name='DIM_1',
    value_name='value'
)

#Combine fuel and type columns to one column "Tech Type"
df_transport_all['Tech Type'] = df_transport_all['fuel'].str.strip() + ' ' + df_transport_all['type'].str.strip()

#Drop columns that are not needed for the analysis 
df_transport_all = df_transport_all.drop(columns=['stage', 'unit', 'type', 'fuel']) 

    ###########################################################################################################
    # Transform to xarray #
#TODO: need to work on knowledge graphs for the transport stage
#knowledge_graph_region = create_region_graph()
#knowledge_graph_coal = create_ff_graph()
    
    # Lifetimes -------
values = transport_lifetime_data["Lifetimes"].unstack().to_numpy(dtype=float)
#Create coordinates
times = transport_lifetime_data.index.levels[0].to_numpy()
types = transport_lifetime_data.index.levels[1].to_numpy()
scipy_params = ["mean", "stdev"]
# Build full array: shape (ScipyParam, Time, Type)
data_array = np.stack([values, np.full_like(values, Standard_deviation_lifetime)], axis=0)
# Create DataArray
transport_lifetime_xr = xr.DataArray(
        data_array,
        dims=["DistributionParams", "Cohort", "Type"],
        coords={
            "DistributionParams": scipy_params,
            "Cohort": times,
            "Type": [str(r) for r in types]
        },
        name="TransportLifetime"
    )

# Expand 2020 across 1880-2100
year_range = np.arange(1880, 2101)
value_2020 = transport_lifetime_xr.sel(Cohort=2020)
transport_lifetime_xr = value_2020.expand_dims(Cohort=year_range).transpose("DistributionParams", "Cohort", "Type")
transport_lifetime_xr = prism.Q_(transport_lifetime_xr, "year")

#Uncomment to check if the xarray looks correct
# print(transport_lifetime_xr.head())
# print(types)
# print("First year:", transport_lifetime_xr.Cohort.values[0])
# print("Last year:", transport_lifetime_xr.Cohort.values[-1])
# print("Number of years:", len(transport_lifetime_xr.Cohort))
# print("Shape of DataArray:", transport_lifetime_xr.shape)

#transport_lifetime_xr = knowledge_graph_electr.rebroadcast_xarray(coal_transport_xr, output_coords=EPG_TECHNOLOGIES, dim="Type") # convert technology names to the standard names from TIMER
#transport_lifetime_xr = coal_lifetime_xr.assign_coords(Type=np.array(transport_lifetime_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

 # Material Intensities -------

   # Set index
transport_materials_data = transport_materials_data.set_index(['Year', 'Tech Type'])

# Convert to 3D array: (Material, Year, Tech)
transport_materials_xr = (
    transport_materials_data
        .to_xarray()           
        .to_array("Material")  
        .rename({"Year": "Cohort", "Tech Type": "Type"})
)
transport_materials_xr.name = "TransportMaterialIntensities"

#Expand 2020 values across 1880-2100
value_2020 = transport_materials_xr.sel(Cohort=2020)
year_range = np.arange(1880, 2101)
transport_materials_xr = value_2020.expand_dims(Cohort=year_range).transpose("Material", "Cohort", "Type")
transport_materials_xr = prism.Q_(transport_materials_xr, "kg/kg/year")

# Uncomment to check if it worked
# print(transport_materials_xr.head())
# print(types)
# print("First year:", transport_materials_xr.Cohort.values[0])
# print("Last year:", transport_materials_xr.Cohort.values[-1])
# print("Number of years:", len(transport_materials_xr.Cohort))
# print("Shape of DataArray:", transport_materials_xr.shape)

#TODO: check about rebroadcasting to knowledge graphs (we also dont have the knowledge graphs yet)
#transport_materials_xr = knowledge_graph_coal.rebroadcast_xarray(transport_materials_xr, output_coords=EPG_TECHNOLOGIES, dim="Type")
#transport_materials_xr = transport_materials_xr.assign_coords(Type=np.array(transport_materials_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

# Transport capacity ------
df_transport_all = df_transport_all.loc[~df_transport_all['DIM_1'].isin([27,28])]  # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
df_transport_all = df_transport_all.loc[df_transport_all['time'].isin(range(year_start, year_end + 1)), ['time', 'DIM_1', 'value', 'Tech Type']]  # only keep relevant years and technology columns
   
# Extract coordinate labels

years = sorted(df_transport_all['time'].unique())
regions = sorted(df_transport_all['DIM_1'].unique())
techtypes = sorted(df_transport_all['Tech Type'].unique())

transportcap_xr = (
    df_transport_all
    .set_index(['time', 'DIM_1', 'Tech Type'])
    .to_xarray()['value']
    .rename({'time': 'Time', 'DIM_1': 'Region', 'Tech Type': 'Type'})
)

transportcap_xr = prism.Q_(transportcap_xr, "MW")
#transportcap_xr = knowledge_graph_region.rebroadcast_xarray(transportcap_xr, output_coords=IMAGE_REGIONS, dim="Region") 
#transportcap_xr = knowledge_graph_coal.rebroadcast_xarray(transportcap_xr, output_coords=FF_TECHNOLOGIES, dim="Type")
transportcap_xr = transportcap_xr.assign_coords(Type=np.array(transportcap_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

#Not sure if this is needed
# TIMER data only start in 1971, so we add a historic tail back to YEAR_FIRST_GRID=1921 #TODO to be adjusted
#transportcap_xr_interp = add_historic_stock(transportcap_xr, YEAR_FIRST_GRID)

# print("=== transportcap_xr ===")
# print(transportcap_xr)
# print("\nDims:", transportcap_xr.dims)
# print("Sizes:", transportcap_xr.sizes)
# print("\nTime labels:", transportcap_xr['Time'].values)
# print("Region labels:", transportcap_xr['Region'].values)
# print("Type labels:", transportcap_xr['Type'].values)
# transportcap_xr.isnull().sum()

# print("=== transportlifetimes_xr ===")
# print(transport_lifetime_xr)
# print("Dims:", transport_lifetime_xr.dims)
# print("Sizes:", transport_lifetime_xr.sizes)
# print("Cohort:", transport_lifetime_xr['Cohort'].values)
# print("Type:", transport_lifetime_xr['Type'].values)
# print("Params:", transport_lifetime_xr['DistributionParams'].values)
# transport_lifetime_xr.isnull().sum()

# print("=== transport_materials_xr ===")
# print(transport_materials_xr)
# print("Dims:", transport_materials_xr.dims)
# print("Sizes:", transport_materials_xr.sizes)
# print("Materials:", transport_materials_xr['Material'].values)
# print("Cohort:", transport_materials_xr['Cohort'].values)
# print("Type:", transport_materials_xr['Type'].values)
# transport_materials_xr.isnull().sum()

#%% Pipelines stage (oil, gas) ---------------------------------------------------------------------------------------------------------------------------------
###########################################################################################################

# def get_preprocessing_data_processing(path_base: str, climate_policy_config: dict, scenario: str, year_start: int, year_end: int, year_out: int):

    # path_external_data_scenario = Path(path_base, "fossil", scenario)
    # test if path_external_data_scenario exists and if not set to standard scenario
    # if not path_external_data_scenario.exists():
        # path_external_data_scenario = Path(path_base, "fossil", STANDARD_SCEN_EXTERNAL_DATA)

   #  assert path_external_data_scenario.is_dir()
###########################################################################################################
    # Read in files #

    # 1. External Data --------------------------------------------- 

    # lifetimes of transport tech in years
pipelines_lifetime_data = pd.read_csv(
    path_base / "data" / "raw" / "fossil_fuels" / "Pipelines" / "Pipelines_lifetimes.csv",
    index_col=["Year", "Tech Type"]
) 
    # material compositions of transport infrastructure in kg/kg/year
pipelines_materials_data = pd.read_csv(path_base / "data" / "raw" / "fossil_fuels" / "Pipelines" / "Pipelines_materials.csv")
    # 2. IMAGE/TIMER files -----------------------------------------
#Transport capacity (stock demand per generation technology) in MW peak capacity
pipelines_oil = Path(path_base, "data", "raw", "fossil_fuels", "Scenario_data", scenario, "oil_pipelines_length_km.csv")
df_pipelines_oil = pd.read_csv(pipelines_oil)
pipelines_gas = Path(path_base, "data", "raw", "fossil_fuels", "Scenario_data", scenario, "gas_pipelines_length_km.csv")
df_pipelines_gas = pd.read_csv(pipelines_gas)

df_pipelines_oil["fuel"] = "oil"
df_pipelines_gas["fuel"] = "gas"

#Combine the dataframes for oil and gas into one dataframe and then melt it to long format so that we have one row per combination of time, region, fuel, technology, and value (stock)
df_pipelines_all = pd.concat(
    [df_pipelines_oil, df_pipelines_gas],
    ignore_index=True
)

df_pipelines_all = df_pipelines_all.melt(
    id_vars=['type', 'time', 'fuel', 'stage', 'unit'],  
    var_name='DIM_1',
    value_name='value'
)

#Combine fuel and type columns to one column "Tech Type"
df_pipelines_all['Tech Type'] = df_pipelines_all['fuel'].str.strip() + ' ' + df_pipelines_all['type'].str.strip()

#Drop columns that are not needed for the analysis
df_pipelines_all = df_pipelines_all.drop(columns=['stage', 'unit', 'type', 'fuel'])  # drop columns that are not needed for the analysis

    ###########################################################################################################
    # Transform to xarray #
#TODO: need to work on knowledge graphs 
#knowledge_graph_region = create_region_graph()
#knowledge_graph_coal = create_ff_graph()

    # Lifetimes -------
values = pipelines_lifetime_data["Lifetimes"].unstack().to_numpy(dtype=float)
#Create coordinates
times = pipelines_lifetime_data.index.levels[0].to_numpy()
types = pipelines_lifetime_data.index.levels[1].to_numpy()
scipy_params = ["mean", "stdev"]
# Build full array: shape (ScipyParam, Time, Type)
data_array = np.stack([values, np.full_like(values, Standard_deviation_lifetime)], axis=0)
# Create DataArray
pipelines_lifetime_xr = xr.DataArray(
        data_array,
        dims=["DistributionParams", "Cohort", "Type"],
        coords={
            "DistributionParams": scipy_params,
            "Cohort": times,
            "Type": [str(r) for r in types]
        },
        name="PipelinesLifetime"
    )
pipelines_lifetime_xr = prism.Q_(pipelines_lifetime_xr, "year")

# Expand 2020 across 1880-2100
year_range = np.arange(1880, 2101)
value_2020 = pipelines_lifetime_xr.sel(Cohort=2020)
pipelines_lifetime_xr = value_2020.expand_dims(Cohort=year_range).transpose("DistributionParams", "Cohort", "Type")

#Uncomment to check if the xarray looks correct
# print(pipelines_lifetime_xr.head())
# print(types)
# print("First year:", pipelines_lifetime_xr.Cohort.values[0])
# print("Last year:", pipelines_lifetime_xr.Cohort.values[-1])
# print("Number of years:", len(pipelines_lifetime_xr.Cohort))
# print("Shape of DataArray:", pipelines_lifetime_xr.shape)

#TODO: check about rebroadcasting to knowledge graphs (we also dont have the knowledge graphs yet)
#pipelines_lifetime_xr = knowledge_graph_electr.rebroadcast_xarray(pipelines_lifetime_xr, output_coords=EPG_TECHNOLOGIES, dim="Type") # convert technology names to the standard names from TIMER
#pipelines_lifetime_xr = pipelines_lifetime_xr.assign_coords(Type=np.array(pipelines_lifetime_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

 # Material Intensities -------
   # Set index
pipelines_materials_data = pipelines_materials_data.set_index(['Year', 'Tech Type'])

   # Convert to 3D array: (Material, Year, Tech)
pipelines_materials_xr = (
    pipelines_materials_data
        .to_xarray()           
        .to_array("Material")  
        .rename({"Year": "Cohort", "Tech Type": "Type"})
)
pipelines_materials_xr.name = "PipelinesMaterialIntensities"

#Expand 2020 values across 1880-2100
value_2020 = pipelines_materials_xr.sel(Cohort=2020)
year_range = np.arange(1880, 2101)
pipelines_materials_xr = value_2020.expand_dims(Cohort=year_range).transpose("Material", "Cohort", "Type")
pipelines_materials_xr = prism.Q_(pipelines_materials_xr, "kg/kg/year")

# Uncomment to check if it worked
# print(pipelines_materials_xr.head())
# print(types)
# print("First year:", pipelines_materials_xr.Cohort.values[0])
# print("Last year:", pipelines_materials_xr.Cohort.values[-1])
# print("Number of years:", len(pipelines_materials_xr.Cohort))
# print("Shape of DataArray:", pipelines_materials_xr.shape)

pipelines_materials_xr = prism.Q_(pipelines_materials_xr, "kg/kg/year")

#TODO: check about rebroadcasting to knowledge graphs (we also dont have the knowledge graphs yet)
#pipelines_materials_xr = knowledge_graph_coal.rebroadcast_xarray(pipelines_materials_xr, output_coords=EPG_TECHNOLOGIES, dim="Type")
#pipelines_materials_xr = pipelines_materials_xr.assign_coords(Type=np.array(pipelines_materials_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

# Pipeline capacity ------
df_pipelines_all = df_pipelines_all.loc[~df_pipelines_all['DIM_1'].isin([27,28])]  # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
df_pipelines_all = df_pipelines_all.loc[df_pipelines_all['time'].isin(range(year_start, year_end + 1)), ['time', 'DIM_1', 'value', 'Tech Type']]  # only keep relevant years and technology columns
    
    # Extract coordinate labels
years = sorted(df_pipelines_all['time'].unique())
regions = sorted(df_pipelines_all['DIM_1'].unique())
techtypes = sorted(df_pipelines_all['Tech Type'].unique())

# Convert to 3D array: (Year, Region, Fuel, Type)
pipelinecap_xr = (
    df_pipelines_all
    .set_index(['time', 'DIM_1', 'Tech Type'])
    .to_xarray()['value']
    .rename({'time': 'Time', 'DIM_1': 'Region', 'Tech Type': 'Type'})
)
pipelinecap_xr = prism.Q_(pipelinecap_xr, "MW")
#pipelinecap_xr = knowledge_graph_region.rebroadcast_xarray(pipelinecap_xr, output_coords=IMAGE_REGIONS, dim="Region") 
#pipelinecap_xr = knowledge_graph_coal.rebroadcast_xarray(pipelinecap_xr, output_coords=FF_TECHNOLOGIES, dim="Type")
pipelinecap_xr = pipelinecap_xr.assign_coords(Type=np.array(pipelinecap_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

# Not sure if this is needed
#TIMER data only start in 1971, so we add a historic tail back to YEAR_FIRST_GRID=1921 #TODO to be adjusted
#pipelinecap_xr_interp = add_historic_stock(pipelinecap_xr, YEAR_FIRST_GRID)

#Check if the xarray looks correct
# print("=== pipelinecap_xr ===")
# print(pipelinecap_xr)
# print("\nDims:", pipelinecap_xr.dims)
# print("Sizes:", pipelinecap_xr.sizes)
# print("\nTime labels:", pipelinecap_xr['Time'].values)
# print("Region labels:", pipelinecap_xr['Region'].values)
# print("Type labels:", pipelinecap_xr['Type'].values)
# pipelinecap_xr.isnull().sum()

# print("=== pipelines_lifetime_xr ===")
# print(pipelines_lifetime_xr)
# print("Dims:", pipelines_lifetime_xr.dims)
# print("Sizes:", pipelines_lifetime_xr.sizes)
# print("Cohort:", pipelines_lifetime_xr['Cohort'].values)
# print("Type:", pipelines_lifetime_xr['Type'].values)
# print("Params:", pipelines_lifetime_xr['DistributionParams'].values)
# pipelines_lifetime_xr.isnull().sum()

# print("=== pipelines_materials_xr ===")
# print(pipelines_materials_xr)
# print("Dims:", pipelines_materials_xr.dims)
# print("Sizes:", pipelines_materials_xr.sizes)
# print("Materials:", pipelines_materials_xr['Material'].values)
# print("Cohort:", pipelines_materials_xr['Cohort'].values)
# print("Type:", pipelines_materials_xr['Type'].values)
# pipelines_materials_xr.isnull().sum()

#%% Bring everything together in a prep_data file ---------------------------------------------------------------------------------------------------------------------------------
###########################################################################################################

#     # Prep_data File # for each stage (extraction, processing, transport, pipelines) 
# # Extraction stage (coal, oil, gas) ---------------------------------------------------------------------------------------------------------------------------------
# # The lifetimes are converted to the proper format for the model (dictionary with keys:distribution name, values:datarrays containing distribution parameters)
# extraction_lifetime_xr = convert_lifetime(extraction_lifetime_xr)
    
#  # bring preprocessing data into a generic format for the model
# prep_data = {}
# prep_data["lifetimes"] = extraction_lifetime_xr
# prep_data["stocks"] = extractioncap_xr
# prep_data["material_intensities"] = extraction_materials_xr
# #prep_data["knowledge_graph"] = create_electricity_graph() #TODO:make knowledge graph for FF
# # add units
# prep_data["stocks"] = prism.Q_(prep_data["stocks"], "")
# prep_data["material_intensities"] = prism.Q_(prep_data["material_intensities"], "")
# prep_data["set_unit_flexible"] = prism.U_(prep_data["stocks"]) # prism.U_ gives the unit back
#     # set_unit_flexible is needed by the model to deal with the fact the in the beginning of the model it doesn't know th data yet and needs to work with a placeholder/flexible unit (see model.py) 

#return prep_data

###########################################################################################################
# # Processing stage (coal, oil, gas) ---------------------------------------------------------------------------------------------------------------------------------
# # The lifetimes are converted to the proper format for the model (dictionary with keys:distribution name, values:datarrays containing distribution parameters)
# processing_lifetime_xr = convert_lifetime(processing_lifetime_xr)
    
#  # bring preprocessing data into a generic format for the model
# prep_data = {}
# prep_data["lifetimes"] = processing_lifetime_xr
# prep_data["stocks"] = processingcap_xr
# prep_data["material_intensities"] = processing_materials_xr
# #prep_data["knowledge_graph"] = create_electricity_graph() #TODO:make knowledge graph for FF
# # add units
# prep_data["stocks"] = prism.Q_(prep_data["stocks"], "")
# prep_data["material_intensities"] = prism.Q_(prep_data["material_intensities"], "")
# prep_data["set_unit_flexible"] = prism.U_(prep_data["stocks"]) # prism.U_ gives the unit back
#     # set_unit_flexible is needed by the model to deal with the fact the in the beginning of the model it doesn't know th data yet and needs to work with a placeholder/flexible unit (see model.py) 

# #return prep_data

# ###########################################################################################################
# # Transportation stage (coal, oil, gas) ---------------------------------------------------------------------------------------------------------------------------------
# # The lifetimes are converted to the proper format for the model (dictionary with keys:distribution name, values:datarrays containing distribution parameters)
# transport_lifetime_xr = convert_lifetime(transport_lifetime_xr)
    
#  # bring preprocessing data into a generic format for the model
# prep_data = {}
# prep_data["lifetimes"] = transport_lifetime_xr
# prep_data["stocks"] = transportcap_xr
# prep_data["material_intensities"] = transport_materials_xr
# #prep_data["knowledge_graph"] = create_electricity_graph() #TODO:make knowledge graph for FF
# # add units
# prep_data["stocks"] = prism.Q_(prep_data["stocks"], "")
# prep_data["material_intensities"] = prism.Q_(prep_data["material_intensities"], "")
# prep_data["set_unit_flexible"] = prism.U_(prep_data["stocks"]) # prism.U_ gives the unit back
#     # set_unit_flexible is needed by the model to deal with the fact the in the beginning of the model it doesn't know th data yet and needs to work with a placeholder/flexible unit (see model.py) 

# #return prep_data


# ###########################################################################################################
# # Processing stage (coal, oil, gas) ---------------------------------------------------------------------------------------------------------------------------------
# # The lifetimes are converted to the proper format for the model (dictionary with keys:distribution name, values:datarrays containing distribution parameters)
# processing_lifetime_xr = convert_lifetime(processing_lifetime_xr)
    
#  # bring preprocessing data into a generic format for the model
# prep_data = {}
# prep_data["lifetimes"] = processing_lifetime_xr
# prep_data["stocks"] = processingcap_xr
# prep_data["material_intensities"] = processing_materials_xr
# #prep_data["knowledge_graph"] = create_electricity_graph() #TODO:make knowledge graph for FF
# # add units
# prep_data["stocks"] = prism.Q_(prep_data["stocks"], "")
# prep_data["material_intensities"] = prism.Q_(prep_data["material_intensities"], "")
# prep_data["set_unit_flexible"] = prism.U_(prep_data["stocks"]) # prism.U_ gives the unit back
#     # set_unit_flexible is needed by the model to deal with the fact the in the beginning of the model it doesn't know th data yet and needs to work with a placeholder/flexible unit (see model.py) 

# #return prep_data

# ###########################################################################################################
# # Pipelines stage (oil, gas) ---------------------------------------------------------------------------------------------------------------------------------
# # The lifetimes are converted to the proper format for the model (dictionary with keys:distribution name, values:datarrays containing distribution parameters)
# pipelines_lifetime_xr = convert_lifetime(pipelines_lifetime_xr)
    
#  # bring preprocessing data into a generic format for the model
# prep_data = {}
# prep_data["lifetimes"] = pipelines_lifetime_xr
# prep_data["stocks"] = pipelinecap_xr
# prep_data["material_intensities"] = pipelines_materials_xr
# #prep_data["knowledge_graph"] = create_electricity_graph() #TODO:make knowledge graph for FF
# # add units
# prep_data["stocks"] = prism.Q_(prep_data["stocks"], "")
# prep_data["material_intensities"] = prism.Q_(prep_data["material_intensities"], "")
# prep_data["set_unit_flexible"] = prism.U_(prep_data["stocks"]) # prism.U_ gives the unit back
#     # set_unit_flexible is needed by the model to deal with the fact the in the beginning of the model it doesn't know th data yet and needs to work with a placeholder/flexible unit (see model.py) 

# #return prep_data


print("Model finished successfully!")