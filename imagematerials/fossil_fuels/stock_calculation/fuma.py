# %% A1) GENERAL IMPORTS, SETTING & STATEMENTS

import pandas as pd
import numpy as np
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

from attributes import constants
from calculations.materials import cohorts_to_materials_typical_np
from calculations.pipelines import pipelines_indexed_stock_growth
from calculations.transport import transport
from calculations.extraction import extraction_stocks
from calculations.storage import storage_indexed_stocks_growth
from calculations.refinery import refinery_stocks
from prepare_output_files import merge_material, merge_total
from mym.read_mym import read_mym_df
from stock_model.inflow_outflow_models import (
    inflow_outflow_typical_np,
    inflow_outflow_dynamic_np,
    inflow_outflow_surplus,
    inflow_outflow_surplus_typical_np,
    inflow_outflow_surplus_int,
)

idx = pd.IndexSlice
# Automatically set working directory to the script's location
script_dir = Path(__file__).resolve().parent
os.chdir(script_dir)
DATA_DIR = Path("data")
print(f" Working directory set to: {script_dir}")




# Set scenario
scenario = (
    "SSP3_H"  # Choose scenario. Available: "SSP2" & "SSP2_2D & SSP1"
)
ref_frac = 0.8  #  refineryRefining capacity used

# %% A2) Load input-data files

# IMAGE drivers
import os

primpersec = read_mym_df(
    os.path.join(
        "data",
        "T2ReMo_may23",
        scenario,
        "PrimPerSec.out"
    )
)
  # primary enery consumption per sector in GJ
final_energy = read_mym_df(
    os.path.join(
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

# %% A3) Prepare data files & formatting

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

# %% A4) Model drivers (fossil fuel use scenarios from IMAGE, by region towards 2100)
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

#%% (Intermediate results) Fuel use data

all_fossil_flows_sankey.to_csv(
    OUTPUT_DIR / scenario / "fuel_use_GJ.csv", index=True
)

select_year = 2019

# %% A5) Generate/prepend historic fossil fuel consumption profile (to avoid inflow shocks in the first year of 'stock-driven' dynamic stock calculations)
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


# %% ###################################################################################################
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
# %% B6) Run supply chain & infrastructure stock models for three fossil fuel types


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

coalresults = coal_infra()
coalresults[0].to_csv(OUTPUT_DIR / scenario / "coal_extraction_stock_kg.csv")
coalresults[1].to_csv(OUTPUT_DIR / scenario / "coal_preparation_stock_kg.csv")
coalresults[2].to_csv(OUTPUT_DIR / scenario / "coal_transport_stock_kg.csv")


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

gasresults = gas_infra()
gasresults[0].to_csv(OUTPUT_DIR / scenario / "gas_extraction_stock_kg.csv")
gasresults[1].to_csv(OUTPUT_DIR / scenario / "gas_transport_stock_kg.csv")
gasresults[2].to_csv(OUTPUT_DIR / scenario / "gas_pipelines_length_km.csv")
gasresults[3].to_csv(OUTPUT_DIR / scenario / "gas_processing_stock_kg.csv")


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

oilresults = oil_infra()
oilresults[0].to_csv(OUTPUT_DIR / scenario / "oil_extraction_stock_kg.csv")
oilresults[1].to_csv(OUTPUT_DIR / scenario / "oil_transport_stock_kg.csv")
oilresults[2].to_csv(OUTPUT_DIR / scenario / "oil_pipelines_length_km.csv")
oilresults[3].to_csv(OUTPUT_DIR / scenario / "oil_storage_volume_m3.csv")
oilresults[4].to_csv(OUTPUT_DIR / scenario / "oil_refinery_stock_kg.csv")

print("=== SCRIPT FINISHED SUCCESSFULLY ===")
