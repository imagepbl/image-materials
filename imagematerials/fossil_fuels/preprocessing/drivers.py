# Outline:
# 1) Import FUMA model & input data files
# 2) Prepare data files & formatting
# 3) Recieve stock calculations for each supply chain stage (to then be used in capacity.py)

#%% Import FUMA model 
#  A1) FUMA MODEL

import pandas as pd
import numpy as np
from pathlib import Path

from imagematerials.read_mym import read_mym_df
from imagematerials.fossil_fuels.preprocessing.stock_demand.pipelines import pipelines_indexed_stock_growth
from imagematerials.fossil_fuels.preprocessing.stock_demand.transport import transport
from imagematerials.fossil_fuels.preprocessing.stock_demand.extraction import extraction_stocks
from imagematerials.fossil_fuels.preprocessing.stock_demand.storage import storage_indexed_stocks_growth
from imagematerials.fossil_fuels.preprocessing.stock_demand.refinery import refinery_stocks
from imagematerials.fossil_fuels import constants
from imagematerials.fossil_fuels.utils import (
    generate_historic_profile,
)

path_base = Path(__file__).resolve()
while path_base.name != "image-materials":
    path_base = path_base.parent
DATA_DIR = Path(path_base, "data", "raw", "fossil_fuels")
IMAGE_DIR = Path(path_base, "data", "raw", "image")
OUTPUT_DIR = Path(DATA_DIR, "Scenario_data")
scenario = constants.scenario


idx = pd.IndexSlice

# A2) Load input-data files

# IMAGE drivers

primpersec = read_mym_df(
    IMAGE_DIR / scenario / "EnergyFlows" / "PrimPerSec.out"
)
  # primary enery consumption per sector in GJ


final_energy = read_mym_df(
    IMAGE_DIR / scenario / "EnergyFlows" / "final_energy_rt.out"
)  

# final enery consumption per sector in PJ


mining_type_share_coal = pd.read_csv(
    DATA_DIR / "DriversFiles" / "coal_mine_type_share.csv",
    index_col=[0],
)
# % of open-cast vs. underground mining


extraction_type_share_gas = pd.read_csv(
    DATA_DIR / "DriversFiles" / "gas_extraction_type_share.csv", index_col=[0]
)  # % of offshore vs. onshore extraction


extraction_type_share_oil = pd.read_csv(
    DATA_DIR / "DriversFiles" / "oil_extraction_type_share.csv", index_col=[0]


)  # % of offshore vs. onshore extraction

# Pipeline data
gas_pipeline_length = pd.read_csv(
    DATA_DIR / "DriversFiles" / "natural_gas_pipeline_length_km_new.csv", index_col=[0]
)  # Regional Gas Pipeline length (km) for the year 2019


oil_pipeline_length = pd.read_csv(
    DATA_DIR / "DriversFiles" / "oil_pipeline_length_km_new.csv", index_col=[0]
)  # Regional Oil Pipeline length (km) for the year 2019


# Storage data
oil_storage_capacity = pd.read_csv(
    DATA_DIR / "DriversFiles" / "oil_storage_capacity_million_barrels_new.csv", index_col=[0]
)  # Regional Oil storage capacity (million barrels, mb)


oil_storage_materials = pd.read_csv(
    DATA_DIR / "DriversFiles" / "material_intensity" / "oil_storage_material_intensity_kg_per_m3.csv",
    index_col=[0],
)  # Material intensity of oil regional oil distribution (kg material per m3 of storage capacity)

# transport vehicle data
transport_demand_domestic = pd.read_csv(
    DATA_DIR / "DriversFiles" / "Transport_demand_domestic_tkm_per_kg.csv",
    index_col=[0],
    sep=";",
).drop(
    columns="Pipeline"
)  # Domestic transport demand per fuel type (tkm/kg, MIND: in the case of gas it is given in tkm/m3)
transport_demand_internat = pd.read_csv(
    DATA_DIR / "DriversFiles" / "Transport_demand_international_tkm_per_kg.csv",
    index_col=[0],
    sep=";",
).drop(
    columns="Pipeline"
)  # International transport demand per fuel type(tkm/kg, MIND: in the case of gas it is given in tkm/m3)
transport_demand_total = (
    transport_demand_domestic + transport_demand_internat
)

transport_materials_per_vehicle = pd.read_csv(
    DATA_DIR / "DriversFiles" / "material_intensity" / "fossil_fuel_freight_vehicle_material_fraction.csv",
    index_col=[0],
).T  # % of the total weight by material in fossil fuel freight transport vehicles
transport_vehicle_weight_per_tkm = pd.read_csv(
    DATA_DIR / "DriversFiles" / "material_intensity" / "fossil_fuel_freight_vehicle_weight_intensity_g_per_tkm.csv",
    index_col=[0],
    header=None,
    sep=";",
).squeeze()  # average intensity of fossil fuel freight transport vehicles weight (g) in stock per tkm (g/tkm) per year so the stock required to continuously provide the service of transportation

# Mining data & material intensities (in kg material per unit of annual production capacity)
production_intensity_coal = pd.read_csv(
    DATA_DIR / "DriversFiles" / "material_intensity" / "coal_production_intensity.csv", index_col=[0]
)  # material use per unit of production capacity (kg/kg yr-1)
production_intensity_gas = pd.read_csv(
    DATA_DIR / "DriversFiles" / "material_intensity" / "Natural_gas_production_intensity.csv",
    index_col=[0],
)  # material use per unit of production capacity (kg/m3 yr-1)
production_intensity_oil = pd.read_csv(
    DATA_DIR / "DriversFiles" / "material_intensity" / "oil_production_intensity.csv", index_col=[0]
)

# Processing & refinery material intensities (in kg material per unit of annual production capacity)
processing_intensity_gas = pd.read_csv(
    DATA_DIR / "DriversFiles" / "material_intensity" / "gas_processing_plant_intensity_kg_per_m3.csv",
    index_col=[0],
)
refinery_intensity_oil = pd.read_csv(
    DATA_DIR / "DriversFiles" / "material_intensity" / "oil_refinery_material_intensity_kg_per_kg.csv",
    index_col=[0],
)
preparation_intensity_coal = pd.read_csv(
    DATA_DIR / "DriversFiles" / "material_intensity" / "coal_preparation_material_intensity_kg_per_kg.csv",
    index_col=[0],
)
# Pipeline material intensity data (kg/km)
gas_pipeline_composition_trans = pd.read_csv(
    DATA_DIR / "DriversFiles" / "material_intensity" / "gas_pipeline_composition_transmission_critical_kg_per_km.csv",
    index_col=[0],
)  # material use (kg) per km of transmission pipeline (kg/km)
gas_pipeline_composition_distr = pd.read_csv(
    DATA_DIR / "DriversFiles" / "material_intensity" / "gas_pipeline_composition_distribution_kg_per_km.csv",
    index_col=[0],
)  # material use (kg) per km of distribution pipeline (kg/km)
oil_pipeline_composition = pd.read_csv(
    DATA_DIR / "DriversFiles" / "material_intensity" / "oil_pipeline_composition_kg_per_km.csv",
    index_col=[0],
)

# A3) Prepare data files & formatting

# adding index names
primpersec.set_index(
    ["time", "DIM_1", "DIM_2"], inplace=True
)  # index: time, 11 primary fuels, 8 secondary fuels
final_energy.set_index(
    ["time", "DIM_1", "DIM_2"], inplace=True
)  # index: time, 28 regions, 6 sectors, columns: 10 secondary fuel types (region 27 is ignored)



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
# primary_x_to_liquid = fossil_flows_primary.loc[
#     idx[:, ["gas", "coal"], "Liquids"], :
# ]
# all_fossil_flows_sankey = pd.concat(
#     [all_fossil_flows, primary_x_to_liquid]
# ).sort_index(level=[0, 1, 2])

# (Intermediate results) Fuel use data
# all_fossil_flows_sankey.to_csv(
#     OUTPUT_DIR / scenario / "fuel_use_GJ.csv", index=True
# )




#%% FUMA stock calculation
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
    return {
        "extraction": extraction_stock_coal,
        "processing": preparation_stock_coal,
        "transport": transport_total_coal,
    }


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
    return {
        "extraction": extraction_stock_gas,
        "transport": transport_total_gas,
        "pipelines": gas_pipelines,
        "processing": processing_stock_gas,
    }


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
        (oil_use_total * constants.GJ_TO_KG_OIL / constants.FRACTION_REFINING_CAP),
        refinery_intensity_weight_oil,
    )  # IN KG
    return {
        "extraction": extraction_stock_oil,
        "transport": transport_total_oil,
        "pipelines": oil_pipelines,
        "storage": oil_storage,
        "refinery": refinery_stock_oil,
    }

# print("drivers.py ran successfully!")

