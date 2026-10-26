"""
This file contains the calculations of the infrastructure stock in the fossil fuel supply chain for
coal (infra_coal()), gas (infra_gas()) and oil (infra_oil()). It takes the IMAGE outout of primary
and final energy demand for coal, gas and oil per year and transforms this into requierements in 
terms of infrastrcuture stock (extraction, processing/refining, transport, pipelines and storage).

# ##################################################################################################
#   SUPPLY CHAIN STOCK CALCULATIONS:
#   Supply Chain Elements:          Coal    Oil     Gas
#   1) Extraction infrastructure     X       X       X
#   2) transport by vehicles         X       X       X
#   3) transport by pipelines                X       X
#   4) Storage                               X       x
#   5) Processing & Refining         x       X       X
####################################################################################################
"""

import pandas as pd

from imagematerials.fossil_fuels.preprocessing.stock_demand.pipelines import pipelines_indexed_stock_growth
from imagematerials.fossil_fuels.preprocessing.stock_demand.transport import transport
from imagematerials.fossil_fuels.preprocessing.stock_demand.extraction import extraction_stocks
from imagematerials.fossil_fuels.preprocessing.stock_demand.storage import storage_indexed_stocks_growth
from imagematerials.fossil_fuels.preprocessing.stock_demand.processing import processing_stocks
from imagematerials.fossil_fuels import constants
from imagematerials.fossil_fuels.utils import (
    generate_historic_profile,
)

idx = pd.IndexSlice

####################################################################################################
#%% Process IMAGE input (primary & final energy demand) to be used in stock demand calculations
####################################################################################################

# Model drivers (fossil fuel use scenarios from IMAGE, by region towards 2100)
# For now, total regional consumption (as primary or final energy) is the driver of infrastructure 
# requirements. In the future, this may be expanded.

# Add labels; don't change order as they are selected later based on position
primary_fuel_types_ff = [
    "coal",
    "conv oil",
    "unconv oil",
    "gas"
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
final_fuel_types_ff2 = [
    "Solids",
    "Liquids1",
    "Liquids2",
    "Sec gas",
]
sectors_ff = [
    "industry",
    "transport",
    "residential",
    "services",
    "other",
    "non-energy",
    "bunkers",
]

def build_all_fossil_flows(
        primpersec: pd.DataFrame,
        final_energy: pd.DataFrame,
    ) -> pd.DataFrame:
    """Combine the IMAGE energy flow tables into one table of fossil fuel use (GJ).

    Fossil fuel use (coal, gas, oil) has two parts, which are added together here:
        1. Final use: final energy consumption per sector (industry, transport, ...),
           converted from PJ to GJ.
        2. Conversion use: primary energy used to produce electricity, hydrogen and
           secondary heat. 
           TODO: check this: Flows into the other secondary carriers (solids, liquids,
           secondary gas) are not taken from the primary table, because their final
           use is already counted in part 1.

    Conventional and unconventional oil, and the two liquid types, are each merged
    into a single "oil" category.

    Parameters
    ----------
    primpersec: pd.DataFrame
        IMAGE output "PrimPerSec" as read by read_mym_df. Primary energy
        consumption in GJ. Columns: time, DIM_1 (primary fuel), DIM_2 (secondary
        carrier), and one column per region.
    final_energy: pd.DataFrame
        IMAGE output "final_energy_rt" as read by read_mym_df. Final
        energy consumption in PJ. Columns: time, DIM_1 (region), DIM_2 (sector),
        and one column per secondary fuel type.

    Returns
    -------
    all_fossil_flows: pd.DataFrame
        Fossil fuel use in GJ. Index: (time, fuel, use category), where fuel is "coal",
        "gas" or "oil", and the use category is either a sector (final use) or one of
        "Electricity", "Hydrogen", "Sec heat" (conversion use). Columns: regions
        (1 to constants.NUM_REGIONS=26).

    Notes
    -----
    Uses the label lists defined above this function, and the constants
    START_YEAR, END_YEAR, NUM_REGIONS and PJ_TO_GJ. The positional selectors
    (e.g. [1, 2, 3, 4]) must match the order of those label lists.
    TODO: read in primpersec & final_energy with mymarray_to_xarray(pym.read_mym()) directly as 
    xarrays and use the knowledge_graph to group categories

    """
    time_dimension = [
        x for x in range(constants.START_YEAR, constants.END_YEAR + 1, 1)
    ]
    regions_counter = [x for x in range(1, constants.NUM_REGIONS + 1, 1)]
    index_pps = pd.MultiIndex.from_product(
        [time_dimension, primary_fuel_types_ff, final_fuel_types_ff1],
        names=["time", "primary fuels", "secondary fuels"],
    )
    index_fin = pd.MultiIndex.from_product(
        [time_dimension, regions_counter, sectors_ff],
        names=["time", "regions", "sectors"],
    )

    # adding index names (returns new frames instead of modifying the inputs in place)
    primpersec = primpersec.set_index(["time", "DIM_1", "DIM_2"])
    # index: time, 11 primary fuels, 8 secondary fuels
    final_energy = final_energy.set_index(["time", "DIM_1", "DIM_2"])
    # index: time, 28 regions, 6 sectors, columns: 10 secondary fuel types (region 27 is ignored)

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

    return all_fossil_flows


def select_fuel(all_fossil_flows: pd.DataFrame,
                fuel: str
                ) -> pd.DataFrame:
    """Return the total use of one fossil fuel per year and region.

    Sums over all use categories (sectors and conversion carriers).

    Parameters
    ----------
    all_fossil_flows: pd.DataFrame
        Table from build_all_fossil_flows(), in GJ.
    fuel: str
        One of "coal", "gas" or "oil".

    Returns
    -------
    fuel_use_total: pd.DataFrame
        Fuel use in GJ. Index: time, columns: regions.

    """
    fuel_use_total = (
        all_fossil_flows.loc[idx[:, fuel, :], :].groupby(level=[0]).sum()
    )  # GJ
    
    return fuel_use_total


####################################################################################################
#%% Capacity/stock demand calculations
####################################################################################################

def coal_infra(
        all_fossil_flows: pd.DataFrame,
        drivers: dict[str, pd.DataFrame | pd.Series],
        ) -> dict[str, pd.DataFrame]:
    """Calculate the coal infrastructure stocks (extraction, preparation, transport).

    Coal use is taken from the fossil fuel flows, smoothed with a 10-year centered
    moving average, converted from GJ to kg, and then used to derive the stock of
    each infrastructure type per region and year.

    Parameters
    ----------
    all_fossil_flows: Fossil fuel use in GJ, as built by build_all_fossil_flows().
        Index: (time, fuel, use category), columns: regions.
    drivers: Input tables from inputs["drivers"] (see inputs.py). The keys used here:
        - "production_intensity_coal": material use per unit of mining capacity,
            summed over all materials to get the total weight per unit.
        - "preparation_intensity_coal": the same for coal preparation plants.
        - "mining_type_share_coal": share of open-cast vs. underground mining.
        - "transport_demand_domestic", "transport_demand_international":
            transport demand per fuel type (tkm/kg), added up to the total demand.
        - "transport_vehicle_weight_per_tkm": freight vehicle weight per tkm,
            as a Series (a single column).

    Returns
    -------
    Dictionary with one table per supply chain stage, all in kg (total weight of infrastructure):
        - "extraction": coal mines (open cast, underground)
        - "processing": coal preparation plants
        - "transport": freight transport of coal
    Each table has an index of (type, time, fuel, stage, unit) and one column per region.

    """
    fuel = "coal"
    # total material weight per unit of capacity (sum over all materials)
    production_intensity_weight_coal = drivers["production_intensity_coal"].sum(axis=1)
    preparation_intensity_weight_coal = drivers["preparation_intensity_coal"].sum(axis=1)
    transport_demand_total = drivers["transport_demand_domestic"] + drivers["transport_demand_international"]

    coal_use_total = generate_historic_profile(
        select_fuel(all_fossil_flows, fuel), fuel
    )  # in GJ
    coal_use_total = coal_use_total.rolling(
        window=10, min_periods=2, center=True
    ).mean()  # coal_use_total_ma moving average of coal_use_total
    extraction_stock_coal = extraction_stocks(
        fuel,
        (coal_use_total * constants.GJ_TO_KG_COAL),
        production_intensity_weight_coal,
        drivers["mining_type_share_coal"],
    )  # in kg (total weight of infrastructure)
    preparation_stock_coal = processing_stocks(
        fuel,
        (coal_use_total * constants.GJ_TO_KG_COAL),
        preparation_intensity_weight_coal,
    )  # in kg (total weight of infrastructure)
    transport_total_coal = transport(
        fuel,
        coal_use_total * constants.GJ_TO_KG_COAL,
        transport_demand_total,
        drivers["transport_vehicle_weight_per_tkm"],
    )  # in kg

    return {
        "extraction": extraction_stock_coal,
        "processing": preparation_stock_coal,
        "transport": transport_total_coal,
    }


def gas_infra(
        all_fossil_flows: pd.DataFrame,
        drivers: dict[str, pd.DataFrame | pd.Series],
        ) -> dict[str, pd.DataFrame]:
    """Calculate the gas infrastructure stocks (extraction, processing, pipelines, transport).

    Gas use is taken from the fossil fuel flows, smoothed with a 10-year centered
    moving average, converted from GJ to m3, and then used to derive the stock of
    each infrastructure type per region and year.

    Parameters
    ----------
    all_fossil_flows: Fossil fuel use in GJ, as built by build_all_fossil_flows().
        Index: (time, fuel, use category), columns: regions.
    drivers: Input tables from inputs["drivers"] (see inputs.py). The keys used here:
        - "production_intensity_gas": material use per unit of extraction capacity,
            summed over all materials to get the total weight per unit.
        - "processing_intensity_gas": the same for gas preparation plants.
        - "extraction_type_share_gas": share of onshore versus offshore extraction.
        - "transport_demand_domestic", "transport_demand_international":
            transport demand per fuel type (tkm/kg), added up to the total demand.
        - "transport_vehicle_weight_per_tkm": freight vehicle weight per tkm,
            as a Series (a single column).

    Returns
    -------
    Dictionary with one table per supply chain stage, all in kg (total weight of infrastructure):
        - "extraction": gas extraction (onshore, offshore)
        - "processing": gas preparation plants
        - "transport": freight transport of gas
        - "pipelines": pipeline transport of gas
    Each table has an index of (type, time, fuel, stage, unit) and one column per region.

    """
    fuel = "gas"
    # total material weight per unit of capacity (sum over all materials)
    production_intensity_weight_gas = drivers["production_intensity_gas"].sum(axis=1)
    processing_intensity_weight_gas = drivers["processing_intensity_gas"].sum(axis=1)
    transport_demand_total = drivers["transport_demand_domestic"] + drivers["transport_demand_international"]

    gas_use_total = generate_historic_profile(
        select_fuel(all_fossil_flows, fuel), fuel
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
        drivers["extraction_type_share_gas"],
    )  # in kg (total weight of infrastructure)

    transport_total_gas = transport(
        fuel,
        gas_use_total * constants.GJ_TO_M3_GAS,
        transport_demand_total,
        drivers["transport_vehicle_weight_per_tkm"],
    )  # in kg
    gas_pipelines = pipelines_indexed_stock_growth(
        fuel, gas_use_total, drivers["gas_pipeline_length"], baseyear
    )  # in km
    processing_stock_gas = processing_stocks(
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


def oil_infra(
        all_fossil_flows: pd.DataFrame,
        drivers: dict[str, pd.DataFrame | pd.Series],
        ) -> dict[str, pd.DataFrame]:
    """Calculate the oil infrastructure stocks (extraction, refinery, pipelines, transport, storage).

    Oil use is taken from the fossil fuel flows, smoothed with a 10-year centered
    moving average, converted from GJ to kg (except for oil storage), 
    and then used to derive the stock of each infrastructure type per region and year.

    Parameters
    ----------
    all_fossil_flows: Fossil fuel use in GJ, as built by build_all_fossil_flows().
        Index: (time, fuel, use category), columns: regions.
    drivers: Input tables from inputs["drivers"] (see inputs.py). The keys used here:
        - "production_intensity_oil": material use per unit of extraction capacity,
            summed over all materials to get the total weight per unit.
        - "refinery_intensity_oil": the same for oil preparation plants.
        - "extraction_type_share_oil": share of onshore versus offshore extraction.
        - "transport_demand_domestic", "transport_demand_international":
            transport demand per fuel type (tkm/kg), added up to the total demand.
        - "transport_vehicle_weight_per_tkm": freight vehicle weight per tkm,
            as a Series (a single column).
        - "oil_storage_capacity": oil storage capacity per region for crude and product

    Returns
    -------
    Dictionary with one table per supply chain stage, all in kg (total weight of infrastructure):
        - "extraction": oil extraction (onshore, offshore)
        - "processing": oil preparation plants/refineries
        - "transport": freight transport of oil
        - "pipelines": pipeline transport of oil
        - "storage": oil storage
    Each table has an index of (type, time, fuel, stage, unit) and one column per region.

    """
    fuel = "oil"
    # total material weight per unit of capacity (sum over all materials)
    production_intensity_weight_oil = drivers["production_intensity_oil"].sum(axis=1)
    refinery_intensity_weight_oil = drivers["refinery_intensity_oil"].sum(axis=1)
    transport_demand_total = drivers["transport_demand_domestic"] + drivers["transport_demand_international"]

    oil_use_total = generate_historic_profile(
        select_fuel(all_fossil_flows, fuel), fuel
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
        drivers["extraction_type_share_oil"],
    )  # in kg (total weight of infrastructure)
    transport_total_oil = transport(
        fuel,
        oil_use_total * constants.GJ_TO_KG_OIL,
        transport_demand_total,
        drivers["transport_vehicle_weight_per_tkm"],
    )  # in kg
    oil_pipelines = pipelines_indexed_stock_growth(
        fuel, oil_use_total, drivers["oil_pipeline_length"], baseyear
    )  # in km
    oil_storage = storage_indexed_stocks_growth(
        fuel,
        oil_use_total,
        drivers["oil_storage_capacity"] * constants.MB_TO_M3,
        baseyear,
    )  # in m3
    refinery_stock_oil = processing_stocks(
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


####################################################################################################
#%% wraper function
####################################################################################################

def compute_all_infra(
        inputs_image: dict[str, dict[str, pd.DataFrame]],
        inputs: dict[str, dict[str, pd.DataFrame | pd.Series]],
    ) -> dict[str, dict[str, pd.DataFrame]]:
    """Calculate the infrastructure stocks of all three supply chains, once.

    Call this once and pass the result on, instead of calling coal_infra(), gas_infra() and 
    oil_infra() separately for every stage.

    Parameters
    ----------
    inputs_image: IMAGE data. Must contain inputs_image["image"]["primpersec"] and
        inputs_image["image"]["final_energy"] (see inputs.py).
    inputs: Other input data as returned by load_inputs() in inputs.py. Only
        inputs["drivers"] is used here.

    Returns
    -------
    Dictionary with the keys "coal", "gas" and "oil". Each holds a dictionary of
    stock tables per stage, e.g. result["oil"]["storage"]. Coal has "extraction",
    "processing" and "transport"; gas additionally has "pipelines"; oil has
    "extraction", "refinery", "transport", "pipelines" and "storage".
    """
    all_fossil_flows = build_all_fossil_flows(
        inputs_image["image"]["primpersec"], inputs_image["image"]["final_energy"]
    )
    drivers = inputs["drivers"]
    return {
        "coal": coal_infra(all_fossil_flows, drivers),
        "gas": gas_infra(all_fossil_flows, drivers),
        "oil": oil_infra(all_fossil_flows, drivers),
    }