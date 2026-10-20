"""
inputs.py -- reads ALL input data of the fossil fuel module in one place.

This file only READS data (nothing is calculated here), so it is the place to
look if you want to know which files the model needs.

Usage (in main.py):

    inputs_image = load_inputs_image(climate_policy_config)
    inputs = load_inputs(path_base, climate_policy_config)

    inputs_image["image"]["primpersec"]      # IMAGE output
    inputs["lifetimes"]["extraction"]        # lifetimes, one table per stage
    inputs["materials"]["extraction"]        # material intensities, one table per stage
    inputs["drivers"]["gas_pipeline_length"] # other fossil fuel data

Where do the files come from?
  * IMAGE output files: from the folder given in the config file
    (climate_policy_config["config_file_path"] + the names in [data_files]).
  * All other files: first looked up in the scenario-specific folder
        <path_base>/fossil_fuels/<scenario name>/
    and if the file is not there, in the standard folder
        <path_base>/fossil_fuels/standard_data/
    So a scenario folder only needs to contain the files that differ from standard.
"""

from pathlib import Path
import pandas as pd

from imagematerials.read_mym import read_mym_df


def load_inputs_image(climate_policy_config):

    # ------------------------------------------------------------------
    # 1. IMAGE output (depends on the IMAGE scenario run)
    # ------------------------------------------------------------------

    image_dir = Path(climate_policy_config["config_file_path"])
    data_files = climate_policy_config["data_files"]

    # TODO: use instead: mymarray_to_xarray(pym.read_mym()) to read in as xarrays
    image = {
        # primary energy consumption per sector (GJ)
        "primpersec": read_mym_df(image_dir / data_files["PrimPerSec"]),
        # final energy consumption per sector (PJ)
        "final_energy": read_mym_df(image_dir / data_files["final_energy_rt"]),
    }

    return {"image": image}


def load_inputs(path_base, climate_policy_config):
    # ------------------------------------------------------------------
    # Where to look for files
    # ------------------------------------------------------------------
    image_dir = Path(climate_policy_config["config_file_path"])
    scenario_name = image_dir.name  # e.g. "SSP2_baseline"

    scenario_dir = Path(path_base, "fossil_fuels", scenario_name)
    standard_dir = Path(path_base, "fossil_fuels", "standard_data")

    def read_input_csv(relative_path, **options):
        """Read one csv file: scenario-specific version if it exists, else the standard one."""
        #TODO: check if this tiered approach is working (currently we only have standard data, so not
        #      needed atm)
        scenario_file = scenario_dir / relative_path
        standard_file = standard_dir / relative_path

        if scenario_file.exists():
            return pd.read_csv(scenario_file, **options)
        if standard_file.exists():
            return pd.read_csv(standard_file, **options)

        raise FileNotFoundError(
            f"Input file not found: {relative_path}\n"
            f"  looked in: {scenario_dir}\n"
            f"  and in:    {standard_dir}"
        )


    # ------------------------------------------------------------------
    # 1. Lifetimes (average lifetime in years, per stage)
    # ------------------------------------------------------------------
    lifetimes = {
        "extraction": read_input_csv("extraction/Extraction_lifetimes.csv", index_col=["Year", "Tech Type"]),
        "processing": read_input_csv("processing/Processing_lifetimes_test.csv", index_col=["Year", "Tech Type"]),
        "storage": read_input_csv("processing/Storage_lifetimes.csv", index_col=["Year", "Tech Type"]),
        "transport": read_input_csv("transport/Transport_lifetimes.csv", index_col=["Year", "Tech Type"]),
        "pipelines": read_input_csv("pipelines/Pipelines_lifetimes_edit.csv", index_col=["Year", "Tech Type"]),
    }

    # ------------------------------------------------------------------
    # 2. Material intensities (per stage)
    # ------------------------------------------------------------------
    materials = {
        "extraction": read_input_csv("extraction/Extraction_materials.csv", index_col=["Year", "Tech Type"]),
        "processing": read_input_csv("processing/Processing_materials.csv", index_col=["Year", "Tech Type"]),
        "storage": read_input_csv("processing/Storage_materials.csv", index_col=["Year", "Tech Type"]),
        "transport": read_input_csv("transport/Transport_materials.csv", index_col=["Year", "Tech Type"]),
        "pipelines": read_input_csv("pipelines/Pipelines_materials_edit.csv", index_col=["Year", "Tech Type"]),
    }

    # ------------------------------------------------------------------
    # 3. Other fossil fuel data, used to calculate the capacity (stocks)
    # ------------------------------------------------------------------
    drivers = {
            # % of open-cast vs. underground mining
            "mining_type_share_coal": read_input_csv("drivers/coal_mine_type_share.csv", index_col=[0]),
            # % of offshore vs. onshore extraction
            "extraction_type_share_gas": read_input_csv("drivers/gas_extraction_type_share.csv", index_col=[0]),
            "extraction_type_share_oil": read_input_csv("drivers/oil_extraction_type_share.csv", index_col=[0]),
            # regional pipeline length (km) for the year 2019
            "gas_pipeline_length": read_input_csv("drivers/natural_gas_pipeline_length_km_new.csv", index_col=[0]),
            "oil_pipeline_length": read_input_csv("drivers/oil_pipeline_length_km_new.csv", index_col=[0]),
            # share of the 2 oil pipeline types (onshore & offshore), global (%)
            "on_offshore_oil_pipeline": read_input_csv("drivers/onshore_offshore_oil_pipeline.csv", index_col=[0]),
            # regional oil storage capacity (million barrels)
            "oil_storage_capacity": read_input_csv("drivers/oil_storage_capacity_million_barrels_new.csv", index_col=[0]),
            # material use per unit of production capacity (coal: kg/kg yr-1, gas: kg/m3 yr-1)
            "production_intensity_coal": read_input_csv("drivers/coal_production_intensity.csv", index_col=[0]),
            "production_intensity_gas": read_input_csv("drivers/Natural_gas_production_intensity.csv", index_col=[0]),
            "production_intensity_oil": read_input_csv("drivers/oil_production_intensity.csv", index_col=[0]),
            # material use per unit of processing capacity
            "processing_intensity_gas": read_input_csv("drivers/gas_processing_plant_intensity_kg_per_m3.csv", index_col=[0]),
            "refinery_intensity_oil": read_input_csv("drivers/oil_refinery_material_intensity_kg_per_kg.csv", index_col=[0]),
            "preparation_intensity_coal": read_input_csv("drivers/coal_preparation_material_intensity_kg_per_kg.csv", index_col=[0]),
            # average weight (g) of freight vehicles in stock per tkm, per year
            # (file has no header and uses ";" as separator; .squeeze() turns it into a single column)
            "transport_vehicle_weight_per_tkm": read_input_csv(
                "drivers/fossil_fuel_freight_vehicle_weight_intensity_g_per_tkm.csv", index_col=[0], header=None, sep=";"
            ).squeeze(),
            # transport demand per fuel type (tkm/kg; for gas it is given in tkm/m3)
            "transport_demand_domestic": read_input_csv(
                "drivers/Transport_demand_domestic_tkm_per_kg.csv", index_col=[0], sep=";"
            ).drop(columns="Pipeline"),
            "transport_demand_international": read_input_csv(
                "drivers/Transport_demand_international_tkm_per_kg.csv", index_col=[0], sep=";"
            ).drop(columns="Pipeline"),
        }

    return {
        "lifetimes": lifetimes,
        "materials": materials,
        "drivers": drivers,
    }
