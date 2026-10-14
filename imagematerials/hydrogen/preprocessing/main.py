#%%
from pathlib import Path
import pandas as pd
import xarray as xr

import prism
import pym

from imagematerials.concepts import create_h2_graph
from imagematerials.util import resolve_circular_economy_scenario, convert_lifetime
from imagematerials.read_mym import read_mym_df, mymarray_to_xarray
from imagematerials.electricity.utils import (
    interpolate_xr, 
    add_historic_stock,
)
from imagematerials.electricity.constants import (
    STD_LIFETIMES_ELECTR,
    YEAR_FIRST_GRID
)
from imagematerials.hydrogen.constants import (
    YEAR_FIRST_H2_PRODUCTION,
    ELECTROLYSIS_TECHNOLOGIES
)

# --- constants ------------------------------------------------------------------------------------
LAST_EXTERNAL_YEAR = 2030        # external data used up to and including this year
FIRST_IMAGE_YEAR = 2040          # IMAGE data used from this year on (gap is interpolated)
FIRST_UPDATED_YEAR = LAST_EXTERNAL_YEAR + 1        # market shares are overwritten from this year on
TECHS_SPLIT = ["electrolysis_AEL", "electrolysis_SOEL"]    # sub-technologies of the combined AEL/SOEL group
TECHS_ELECTROLYSIS = ["electrolysis_PEMEL", *TECHS_SPLIT]
TECHS_CBF = ["CBF+CCS_tech1", "CBF_tech1"]


def get_preprocessing_data_hydrogen(path_base: str,
                               climate_policy_config: dict,
                               circular_economy_config: dict,
                               resource_efficiency_flags: dict = None
                               ):

    scen = Path(climate_policy_config["config_file_path"]).name
    path_external_data_standard = Path(path_base, "hydrogen", "standard_data")
    path_external_data_scenario = Path(path_base, "hydrogen", scen) #test
    # test if path_external_data_scenario exists and if not set to standard scenario
    if not path_external_data_scenario.exists():
        path_external_data_scenario = Path(path_base, "hydrogen", "standard_data")

    year_start = YEAR_FIRST_GRID
    year_end = 2100

    #===================================================================================================
    # installed capacity of hydrogen production from IMAGE-energy (in MW)
    h2cap_alltypes = mymarray_to_xarray(pym.read_mym(
            climate_policy_config["config_file_path"] /
            climate_policy_config["data_files"]["H2Cap"]),
            dim_1="NRT", dim_2="HydrogenSourcesTotal")

    h2cap_alltypes = prism.Q_(h2cap_alltypes, "MW")
    h2cap_alltypes = h2cap_alltypes.rename({"t": "time","NRT": "Region","HydrogenSourcesTotal": "Type"})
    h2cap_alltypes = h2cap_alltypes.drop_sel(Type="total", Region="WORLD")

    graph_h2 = create_h2_graph()
    types = sorted(["CBF", "CBF+CCS", "electrolysis"])

    h2cap = graph_h2.aggregate_sum(h2cap_alltypes, output_coords=types)

    # TIMER data only start in 1971, so we add a historic tail back to YEAR_FIRST_GRID=1921 #TODO to be adjusted
    h2cap_da_interp = interpolate_xr(h2cap, year_start, year_end) # fills the years before with the first value (which is 0), to align timeseries with other sectors (e.g. electricity)
    # to use the module SharesInflowStocks the higher level technology types are defined as `SuperType`
    # while the subtechnologies for which the shares are provided have the dimension `Type`
    h2cap_da_interp = h2cap_da_interp.rename({"Type": "SuperType"})
    h2cap_da_interp = h2cap_da_interp.transpose("time", "Region", "SuperType")

    #===============================================================================================
    # external data

    # material intensities (kg/MW) -----------------------------------------------------------------
    # material compositions of hydrogen production technologies (kg/MW)
    material_intensities = pd.read_csv(path_external_data_standard / "hydrogen_material_intensities_long.csv",
                            usecols=["time","technology","sub_technology","material","value"])
    # create new column with sub-technology name if available, otherwise technology name
    # material_intensities["Type"] = material_intensities["sub_technology"].fillna(material_intensities["technology"])
    material_intensities["Type"] = (
            material_intensities["technology"] + "_" + material_intensities["sub_technology"]
        )
    material_intensities_da = (
        material_intensities.set_index(["time", "Type", "material"])["value"]
        .to_xarray()
        .rename({"time": "Cohort"})
    ).sortby("Type")
    material_intensities_da = prism.Q_(material_intensities_da, "kg/MW")
    material_intensities_da.name = "HydrogenMaterialIntensities"
    material_intensities_da_interp = interpolate_xr(material_intensities_da, year_start, year_end)


    # lifetimes (yr) -------------------------------------------------------------------------------
    lifetimes = pd.read_csv(path_external_data_standard / "hydrogen_lifetimes.csv",
                            usecols=["time","technology","sub_technology","value"])
    lifetimes["Type"] = (
                lifetimes["technology"] + "_" + lifetimes["sub_technology"]
            )
    lifetimes_da = (
        lifetimes.set_index(["time", "Type"])["value"]
        .to_xarray()
        .rename({"time": "Cohort"})
    ).sortby("Type")
    lifetimes_da = lifetimes_da.expand_dims({"DistributionParams": ["mean", "stdev"]})
    lifetimes_da.name = "GenerationLifetime"

    lifetimes_da_interp = interpolate_xr(lifetimes_da, year_start, year_end) 
    lifetimes_da_interp.loc[dict(DistributionParams="stdev")] = lifetimes_da_interp.loc[dict(DistributionParams="mean")] * STD_LIFETIMES_ELECTR

    # The lifetimes are converted to the proper format for the model (dictionary with keys:distribution name, values:datarrays containing distribution parameters)
    lifetimes_da_interp = convert_lifetime(lifetimes_da_interp)


    # market shares (%) ----------------------------------------------------------------------------
    market_shares_external = pd.read_csv(path_external_data_standard / "hydrogen_market_shares_long.csv",
                            usecols=["time","technology","sub_technology","value"])

    market_shares_external["Type"] = (
                market_shares_external["technology"] + "_" + market_shares_external["sub_technology"]
            )
    market_shares_external = (
        market_shares_external.set_index(["time", "Type"])["value"]
        .to_xarray()
        .rename({"time": "Cohort"})
    )
    market_shares_external = prism.Q_(market_shares_external, "dimensionless")
    market_shares_external.name = "HydrogenMarketShares"

    market_shares_external = interpolate_xr(market_shares_external, year_start, year_end)

    # Calculate merged version of TIMER and external market shares (from literature)

    # --- 1. IMAGE-energy shares (PEMEL vs. AEL/SOEL) ----------------------------------------------
    electrolysis_capacity = h2cap_alltypes.sel(Type=ELECTROLYSIS_TECHNOLOGIES)
    capacity_by_group = (
        graph_h2.aggregate_sum(
            electrolysis_capacity,
            output_coords=["electrolysis_PEMEL", "electrolysis_AEL/SOEL"],
        )
        .sum(dim="Region")
    )
    shares_image = (capacity_by_group / capacity_by_group.sum(dim="Type")).rename(time="Cohort")

    # --- 2. External shares, aggregated to the same groups ----------------------------------------
    shares_electrolysis_external = market_shares_external.sel(
        Type=TECHS_ELECTROLYSIS
    )
    shares_grouped_external = graph_h2.aggregate_sum(
        shares_electrolysis_external, output_coords=["electrolysis_PEMEL", "electrolysis_AEL/SOEL"]
    )

    # --- 3. Internal AEL vs. SOEL split (relative shares within AEL/SOEL) -------------------------
    ael_soel_split = shares_electrolysis_external.sel(
        Type=TECHS_SPLIT
    )
    ael_soel_split = graph_h2.rebroadcast_xarray(ael_soel_split, output_coords=TECHS_SPLIT)
    ael_soel_split = ael_soel_split / ael_soel_split.sum(dim="Type")

    # --- 4. Combine external (<=2030) and IMAGE (>=2040), then interpolate the gap ----------------
    shares_combined = xr.concat(
        [
            shares_grouped_external.sel(Cohort=slice(None, LAST_EXTERNAL_YEAR)),
            shares_image.sel(Cohort=slice(FIRST_IMAGE_YEAR, None)),
        ],
        dim="Cohort",
    )
    shares_combined_interp = interpolate_xr(shares_combined, year_start, year_end)

    # --- 5. Split AEL/SOEL back into AEL and SOEL -------------------------------------------------
    ael_soel_total = shares_combined_interp.sel(Type="electrolysis_AEL/SOEL", drop=True)
    split_shares = [
        (ael_soel_total * ael_soel_split.sel(Type=tech, drop=True)).expand_dims(Type=[tech])
        for tech in TECHS_SPLIT
    ]
    shares_electrolysis = xr.concat([shares_combined_interp, *split_shares], dim="Type").sel(Type=TECHS_ELECTROLYSIS)


    # --- 6. Write back into the market shares ---------------------------------------------------------

    shares_final_non_electrolysis = market_shares_external.sel(Type=TECHS_CBF)
    # merge both xarrays and sort the coordinates of the type dim alphabetically
    shares_final = xr.concat([shares_final_non_electrolysis, shares_electrolysis], dim="Type").sortby("Type") 

    #===================================================================================================
    #  Prep_data File

    # bring preprocessing data into a generic format for the model
    prep_data = {}
    prep_data["stocks"] = h2cap_da_interp
    prep_data["lifetimes"] = lifetimes_da_interp
    prep_data["shares"] = shares_final
    prep_data["material_intensities"] = material_intensities_da_interp
    prep_data["knowledge_graph"] = graph_h2
    prep_data["set_unit_flexible"] = str(prism.U_(prep_data["stocks"])) # prism.U_ gives the unit back

    return prep_data