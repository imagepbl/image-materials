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
TECHS_SPLIT = ["AEL", "SOEL"]    # sub-technologies of the combined AEL/SOEL group
TECHS_ELECTROLYSIS = ["PEMEL", *TECHS_SPLIT]
TECHS_CBF = ["CBF+CCS_tech1", "CBF_tech1"]


def get_preprocessing_data_hydrogen(path_base: str,
                               climate_policy_config: dict,
                               circular_economy_config: dict,
                               resource_efficiency_flags: dict = None
                               ):
    
    # # TODO absolute path of file "preprocessing.py" ? current solution can differ depending on IDE used
    # path_current = Path().resolve()
    # path_base = path_current.parent.parent.parent # base path of the project -> image-materials
    # path_base = Path(path_base, "data", "raw")

    scen = Path(climate_policy_config["config_file_path"]).name
    path_external_data_standard = Path(path_base, "hydrogen", "standard_data")
    path_external_data_scenario = Path(path_base, "hydrogen", scen) #test
    # test if path_external_data_scenario exists and if not set to standard scenario
    if not path_external_data_scenario.exists():
        path_external_data_scenario = Path(path_base, "hydrogen", "standard_data")

    year_start = YEAR_FIRST_GRID
    year_end = 2100

    #===================================================================================================
    # Timer data
    h2cap_alltypes = mymarray_to_xarray(pym.read_mym(Path(path_base, "image", "SSP2_baseline", "EnergyServices", "H2Cap.out")),
                            dim_1="NRT", dim_2="HydrogenSourcesTotal")

    h2cap_alltypes = prism.Q_(h2cap_alltypes, "GW")
    h2cap_alltypes = h2cap_alltypes.rename({'t': 'time','NRT': 'Region','HydrogenSourcesTotal': 'Type'})
    h2cap_alltypes = h2cap_alltypes.drop_sel(Type="total", Region="WORLD")

    graph_h2 = create_h2_graph()
    types = ["CBF", "CBF+CCS", "electrolysis"]
    h2cap = graph_h2.aggregate_sum(h2cap_alltypes, output_coords=types)

    # TIMER data only start in 1971, so we add a historic tail back to YEAR_FIRST_GRID=1921 #TODO to be adjusted
    h2cap_da_interp = interpolate_xr(h2cap, year_start, year_end) # fills the years before with the first value (which is 0), to align timeseries with other sectors (e.g. electricity)
    # to use the module SharesInflowStocks the higher level technology types are defined as `SuperType`
    # while the subtechnologies for which the shares are provided have the dimension `Type`
    h2cap_da_interp = h2cap_da_interp.rename({"Type": "SuperType"})
    h2cap_da_interp = h2cap_da_interp.transpose("time", "Region", "SuperType")

    #===================================================================================================
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
    )
    material_intensities_da = prism.Q_(material_intensities_da, "kg/MW")
    material_intensities_da.name = "HydrogenMaterialIntensities"
    # material_intensities_da = knowledge_graph_electr.rebroadcast_xarray(material_intensities_da, output_coords=EPG_SUB_TECHNOLOGIES, dim="Type")
    material_intensities_da_interp = interpolate_xr(material_intensities_da, year_start, year_end)


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

    ##### TEST 1 #######################################################################################
    #%% Calculate merged version of TIMER and CML market shares

    # # data coming from the IMAGE-energy model = h2cap_alltypes
    # # external market share data = market_shares_da_interp
    # h2cap_electrolysis = h2cap_alltypes.sel(Type=ELECTROLYSIS_TECHNOLOGIES)
    # types = ["IMAGE-energy PEMEL", "IMAGE-energy AEL/SOEL"]
    # test = graph_h2.aggregate_sum(h2cap_electrolysis, output_coords=types).sum(dim="Region")
    # ms = test / test.sum(dim="Type")
    # ms = ms.rename({"time": "Cohort"})
    # mapping = {
    #     "IMAGE-energy PEMEL": "PEMEL",
    #     "IMAGE-energy AEL/SOEL": "AEL/SOEL",
    # }
    # ms = ms.assign_coords(Type=[mapping.get(t, t) for t in ms["Type"].values])

    # # cml
    # # 1. 
    # shares_el = market_shares_da_interp.sel(Type=["electrolysis_AEL", "electrolysis_PEMEL", "electrolysis_SOEL"])
    # # 2.
    # shares_el_agg = graph_h2.aggregate_sum(shares_el, output_coords=["PEMEL", "AEL/SOEL"])
    # # 3.
    # shares_ael_soel = shares_el.sel(Type = ["electrolysis_AEL", "electrolysis_SOEL"])
    # mapping = {
    #     "electrolysis_AEL": "AEL",
    #     "electrolysis_SOEL": "SOEL",
    # }
    # shares_ael_soel = shares_ael_soel.assign_coords(Type=[mapping.get(t, t) for t in shares_ael_soel["Type"].values])
    # # shares_ael_soel = shares_ael_soel.rename({"electrolysis_AEL": "AEL", "electrolysis_SOEL": "SOEL"})
    # shares_ael_soel = shares_ael_soel / shares_ael_soel.sum(dim="Type")

    # # 4.-6
    # x2 = xr.concat([shares_el_agg.sel(Cohort=slice(None,2030)), ms.sel(Cohort=slice(2040,None))], dim="Cohort")
    # x2_interp = interpolate_xr(x2, year_start, year_end)

    # # 7.
    # x3 = x2_interp.copy()
    # ael = (x3.sel(Type="AEL/SOEL", drop=True) * shares_ael_soel.sel(Type="AEL", drop=True)).expand_dims(Type=["AEL"])
    # soel = (x3.sel(Type="AEL/SOEL", drop=True)  * shares_ael_soel.sel(Type="SOEL", drop=True)).expand_dims(Type=["SOEL"]) 
    # x3 = xr.concat([x3, ael, soel], dim="Type")

    # # 8.
    # final = market_shares_da_interp.copy()
    # mapping = {
    #     "electrolysis_AEL": "AEL",
    #     "electrolysis_SOEL": "SOEL",
    #     "electrolysis_PEMEL": "PEMEL",
    # }

    # cohorts = final["Cohort"].sel(Cohort=slice(2031, None)).values  # "after 2030"; use 2030 if you want it included

    # for target, source in mapping.items():
    #     final.loc[dict(Type=target, Cohort=cohorts)] = (
    #         x3.sel(Type=source, Cohort=cohorts).values
    #     )



    ##### TEST 2 #######################################################################################
    #%% Calculate merged version of TIMER and CML market shares


    # --- 1. IMAGE-energy shares (PEMEL vs. AEL/SOEL) --------------------------------------------------
    electrolysis_capacity = h2cap_alltypes.sel(Type=ELECTROLYSIS_TECHNOLOGIES)
    capacity_by_group = (
        graph_h2.aggregate_sum(
            electrolysis_capacity,
            output_coords=["PEMEL", "AEL/SOEL"], #"IMAGE-energy PEMEL", "IMAGE-energy AEL/SOEL"
        )
        .sum(dim="Region")
    )
    shares_image = (capacity_by_group / capacity_by_group.sum(dim="Type")).rename(time="Cohort")
    # shares_image = shares_image.assign_coords(
    #     Type=shares_image["Type"].str.replace("IMAGE-energy ", "", regex=False)
    # )

    # --- 2. External shares, aggregated to the same groups --------------------------------------------
    shares_electrolysis_external = market_shares_external.sel(
        Type=[f"electrolysis_{tech}" for tech in TECHS_ELECTROLYSIS]
    )
    shares_grouped_external = graph_h2.aggregate_sum(
        shares_electrolysis_external, output_coords=["PEMEL", "AEL/SOEL"]
    )

    # --- 3. Internal AEL vs. SOEL split (relative shares within AEL/SOEL) -----------------------------
    ael_soel_split = shares_electrolysis_external.sel(
        Type=[f"electrolysis_{tech}" for tech in TECHS_SPLIT]
    )
    ael_soel_split = graph_h2.rebroadcast_xarray(ael_soel_split, output_coords=["AEL", "SOEL"]) # rename
    ael_soel_split = ael_soel_split / ael_soel_split.sum(dim="Type")

    # --- 4. Combine external (<=2030) and IMAGE (>=2040), then interpolate the gap --------------------
    shares_combined = xr.concat(
        [
            shares_grouped_external.sel(Cohort=slice(None, LAST_EXTERNAL_YEAR)),
            shares_image.sel(Cohort=slice(FIRST_IMAGE_YEAR, None)),
        ],
        dim="Cohort",
    )
    shares_combined_interp = interpolate_xr(shares_combined, year_start, year_end)

    # --- 5. Split AEL/SOEL back into AEL and SOEL -----------------------------------------------------
    ael_soel_total = shares_combined_interp.sel(Type="AEL/SOEL", drop=True)
    split_shares = [
        (ael_soel_total * ael_soel_split.sel(Type=tech, drop=True)).expand_dims(Type=[tech])
        for tech in TECHS_SPLIT
    ]
    shares_electrolysis = xr.concat([shares_combined_interp, *split_shares], dim="Type").sel(Type=TECHS_ELECTROLYSIS)


    # --- 6. Write back into the market shares ---------------------------------------------------------

    shares_final_non_electrolysis = market_shares_external.sel(Type=TECHS_CBF)
    shares_final = xr.concat([shares_final_non_electrolysis, shares_electrolysis], dim="Type")
    # rename again to electrolysis_AEL ?

    ########################################################################################
    

    # lifetimes (yr) ----------------------------------------------------------------------------
    # 
    lifetimes = pd.read_csv(path_external_data_standard / "hydrogen_lifetimes.csv",
                            usecols=["time","technology","sub_technology","value"])
    lifetimes["Type"] = (
                lifetimes["technology"] + "_" + lifetimes["sub_technology"]
            )
    lifetimes_da = (
        lifetimes.set_index(["time", "Type"])["value"]
        .to_xarray()
        .rename({"time": "Cohort"})
    )
    lifetimes_da = lifetimes_da.expand_dims({"DistributionParams": ["mean", "stdev"]})
    lifetimes_da.name = "GenerationLifetime"

    lifetimes_da_interp = interpolate_xr(lifetimes_da, year_start, year_end) 
    lifetimes_da_interp.loc[dict(DistributionParams="stdev")] = lifetimes_da_interp.loc[dict(DistributionParams="mean")] * STD_LIFETIMES_ELECTR

    # The lifetimes are converted to the proper format for the model (dictionary with keys:distribution name, values:datarrays containing distribution parameters)
    lifetimes_da_interp = convert_lifetime(lifetimes_da_interp)

    #===================================================================================================
    #%%  Prep_data File

    print("h2cap_da_interp \n", h2cap_da_interp.SuperType.values)
    print("h2cap_da_interp dims \n", h2cap_da_interp.dims)
    print("h2cap_da_interp \n", h2cap_da_interp)
    print("lifetimes_da_interp \n", lifetimes_da_interp)
    print("shares_final \n", shares_final.Type.values)
    print("material_intensities_da_interp \n", material_intensities_da_interp.Type.values)

    # bring preprocessing data into a generic format for the model
    prep_data = {}
    prep_data["stocks"] = h2cap_da_interp
    prep_data["lifetimes"] = lifetimes_da_interp
    prep_data["shares"] = shares_final
    prep_data["material_intensities"] = material_intensities_da_interp
    prep_data["knowledge_graph"] = graph_h2
    prep_data["set_unit_flexible"] = str(prism.U_(prep_data["stocks"])) # prism.U_ gives the unit back

    return prep_data