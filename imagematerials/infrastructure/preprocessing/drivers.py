"""Load and process the scenario driver inputs (sections 1-4 of v5).

The drivers are the time series that vary by IMAGE scenario:

- VKM/PKM/TKM (road and rail vehicle-kilometres) from the IMAGE TIMER-TRAVEL model
- GDP per capita
- Urban and rural population
- Human Development Index (HDI)

Each loader reads from disk and returns a tidy pandas object indexed by year.
"""
from pathlib import Path

import pandas as pd

from imagematerials.infrastructure.constants import COLUMN_MAPPING
from imagematerials.read_mym import read_mym_df

TKMS_LABEL = ["regions", "Inland ships", "Cargo Trains", "medium truck", "heavy truck",
              "Cargo Planes", "Ships", "empty", "total"]
PKMS_LABEL = ["regions", "walking", "Bikes", "Buses", "Trains", "Cars", "HST",
              "Planes", "total"]


def load_region_list(standard_data_dir: Path):
    """Load the region list from ``region_load.csv`` (one entry per IMAGE region)."""
    region_load_in = pd.read_csv(standard_data_dir / "region_load.csv",
                                  index_col=0, names=None).transpose()
    return list(region_load_in.columns.values)


def load_data_2024(standard_data_dir: Path):
    """Load the per-region 2024 reference values used to anchor regressions."""
    return pd.read_excel(standard_data_dir / "2024-data.xlsx", index_col=0)


def process_vkm(image_dir: Path, region_list):
    """Build total road VKM and rail/HST VKM per region.

    Combines IMAGE PKM/TKM outputs, converts pkm/tkm to vkm using occupancy
    assumptions, sums road modes (Buses + Cars + Trucks), and pivots to
    ``(years × region)``. Also returns rail and HST as separate pivots.
    """
    updated_vkm_pkm = read_mym_df(str(image_dir / "EnergyServices" / "trp_trvl_pkm.out"))
    updated_vkm_tkm = read_mym_df(str(image_dir / "EnergyServices" / "trp_frgt_Tkm.out"))

    updated_vkm_pkm.set_index(["time"], inplace=True)
    updated_vkm_tkm.set_index(["time"], inplace=True)

    updated_vkm_tkm.columns = TKMS_LABEL
    updated_vkm_pkm.columns = PKMS_LABEL

    updated_vkm_tkm["Trucks"] = updated_vkm_tkm["medium truck"] + updated_vkm_tkm["heavy truck"]
    updated_vkm_tkm = updated_vkm_tkm.drop(columns=["medium truck", "heavy truck", "empty", "total"])
    updated_vkm_pkm = updated_vkm_pkm.drop(columns=["walking", "total"])

    updated_vkm_tkm.iloc[:, 1:] = updated_vkm_tkm.iloc[:, 1:] * 1000000
    updated_vkm_pkm.iloc[:, 1:] = updated_vkm_pkm.iloc[:, 1:] * 1000000000000

    vkm = pd.concat([updated_vkm_pkm, updated_vkm_tkm], axis=1)
    vkm = vkm.loc[:, ~vkm.columns.duplicated()]
    vkm = vkm.rename_axis("years")
    vkm = vkm[~vkm["regions"].isin([27, 28])]

    # Conversion of pkm and tkm to vkm
    vkm["Buses_vkm"] = vkm["Buses"] * 0.06 / 23 * 0.43 + vkm["Buses"] * 0.94 / 57 * 0.43
    vkm["Cars_vkm"] = vkm["Cars"] / (4 * 0.45)
    vkm["Bicycle_vkm"] = vkm["Bikes"]
    vkm["Trucks_vkm"] = (vkm["Trucks"] * 0.04 / 0.74
                        + vkm["Trucks"] * 0.48 / 7.95
                        + vkm["Trucks"] * 0.48 / 14)

    vkm["Trains_VKM"] = vkm["Trains"] / 400 + (vkm["Cargo Trains"] / 4165 * 0.45)
    vkm["HST"] = vkm["HST"] / 472
    vkm_rail = vkm[["Trains_VKM", "regions"]]
    vkm_HST = vkm[["HST", "regions"]]

    vkm_rail_pivoted = vkm_rail.pivot(columns="regions", values="Trains_VKM")
    vkm_rail_HST_pivoted = vkm_HST.pivot(columns="regions", values="HST")
    vkm_rail_pivoted.columns = region_list
    vkm_rail_HST_pivoted.columns = region_list

    vkm["total_vkm"] = vkm["Buses_vkm"] + vkm["Cars_vkm"] + vkm["Trucks_vkm"]
    vkm = vkm[["total_vkm", "regions"]]
    vkm_pivoted = vkm.pivot(columns="regions", values="total_vkm")
    vkm_pivoted.columns = region_list

    # COVID correction for road VKM (v5 smoothing mechanism, same as rail)
    if 2020 in vkm_pivoted.index and 2019 in vkm_pivoted.index:
        vkm_pivoted.loc[2020] = vkm_pivoted.loc[2019]
    if 2021 in vkm_pivoted.index and 2022 in vkm_pivoted.index:
        vkm_pivoted.loc[2021] = vkm_pivoted.loc[2022]

    return vkm_pivoted, vkm_rail_pivoted, vkm_rail_HST_pivoted


def process_gdp_and_population(image_dir: Path, region_list):
    """Build per-region per-year GDP per capita, urban population, rural population."""
    gdp_cap = read_mym_df(str(image_dir / "Socioeconomic" / "gdp_pc.scn"))
    urban_pop_df = read_mym_df(str(image_dir / "Socioeconomic" / "URBPOPTOT.out"))
    rural_pop_df = read_mym_df(str(image_dir / "Socioeconomic" / "RURPOPTOT.out"))

    gdp_cap = gdp_cap.iloc[:, :-2]
    urban_pop_df = urban_pop_df.iloc[:, :-2]
    rural_pop_df = rural_pop_df.iloc[:, :-2]

    gdp_cap.columns = region_list
    urban_pop_df.columns = region_list
    rural_pop_df.columns = region_list

    gdp_cap = gdp_cap.rename_axis("years")
    urban_pop_df = urban_pop_df.rename_axis("years")
    rural_pop_df = rural_pop_df.rename_axis("years")

    # COVID correction for GDP (v5: smooth out COVID-related mobility changes)
    # Infrastructure does not decline with GDP during short dips
    if 2020 in gdp_cap.index and 2019 in gdp_cap.index:
        gdp_cap.loc[2020] = gdp_cap.loc[2019]
    if 2021 in gdp_cap.index and 2022 in gdp_cap.index:
        gdp_cap.loc[2021] = gdp_cap.loc[2022]

    gdp_cap_new = gdp_cap
    rural_pop_df = rural_pop_df.rename(columns=COLUMN_MAPPING)
    urban_pop_df = urban_pop_df.rename(columns=COLUMN_MAPPING)

    urban_population = urban_pop_df.sort_index(axis=1)
    rural_population = rural_pop_df.sort_index(axis=1)

    return gdp_cap_new, urban_population, rural_population


def process_hdi(standard_data_dir: Path):
    """Load HDI by IMAGE region, interpolate yearly 1971-2100."""
    hdi = pd.read_excel(standard_data_dir / "HDI-kedi-et-al-2024-SSP2.xlsx", index_col=0)
    hdi = hdi.reset_index()
    if "Area" in hdi.columns:
        hdi = hdi.drop(columns=["Area"])
    if "ISO3" in hdi.columns:
        hdi = hdi.drop(columns=["ISO3"])
    if "ISOCode" in hdi.columns:
        hdi = hdi.drop(columns=["ISOCode"])

    IMAGE_hdi = hdi.groupby("IMAGE-region").mean(numeric_only=True)
    all_years = list(range(1970, 2101))
    IMAGE_hdi = IMAGE_hdi.reindex(columns=all_years)
    IMAGE_hdi = IMAGE_hdi.interpolate(method="linear", axis=1)

    IMAGE_hdi_transposed = IMAGE_hdi.T
    IMAGE_hdi_transposed = IMAGE_hdi_transposed.rename(columns=COLUMN_MAPPING)
    if 1970 in IMAGE_hdi_transposed.index:
        IMAGE_hdi_transposed = IMAGE_hdi_transposed.drop(index=1970)
    sorted_IMAGE_hdi = IMAGE_hdi_transposed.sort_index(axis=1)
    sorted_IMAGE_hdi = sorted_IMAGE_hdi.loc[sorted_IMAGE_hdi.index <= 2100]
    return sorted_IMAGE_hdi
