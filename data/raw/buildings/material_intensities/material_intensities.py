import pandas as pd
import numpy as np


# dictionary that maps RASMI R32 Regions to IMAGE R26 Regions
image_to_rasmi = {
    1: 'OECD_CAN',
    2: 'OECD_USA',
    3: 'LAM_MEX',
    4: 'LAM_LAM-L',
    5: 'LAM_BRA',
    6: 'LAM_LAM-L',
    7: ['MAF_MEA-H', 'MAF_MEA-M', 'MAF_NAF'],
    8: ['MAF_SSA-L', 'MAF_SSA-M'],
    9: ['MAF_SSA-L', 'MAF_SSA-M'],
    10: 'MAF_SAF',
    11: ['OECD_EFTA', 'OECD_EU12-H', 'OECD_EU12-M', 'OECD_EU15'],
    12: 'REF_EEU-FSU',
    13: 'OECD_TUR',
    14: ['REF_EEU-FSU', 'ASIA_TWN'],
    15: ['OECD_EEU', 'REF_CAS'],
    16: 'REF_RUS',
    17: ['MAF_MEA-H', 'MAF_MEA-M'],
    18: ['ASIA_IND', 'ASIA_OAS-CPA', 'ASIA_PAK'],
    19: 'OECD_KOR',
    20: 'ASIA_CHN',
    21: ['ASIA_OAS-L', 'ASIA_OAS-M'],
    22: 'ASIA_IDN',
    23: 'OECD_JPN',
    24: 'OECD_AUNZ',
    25: ['ASIA_OAS-L', 'ASIA_OAS-M'],
    26: ['MAF_SSA-L', 'MAF_SSA-M']
}

image_to_material_cities = {
    1: ["CAN"],
    2: ["USA", "SPM", "GUM", "MNP"],  # MNP assumed
    3: ["MEX"],
    4: ["ATG", "BRB", "BMU", "BHS", "BLZ", "CYM", "CRI", "CUB", "DMA", "DOM",
        "SLV", "GRD", "GTM", "HTI", "HND", "JAM", "MTQ", "MSR", "ABW", "AIA",
        "CUW", "NIC", "PAN", "PRI", "KNA", "LCA", "SXM", "TTO", "VCT", "VGB",
        "VIR", "GLP", "TCA", "BES", "BLM", "MAF"],
    5: ["BRA"],
    6: ["ARG", "BOL", "CHL", "COL", "ECU", "GUF", "FLK", "GUY", "SUR", "PRY",
        "PER", "URY", "VEN"],
    7: ["DZA", "EGY", "LBY", "MAR", "TUN", "ESH"],
    8: ["BEN", "BFA", "CPV", "CMR", "CAF", "TCD", "CIV", "COD", "GMB", "GHA",
        "GIN", "GNB", "LBR", "MLI", "MRT", "NER", "NGA", "COG", "SHN", "SEN",
        "SLE", "TGO", "STP", "GNQ", "GAB"],
    9: ["BDI", "COM", "DJI", "ERI", "ETH", "KEN", "MDG", "MUS", "REU", "RWA",
        "SYC", "SOM", "SSD", "SDN", "UGA",
        "MYT"],
    10: ["ZAF"],
    11: ["ISL", "LIE", "NOR", "CHE", "MLT", "AND", "AUT", "BEL", "DNK", "FRO",
         "FIN", "FRA", "DEU", "GIB", "GRC", "IRL", "ITA", "LUX", "NLD", "PRT",
         "SMR", "ESP", "SWE", "GBR",
         "SJM", "GGY", "IMN", "JEY", "ALA"],
    12: ["ALB", "BIH", "HRV", "MNE", "MKD", "SRB", "CYP", "CZE", "EST", "HUN",
         "POL", "SVK", "SVN", "BGR", "LVA", "LTU", "ROU",
         "XKO", "XAD", "ZNC"],
    13: ["TUR"],
    14: ["BLR", "MDA", "UKR"],
    15: ["KAZ", "KGZ", "TJK", "TKM", "UZB"],
    16: ["ARM", "AZE", "GEO", "RUS", "SGS"],
    17: ["BHR", "ISR", "KWT", "OMN", "QAT", "SAU", "ARE", "IRN", "IRQ", "JOR",
         "LBN", "PSE", "SYR", "YEM"],
    18: ["IND"],
    19: ["PRK", "KOR"],
    20: ["CHN", "MNG", "TWN"],
    21: ["MMR", "BRN", "KHM", "LAO", "MYS", "PHL", "SGP", "THA", "VNM", "TLS"],
    22: ["PNG", "IDN"],
    23: ["JPN"],
    24: ["AUS", "CXR", "CCK", "COK", "NZL", "NFK", "ASM", "SLB", "FJI", "FSM",
         "PYF", "KIR", "NCL", "NIU", "VUT", "NRU", "TKL", "TON", "TUV", "WLF",
         "WSM", "PLW", "MHL",
         "UMI"],
    25: ["BGD", "LKA", "AFG", "BTN", "MDV", "NPL", "PAK"],
    26: ["AGO", "GRL", "MOZ", "MWI", "LSO", "BWA", "TZA", "NAM", "SWZ", "ZMB", "ZWE"],
}

housing_type_image_to_rasmi = {
    1: 'RS',  # detached house to residential single family
    2: 'RS',  # semi detached house to residential single family
    3: 'RM',  # appartement to residential multi-family
    4: 'RM',  # high-rise to residential multi-family
}

housing_type_rasmi_to_image = {
    'RS': [1, 2],  # detached house and semi detached house to
    'RM': [3, 4]   # appartement and high-rise to
}

housing_type_to_rasmi_building_structure = {
    1: ['C', 'M', 'S', 'T'],  # assumption that detached housing are average all structures
    2: ['C', 'M', 'S', 'T'],  # assumption that semi detached housing are average all structures
    3: ['C', 'S'],  # assumption that appartement are only made out of cement and steel structures
    4: ['C', 'S']  # assumption that high-rise are made out of cement and steel structures
}

material_list_rasmi = ["steel", "concrete", "wood", "copper", "aluminum", "glass", "brick"]
mis_list_target = ["steel", "concrete", "wood", "copper", "aluminium", "glass", "brick"]

# regions where total steel consumption is higher than what is estimated by image-materials sectors buildings & vehicles
STEEL_REGIONS_ADAPT_ENABLED = True
steel_regions_adapt = [4, 8, 9, 22, 25, 26] if STEEL_REGIONS_ADAPT_ENABLED else []


def load_mi_rasmi():
    """Read in RASMI material intensity data."""
    return pd.read_excel(
        "MI_ranges_20230905.xlsx", index_col=[0, 1, 2, 3, 4],
        sheet_name=["concrete", "brick", "wood", "steel", "glass", "plastics", "aluminum", "copper"]
    )


def load_mi_image_mat():
    """Read in old IMAGE-Materials CP material intensities and clear the data."""
    mi_image_mat = pd.read_csv("Building_materials_old.csv", index_col=[0, 1, 2])
    mi_image_mat.loc[:, :] = np.nan
    return mi_image_mat


def load_mi_image_mat_commercial():
    return pd.read_csv("materials_commercial_old.csv", index_col=[0, 1, 2])


def load_material_cities():
    """Read in MaterialCities data, indexed by GID_0 with only the GFA_* columns kept."""
    material_cities = pd.read_csv("MaterialCities_adm_gfa_ADM0.csv", sep=";")
    material_cities = material_cities.set_index("GID_0")
    material_cities = material_cities.drop(columns=["NLC_count", "population", "SSP5", "SSP32", "ADM0_NAME", "ADM0_area_sq_km"])
    return material_cities


def aggregate_material_cities_to_image_regions(material_cities: pd.DataFrame):
    """Sum GFA_* columns per country (GID_0) up to IMAGE regions, using image_to_material_cities."""
    gid_0_to_image_region = {
        gid_0: image_region
        for image_region, gid_0_list in image_to_material_cities.items()
        for gid_0 in gid_0_list
    }

    material_cities_image = material_cities.copy()
    material_cities_image["image_region"] = material_cities_image.index.map(gid_0_to_image_region)
    material_cities_image = material_cities_image.groupby("image_region").sum()
    material_cities_image.index.name = "Region"
    return material_cities_image


def structure_type_shares(material_cities_image: pd.DataFrame, housing_prefix: str):
    """
    Share of concrete (C), masonry (M), steel (S) and timber (T) structure types per IMAGE region,
    based on GFA_<housing_prefix>_* columns. The U (unknown) column is excluded from the shares.

    housing_prefix: "RS" (residential single-family) or "RM" (residential multi-family)
    """
    structure_cols = {
        "concrete": f"GFA_{housing_prefix}_C",
        "masonry": f"GFA_{housing_prefix}_M",
        "steel": f"GFA_{housing_prefix}_S",
        "timber": f"GFA_{housing_prefix}_T",
    }

    gfa = material_cities_image[list(structure_cols.values())]
    shares = gfa.div(gfa.sum(axis=1), axis=0)
    shares = shares.rename(columns={v: k for k, v in structure_cols.items()})
    return shares


def expand_image_mat_mi(mi_image_mat: pd.DataFrame):
    # also get 2030 data in there
    # Get all unique index values for Region and HousingType
    regions = mi_image_mat.index.get_level_values(1).unique()
    housing_types = mi_image_mat.index.get_level_values(2).unique()

    # Create new index tuples for 2030
    new_index = pd.MultiIndex.from_product([[2030], regions, housing_types], names=mi_image_mat.index.names)

    # Add missing 2030 rows if not present
    mi_image_mat = mi_image_mat.reindex(mi_image_mat.index.union(new_index))

    # Now you can safely assign values for 2030
    mi_image_mat.loc[(2030, slice(None), slice(None)), :] = mi_image_mat.loc[(2020, slice(None), slice(None)), :].values
    return mi_image_mat


structure_code_to_share_name = {
    "C": "concrete",
    "M": "masonry",
    "S": "steel",
    "T": "timber",
}


def effective_structure_weights(structure_shares: pd.DataFrame, allowed_structures: list,
                                 nonzero_structures_by_region: pd.DataFrame = None):
    """
    Effective weight per allowed structure type, for every region in structure_shares, renormalized
    over allowed_structures. Mirrors the fallback used in weighted_structure_mi: if the allowed
    structures have zero combined share in a region, falls back to an equal-weight average over
    whichever allowed structures have non-zero RASMI data in that region (nonzero_structures_by_region,
    a region-indexed DataFrame of bools per structure code); if that's not supplied, falls back to an
    equal-weight average over all allowed_structures.
    """
    share_names = [structure_code_to_share_name[s] for s in allowed_structures]
    weights = structure_shares[share_names].copy()
    weights.columns = allowed_structures

    row_sums = weights.sum(axis=1)
    zero_mask = (row_sums == 0) | row_sums.isna()

    normalized = weights.div(row_sums, axis=0)

    if zero_mask.any():
        for region in weights.index[zero_mask]:
            if nonzero_structures_by_region is not None and region in nonzero_structures_by_region.index:
                available = [s for s in allowed_structures if nonzero_structures_by_region.loc[region, s]]
            else:
                available = allowed_structures
            if not available:
                normalized.loc[region, :] = np.nan
                continue
            normalized.loc[region, :] = 0.0
            normalized.loc[region, available] = 1.0 / len(available)

    return normalized


def weighted_structure_mi(filtered_mis: pd.Series, structure_shares: pd.DataFrame, image_region: int,
                           allowed_structures: list):
    """
    Weight a structure-indexed MI series by each structure type's relative GFA share in image_region,
    renormalized over only the structures allowed for the housing type (housing_type_to_rasmi_building_structure).
    Falls back to a plain mean if shares aren't available (e.g. region missing from material_cities_image),
    or if the allowed structures have zero combined GFA share in the region (renormalizes over whichever
    allowed structures do have non-zero share instead).
    """
    mi_by_structure = filtered_mis.groupby(filtered_mis.index.get_level_values("structure")).mean()

    if image_region not in structure_shares.index:
        nonzero = mi_by_structure[mi_by_structure != 0]
        return nonzero.mean() if not nonzero.empty else np.nan

    share_names = [structure_code_to_share_name[s] for s in allowed_structures]
    weights = structure_shares.loc[image_region, share_names]

    if weights.sum() == 0 or pd.isna(weights.sum()):
        # none of the allowed structures have GFA in this region: fall back to a plain
        # mean over whichever allowed structures have non-zero RASMI data
        available = [s for s in allowed_structures
                     if s in mi_by_structure.index and pd.notna(mi_by_structure[s]) and mi_by_structure[s] != 0]
        if not available:
            return np.nan
        return mi_by_structure[available].mean()

    weights = weights / weights.sum()

    weighted_sum = 0.0
    for structure_code, share_name in zip(allowed_structures, share_names):
        if structure_code not in mi_by_structure.index or pd.isna(mi_by_structure[structure_code]):
            continue
        weighted_sum += mi_by_structure[structure_code] * weights[share_name]

    return weighted_sum


def replace_old_mis_with_rasmi(mi_image_mat: pd.DataFrame, mi_rasmi: pd.DataFrame, material_list_rasmi: list,
                                rs_structure_shares: pd.DataFrame, rm_structure_shares: pd.DataFrame,
                                start_year: int = 2020, target_year: int = 2050, data_value="p_50"):
    """
    Replace old material intensities in IMAGE-Materials with RASMI values for concrete.

    material_name: str, non capitalised from rasmi
    mis_list_target_name: str, material intensity in IMAGE-Materials
    data_value: str, can also be 0, 25, 75 or 100 percentile
    rs_structure_shares / rm_structure_shares: per-region structure type shares from structure_type_shares(),
        used to weight the MI values instead of taking a plain mean across structures.
    """
    # standardize column names at the top of the function
    mi_image_mat.columns = [c.lower().replace("aluminum", "aluminium") for c in mi_image_mat.columns]
    mi_image_mat_update = mi_image_mat.copy()

    for material_name in material_list_rasmi:
        # ensure lower case
        if material_name.lower() == "aluminum":
            material_name_image = "aluminium"
        else:
            material_name_image = material_name.lower()

        print(material_name_image, material_name)
        material_intensities = mi_rasmi.get(material_name)

        # loop through all IMAGE classes and the according rasmi regions
        for image_region, rasmi_region in image_to_rasmi.items():
            # convert str to list is necessary:
            if isinstance(rasmi_region, str):
                rasmi_region = [rasmi_region]

            # loop through IMAGE regions and get the mean concrete mi value for each region for RS and RM (housing types)
            for housingtype_image, housingtype_rasmi in housing_type_image_to_rasmi.items():

                if image_region in steel_regions_adapt and material_name == "steel":
                    data_value = "p_5"
                elif image_region in [20] and material_name == "concrete":  # China concrete adaptation to highest value
                    data_value = "p_100"
                    print("done")
                else:
                    data_value = "p_50"

                filtered_mis = material_intensities[material_intensities.index.get_level_values('R5_32').isin(rasmi_region)  # filter for the right region
                                        & material_intensities.index.get_level_values('function').isin([housingtype_rasmi])  # filter for the right housing type of rasmi
                                        & material_intensities.index.get_level_values('structure').isin(housing_type_to_rasmi_building_structure[housingtype_image])].loc[:, data_value]  # filter for the right building structure

                structure_shares = rs_structure_shares if housingtype_rasmi == "RS" else rm_structure_shares
                mean_mi_value = weighted_structure_mi(filtered_mis, structure_shares, image_region,
                                                       housing_type_to_rasmi_building_structure[housingtype_image])
                # replace old values with new values
                mi_image_mat_update.loc[([start_year, target_year], image_region, housingtype_image), material_name_image] = mean_mi_value
                # save as csv
    mi_image_mat_update.to_csv("Building_materials_rasmi.csv")
    print("done")
    return mi_image_mat_update


def replace_old_mis_with_rasmi_resource_efficient(mi_image_mat: pd.DataFrame,
                                                    mi_rasmi: pd.DataFrame,
                                                    material_list_rasmi: list,
                                                    rs_structure_shares: pd.DataFrame,
                                                    rm_structure_shares: pd.DataFrame,
                                                    start_year: int = 2020,
                                                    switch_year: int = 2030,
                                                    target_year: int = 2050,
                                                    data_value="p_50"):
    """
    Replace old material intensities in IMAGE-Materials with RASMI values for concrete.

    material_name: str, non capitalised from rasmi
    mis_list_target_name: str, name of the material intensity in IMAGE-Materials
    data_value: str, can also be 0, 25, 75 or 100 percentile
    rs_structure_shares / rm_structure_shares: per-region structure type shares from structure_type_shares(),
        used to weight the MI values instead of taking a plain mean across structures.
    """
    # standardize column names at the top of the function
    mi_image_mat.columns = [c.lower().replace("aluminum", "aluminium") for c in mi_image_mat.columns]

    for material_name in material_list_rasmi:
        # ensure lower case
        if material_name.lower() == "aluminum":
            material_name_image = "aluminium"
        else:
            material_name_image = material_name.lower()

        print(material_name_image, material_name)
        material_intensities = mi_rasmi.get(material_name)

        # loop through all years
        for year in [start_year, switch_year, target_year]:
            print(year)
            if year == start_year:
                data_value = "p_50"
            if year == switch_year:
                data_value = "p_50"
            if year == target_year:
                data_value = "p_25"
            # loop through all IMAGE classes and the according rasmi regions
            for image_region, rasmi_region in image_to_rasmi.items():
                # convert str to list is necessary:
                if isinstance(rasmi_region, str):
                    rasmi_region = [rasmi_region]

                # loop through IMAGE regions and get the mean concrete mi value for each region for RS and RM (housing types)
                for housingtype_image, housingtype_rasmi in housing_type_image_to_rasmi.items():

                    if image_region in steel_regions_adapt and material_name == "steel":
                        data_value = "p_5"
                    else:
                        if year in [start_year] and image_region not in steel_regions_adapt and material_name != "steel":
                            print('switch back')
                            data_value = "p_50"
                        if year in [switch_year] and image_region not in steel_regions_adapt and material_name != "steel":
                            print('switch to more efficient')
                            data_value = "p_25"

                    filtered_mis = material_intensities[material_intensities.index.get_level_values('R5_32').isin(rasmi_region)  # filter for the right region
                                            & material_intensities.index.get_level_values('function').isin([housingtype_rasmi])  # filter for the right housing type of rasmi
                                            & material_intensities.index.get_level_values('structure').isin(housing_type_to_rasmi_building_structure[housingtype_image])].loc[:, data_value]  # filter for the right building structure

                    structure_shares = rs_structure_shares if housingtype_rasmi == "RS" else rm_structure_shares
                    mean_mi_value = weighted_structure_mi(filtered_mis, structure_shares, image_region,
                                                           housing_type_to_rasmi_building_structure[housingtype_image])
                    # replace old values with new values
                    mi_image_mat.loc[([year], image_region, housingtype_image), material_name_image] = mean_mi_value

                # save as csv
    mi_image_mat.to_csv("Building_materials_rasmi_resource_efficient.csv")
    print("done")
    return mi_image_mat
