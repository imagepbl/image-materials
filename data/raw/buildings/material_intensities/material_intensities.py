from pathlib import Path

import pandas as pd
import numpy as np

# The preprocessing reads the MI tables from data/raw/buildings/<scenario>/, so the
# generated CSVs are written straight into every scenario subfolder.
SCENARIO_DIRS = [
    Path(__file__).resolve().parent.parent / "SSP2_CP",
    Path(__file__).resolve().parent.parent / "SSP2_2D_RE",
]


def _write_to_scenario_dirs(df: pd.DataFrame, filename: str):
    """Write ``df`` as ``filename`` into every scenario subfolder under data/raw/buildings/."""
    for scenario_dir in SCENARIO_DIRS:
        out_path = scenario_dir / filename
        df.to_csv(out_path)
        print(f"wrote {out_path}")


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

# IMAGE-Materials commercial building types. RASMI resolves only a single
# non-residential function ("NR"), so every commercial type is assigned the same
# regionalised NR material intensity, weighted over all four structure types.
commercial_types_image = ["Offices", "Retail+", "Hotels+", "Govt+"]
commercial_type_to_rasmi_building_structure = {
    building_type: ['C', 'M', 'S', 'T'] for building_type in commercial_types_image
}

# RASMI sheet names (aluminum) and their IMAGE-Materials column names (aluminium).
material_list_rasmi = ["steel", "concrete", "wood", "copper", "aluminum", "glass", "brick", "plastics"]
mis_list_target = ["steel", "concrete", "wood", "copper", "aluminium", "glass", "brick", "plastics"]

# RASMI has almost no empirical building records for aluminium (93 datapoints across 384
# region/structure/function cells), so its imputation collapses p_50 to a near-constant
# global prior (~0.49 kg/m2) with no regional or typological signal, driving building
# aluminium stocks well below the material-flow literature. Aluminium is therefore taken
# from the older Deetman et al. dataset instead.
MATERIALS_FROM_DEETMAN = ["aluminium"]

# Deetman aluminium intensities (residential and commercial) are themselves on the high
# side - the raw Deetman values put the aluminium building stock well above independent
# estimates. A single global calibration factor is applied to every Deetman aluminium
# value to reconcile the stock with two benchmarks: USA buildings 2009 ~155 kg/cap and
# Europe (WEU+CEU) 2013 ~49 Mt. 0.64 lands both within a few percent
# (USA ~159 kg/cap, Europe ~49 Mt).
DEETMAN_ALUMINIUM_CALIBRATION = 0.64

# Copper is equally data-poor in RASMI (~30 datapoints), so its p_50 is also a flat
# ~0.18 kg/m2 prior. Deetman copper is not a good alternative either (detached ~1.7-2.4
# and commercial ~3.5 overshoot the literature, and the Deetman semi-detached / high-rise
# rows are 0.01 placeholders). We instead take RASMI's p_75 for copper: still on the RASMI
# methodology, but away from the degenerate median. p_75 ~ 0.27 kg/m2.
COPPER_RASMI_PERCENTILE = "p_75"

# Deetman residential type 2 (semi-detached) aluminium is an unpopulated placeholder -
# a single value (0.23) flat across all 26 regions, unlike the regionally differentiated
# type 1/3/4 values - so it is taken from the nearest real building type, detached (type 1).
# Type 4 (high-rise) has genuine regionalised Deetman aluminium and is left as-is.
DEETMAN_RESIDENTIAL_TYPE_FALLBACK = {2: 1}


def load_mi_rasmi():
    """Read in RASMI material intensity data."""
    return pd.read_excel(
        "MI_ranges_20230905.xlsx", index_col=[0, 1, 2, 3, 4],
        sheet_name=["concrete", "brick", "wood", "steel", "glass", "plastics", "aluminum", "copper"]
    )


def load_mi_deetman_residential():
    """Read the Deetman et al. residential MI table (kg/m2), indexed by (Year, Region, Building_type).

    Column names are title-case in the file (Aluminium, Copper, ...); they are lower-cased
    here to match the IMAGE-Materials material naming. The aluminium column is scaled by
    DEETMAN_ALUMINIUM_CALIBRATION.
    """
    df = pd.read_csv("Building_materials_deetman.csv", index_col=[0, 1, 2])
    df.columns = [c.lower() for c in df.columns]
    df.index.names = ["Year", "Region", "Building_type"]
    df["aluminium"] *= DEETMAN_ALUMINIUM_CALIBRATION
    return df


def load_mi_deetman_commercial():
    """Read the Deetman et al. commercial MI table (kg/m2), indexed by (Year, Material),
    one column per commercial building type (Offices / Retail+ / Hotels+ / Govt+).

    The aluminium row is scaled by DEETMAN_ALUMINIUM_CALIBRATION.
    """
    df = pd.read_csv("materials_commercial_old.csv", index_col=[0, 1])
    df.index.names = ["Year", "Material"]
    df.loc[(slice(None), "aluminium"), :] *= DEETMAN_ALUMINIUM_CALIBRATION
    return df


def _fill_residential_deetman_al_cu(mi_image_mat: pd.DataFrame, years: list) -> pd.DataFrame:
    """Overwrite the aluminium column of a residential MI table with Deetman values.

    ``mi_image_mat`` is indexed by (Year, Region, Building_type). Deetman only provides
    2020 and 2050 (identical, no trend), so the 2020 value is used for every year in
    ``years``. Placeholder type 2 / type 4 rows fall back to types 1 / 3
    (see DEETMAN_RESIDENTIAL_TYPE_FALLBACK). Copper is not touched here - it stays on
    RASMI (see COPPER_RASMI_PERCENTILE).
    """
    deetman = load_mi_deetman_residential()
    deetman_2020 = deetman.xs(2020, level="Year")  # (Region, Building_type) x material

    for material in MATERIALS_FROM_DEETMAN:
        for region in image_regions:
            for building_type in image_housing_types:
                src_type = DEETMAN_RESIDENTIAL_TYPE_FALLBACK.get(building_type, building_type)
                value = deetman_2020.loc[(region, src_type), material]
                for year in years:
                    mi_image_mat.loc[(year, region, building_type), material] = value
    return mi_image_mat


def _fill_commercial_deetman_al_cu(mi_image_mat_commercial: pd.DataFrame, years: list) -> pd.DataFrame:
    """Overwrite the aluminium rows of a commercial MI table with Deetman values.

    ``mi_image_mat_commercial`` is indexed by (Year, Region, Material) with one column per
    commercial building type. The Deetman commercial MI is not regionalised, so the same
    per-type value is written to every IMAGE region. Copper is not touched here - it stays
    on RASMI (see COPPER_RASMI_PERCENTILE).
    """
    deetman = load_mi_deetman_commercial()
    deetman_2020 = deetman.xs(2020, level="Year")  # Material x building type

    for material in MATERIALS_FROM_DEETMAN:
        row = deetman_2020.loc[material]  # one value per commercial building type
        for year in years:
            for region in image_regions:
                mi_image_mat_commercial.loc[(year, region, material), :] = row.values
    return mi_image_mat_commercial


# IMAGE R26 regions and the 4 residential building types the MI table is defined over.
image_regions = list(image_to_rasmi.keys())
image_housing_types = list(housing_type_image_to_rasmi.keys())


def build_mi_image_mat(years=(2020, 2050), materials=mis_list_target):
    """Build an empty IMAGE-Materials residential MI table (all NaN).

    Indexed by (Year, Region, Building_type) over the given ``years``, the 26 IMAGE
    regions and the 4 residential building types, with one column per material. The
    values are filled in from RASMI by ``replace_old_mis_with_rasmi`` / its resource
    efficient variant.
    """
    index = pd.MultiIndex.from_product(
        [list(years), image_regions, image_housing_types],
        names=["Year", "Region", "Building_type"],
    )
    return pd.DataFrame(np.nan, index=index, columns=list(materials))


def build_mi_image_mat_commercial(years=(2020, 2050), materials=mis_list_target):
    """Build an empty IMAGE-Materials commercial MI table (all NaN).

    Indexed by (Year, Region, Material) over the given ``years``, the 26 IMAGE regions
    and the RASMI materials, with one column per commercial building type. The values
    are filled in from RASMI by ``replace_commercial_mis_with_rasmi``.
    """
    index = pd.MultiIndex.from_product(
        [list(years), image_regions, list(materials)],
        names=["Year", "Region", "Material"],
    )
    return pd.DataFrame(np.nan, index=index, columns=list(commercial_types_image))


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

    housing_prefix: "RS" (residential single-family), "RM" (residential multi-family) or
        "NR" (non-residential / commercial)
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
    Fill the (empty) IMAGE-Materials MI table from build_mi_image_mat() with RASMI values.

    material_name: str, non capitalised from rasmi
    mis_list_target_name: str, material intensity in IMAGE-Materials
    data_value: str, can also be 0, 25, 75 or 100 percentile
    rs_structure_shares / rm_structure_shares: per-region structure type shares from structure_type_shares(),
        used to weight the MI values instead of taking a plain mean across structures.

    Aluminium is not taken from RASMI (its data is too sparse; see MATERIALS_FROM_DEETMAN)
    but filled from the Deetman et al. table afterwards. Copper stays on RASMI but uses the
    p_75 percentile (see COPPER_RASMI_PERCENTILE) because its p_50 is a degenerate prior.
    """
    mi_image_mat_update = mi_image_mat.copy()

    for material_name in material_list_rasmi:
        # ensure lower case
        if material_name.lower() == "aluminum":
            material_name_image = "aluminium"
        else:
            material_name_image = material_name.lower()

        if material_name_image in MATERIALS_FROM_DEETMAN:
            continue  # filled from Deetman below

        # copper's RASMI p_50 is a flat ~0.18 prior (almost no data); use p_75 instead
        material_data_value = COPPER_RASMI_PERCENTILE if material_name_image == "copper" else "p_50"

        print(material_name_image, material_name)
        material_intensities = mi_rasmi.get(material_name)

        # loop through all IMAGE classes and the according rasmi regions
        for image_region, rasmi_region in image_to_rasmi.items():
            # convert str to list is necessary:
            if isinstance(rasmi_region, str):
                rasmi_region = [rasmi_region]

            # loop through IMAGE regions and get the mean concrete mi value for each region for RS and RM (housing types)
            for housingtype_image, housingtype_rasmi in housing_type_image_to_rasmi.items():

                filtered_mis = material_intensities[material_intensities.index.get_level_values('R5_32').isin(rasmi_region)  # filter for the right region
                                        & material_intensities.index.get_level_values('function').isin([housingtype_rasmi])  # filter for the right housing type of rasmi
                                        & material_intensities.index.get_level_values('structure').isin(housing_type_to_rasmi_building_structure[housingtype_image])].loc[:, material_data_value]  # filter for the right building structure

                structure_shares = rs_structure_shares if housingtype_rasmi == "RS" else rm_structure_shares
                mean_mi_value = weighted_structure_mi(filtered_mis, structure_shares, image_region,
                                                       housing_type_to_rasmi_building_structure[housingtype_image])
                # replace old values with new values
                mi_image_mat_update.loc[([start_year, target_year], image_region, housingtype_image), material_name_image] = mean_mi_value
                # save as csv

    # aluminium from Deetman (RASMI too sparse)
    mi_image_mat_update = _fill_residential_deetman_al_cu(
        mi_image_mat_update, [start_year, target_year])

    _write_to_scenario_dirs(mi_image_mat_update, "Building_materials_rasmi.csv")
    print("done")
    return mi_image_mat_update


def replace_commercial_mis_with_rasmi(mi_image_mat_commercial: pd.DataFrame, mi_rasmi: pd.DataFrame,
                                       material_list_rasmi: list, nr_structure_shares: pd.DataFrame,
                                       start_year: int = 2020, target_year: int = 2050,
                                       data_value: str = "p_50"):
    """
    Fill the (empty) IMAGE-Materials commercial MI table from build_mi_image_mat_commercial()
    with regionalised RASMI values for the non-residential ("NR") function.

    RASMI resolves only a single non-residential function, so the same regionalised MI is
    written to every commercial building type. The value is weighted over the C/M/S/T
    structure types by their per-region non-residential GFA share from MaterialCities
    (nr_structure_shares), mirroring the residential approach in replace_old_mis_with_rasmi.

    Aluminium is not taken from RASMI (its data is too sparse; see MATERIALS_FROM_DEETMAN)
    but filled from the Deetman et al. commercial table afterwards. Copper stays on RASMI
    but uses the p_75 percentile (see COPPER_RASMI_PERCENTILE).
    """
    mi_image_mat_commercial_update = mi_image_mat_commercial.copy()
    allowed_structures = ['C', 'M', 'S', 'T']

    for material_name in material_list_rasmi:
        material_name_image = "aluminium" if material_name.lower() == "aluminum" else material_name.lower()

        if material_name_image in MATERIALS_FROM_DEETMAN:
            continue  # filled from Deetman below

        # copper's RASMI p_50 is a flat ~0.18 prior (almost no data); use p_75 instead
        material_data_value = COPPER_RASMI_PERCENTILE if material_name_image == "copper" else data_value

        print(material_name_image, material_name)
        material_intensities = mi_rasmi.get(material_name)

        # loop through all IMAGE regions and the according RASMI regions
        for image_region, rasmi_region in image_to_rasmi.items():
            if isinstance(rasmi_region, str):
                rasmi_region = [rasmi_region]

            filtered_mis = material_intensities[
                material_intensities.index.get_level_values('R5_32').isin(rasmi_region)  # filter for the right region
                & material_intensities.index.get_level_values('function').isin(['NR'])  # non-residential function
                & material_intensities.index.get_level_values('structure').isin(allowed_structures)  # all structure types
            ].loc[:, material_data_value]

            mean_mi_value = weighted_structure_mi(filtered_mis, nr_structure_shares, image_region,
                                                  allowed_structures)
            # same regionalised NR value for every commercial building type
            mi_image_mat_commercial_update.loc[
                ([start_year, target_year], image_region, material_name_image), :] = mean_mi_value

    # aluminium from Deetman (RASMI too sparse)
    mi_image_mat_commercial_update = _fill_commercial_deetman_al_cu(
        mi_image_mat_commercial_update, [start_year, target_year])

    # Written under a distinct name so it sits alongside (does not overwrite) the
    # legacy non-regionalised materials_commercial_rasmi.csv. The buildings
    # preprocessing switches between the two via USE_REGIONALIZED_COMMERCIAL_MI in
    # imagematerials/buildings/preprocessing/materials.py.
    _write_to_scenario_dirs(mi_image_mat_commercial_update,
                            "materials_commercial_rasmi_regionalized.csv")
    print("done")
    return mi_image_mat_commercial_update


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
    Fill the (empty) IMAGE-Materials MI table from build_mi_image_mat() with resource
    efficient RASMI values (lower percentiles in later years).

    material_name: str, non capitalised from rasmi
    mis_list_target_name: str, name of the material intensity in IMAGE-Materials
    data_value: str, can also be 0, 25, 75 or 100 percentile
    rs_structure_shares / rm_structure_shares: per-region structure type shares from structure_type_shares(),
        used to weight the MI values instead of taking a plain mean across structures.

    Aluminium is not taken from RASMI (its data is too sparse; see MATERIALS_FROM_DEETMAN)
    but filled from the Deetman et al. table afterwards; Deetman has no resource-efficiency
    trajectory, so the same value is used in every year. Copper stays on RASMI but at the
    p_75 percentile in every year (see COPPER_RASMI_PERCENTILE) - its p_50 / p_25 are
    degenerate priors, so no resource-efficiency reduction is applied to copper.
    """
    for material_name in material_list_rasmi:
        # ensure lower case
        if material_name.lower() == "aluminum":
            material_name_image = "aluminium"
        else:
            material_name_image = material_name.lower()

        if material_name_image in MATERIALS_FROM_DEETMAN:
            continue  # filled from Deetman below

        print(material_name_image, material_name)
        material_intensities = mi_rasmi.get(material_name)

        # loop through all years; p_50 until the switch year, p_25 from the target year on,
        # applied uniformly to every material and region
        for year in [start_year, switch_year, target_year]:
            print(year)
            if material_name_image == "copper":
                data_value = COPPER_RASMI_PERCENTILE  # no RE reduction for copper (see docstring)
            elif year == target_year:
                data_value = "p_25"
            else:
                data_value = "p_50"
            # loop through all IMAGE classes and the according rasmi regions
            for image_region, rasmi_region in image_to_rasmi.items():
                # convert str to list is necessary:
                if isinstance(rasmi_region, str):
                    rasmi_region = [rasmi_region]

                # loop through IMAGE regions and get the mean concrete mi value for each region for RS and RM (housing types)
                for housingtype_image, housingtype_rasmi in housing_type_image_to_rasmi.items():

                    filtered_mis = material_intensities[material_intensities.index.get_level_values('R5_32').isin(rasmi_region)  # filter for the right region
                                            & material_intensities.index.get_level_values('function').isin([housingtype_rasmi])  # filter for the right housing type of rasmi
                                            & material_intensities.index.get_level_values('structure').isin(housing_type_to_rasmi_building_structure[housingtype_image])].loc[:, data_value]  # filter for the right building structure

                    structure_shares = rs_structure_shares if housingtype_rasmi == "RS" else rm_structure_shares
                    mean_mi_value = weighted_structure_mi(filtered_mis, structure_shares, image_region,
                                                           housing_type_to_rasmi_building_structure[housingtype_image])
                    # replace old values with new values
                    mi_image_mat.loc[([year], image_region, housingtype_image), material_name_image] = mean_mi_value

                # save as csv

    # aluminium from Deetman (RASMI too sparse)
    mi_image_mat = _fill_residential_deetman_al_cu(
        mi_image_mat, [start_year, switch_year, target_year])

    _write_to_scenario_dirs(mi_image_mat, "Building_materials_rasmi_resource_efficient.csv")
    print("done")
    return mi_image_mat
