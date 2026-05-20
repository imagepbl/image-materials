"""Material intensities for all infrastructure types (section 10 first half).

Builds the ``(Type, material, Region, Cohort)`` ``xr.DataArray`` of
kg/km² for roads, bridges, tunnels, parking, and rail, applying the
80/20 permanent-aggregate split (80% of aggregate is held as a permanent,
non-replaceable subgrade and exported separately for the reporting
step in ``infrastructure.infrastructure``).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import prism

from imagematerials.infrastructure.constants import FIRST_YEAR_GRID

# GRIP road class map for material intensity lookup (1=motorway ... 5=local)
MI_ROAD_CLASS_MAP = {
    "motorway": 1, "primary": 2, "secondary": 3, "tertiary": 4,
    "cycle": 5, "informal": 5, "local": 5, "pedestrian": 5,
}

# Pavingstone decomposition factors (from original v5 TRIPI 2024)
PAVINGSTONE_FACTORS = {"brick": 0.0381, "stone": 0.2705, "concrete": 0.6913}

# Base materials that contribute to final material intensities
BASE_MATERIALS = ["asphalt", "brick", "concrete", "metal", "stone", "wood"]
OUTPUT_MATERIALS = ["asphalt", "brick", "concrete", "steel", "stone", "wood", "aggregate"]

# Extended material list (all materials that appear across all element types)
EXTENDED_MATERIALS = ["aggregate", "asphalt", "concrete", "steel", "wood",
                       "stone", "brick", "copper", "zinc", "plastics",
                       "aluminium", "bronze"]

ROAD_ABBR_MAP = {
    "motorway": "mot", "primary": "pri", "secondary": "sec", "tertiary": "ter",
    "cycle": "cyc", "informal": "inf", "local": "loc", "pedestrian": "ped",
}
REGION_ABBR_MAP = {"urban": "urb", "rural": "rur"}
PAVE_ABBR_MAP = {"paved": "pav", "unpaved": "unp"}

# Parking uses same MI as local roads (v5: parking_mi = local_road_mi)
PARKING_LOCAL_ROAD_MAP = {
    "urban_parking_paved": "urban_local_paved",
    "urban_parking_unpaved": "urban_local_unpaved",
    "rural_parking_paved": "rural_local_paved",
    "rural_parking_unpaved": "rural_local_unpaved",
}

RAIL_MI_CATEGORY = {
    "total_rail": "standard_rail",
    "total_HST": "highspeed_rail",
    "total_urban_rail": "urban_rail",
    "rail_bridge": "rail_bridges",
    "rail_tunnel": "rail_tunnels",
}


def build_material_intensities(standard_data_dir: Path, scenario_data_dir: Path,
                                 final_all_types, all_types, bt_active_types,
                                 parking_types, rail_element_types,
                                 all_active_types, all_obsolete_types,
                                 obs_to_active_map, region_coords, end_year):
    """Return ``(material_intensities, permanent_aggregate_mi)`` DataArrays."""
    mat_int = pd.read_excel(scenario_data_dir / "material-intensity-roads.xlsx", index_col=0)
    materials_paving = pd.read_excel(standard_data_dir / "Materials-paving.xlsx", index_col=0)

    # --- Material intensities via Materials-paving.xlsx ---
    # Parse material shares: group by Road_type, Region_type, Paving_type
    paving_shares_dict = {}
    for road_type, road_group in materials_paving.groupby("Road_type"):
        for region_type, region_group in road_group.groupby("Region_type"):
            for paving_type, paving_group in region_group.groupby("Paving_type"):
                # Extract the 3-letter abbreviation as used in original v5
                var_key = f"{road_type[:3]}_{region_type[:3]}_{paving_type[:3]}"
                pg = paving_group.iloc[:, 3:]  # Drop Road_type, Region_type, Paving_type cols
                pg.index = pg.index.str.split("_").str[-1]  # Extract material name
                paving_shares_dict[var_key] = pg

    # Compute material intensity per road type (kg/km2) per region
    mi_per_type = {}  # type_name -> dict(material -> Series(region_values))

    for t in all_types:
        parts = t.split("_")
        loc_type = parts[0]     # 'urban' or 'rural'
        road_type = parts[1]    # 'motorway', 'primary', etc.
        pave_type = parts[2]    # 'paved' or 'unpaved'
        grip_rc = MI_ROAD_CLASS_MAP.get(road_type, 5)

        road_abbr = ROAD_ABBR_MAP.get(road_type, road_type[:3])
        region_abbr = REGION_ABBR_MAP.get(loc_type, loc_type[:3])
        pave_abbr = PAVE_ABBR_MAP.get(pave_type, pave_type[:3])

        paving_key = f"{road_abbr}_{region_abbr}_{pave_abbr}"

        # Initialize material dict per region
        mat_result = {m: pd.Series(0.0, index=region_coords) for m in OUTPUT_MATERIALS}

        # Get material intensity rows from mat_int for this road class
        if grip_rc in mat_int.index:
            rc_mats = mat_int.loc[grip_rc]
            if isinstance(rc_mats, pd.Series):
                rc_mats = rc_mats.to_frame().T

            # Get mat_int values per material (per region)
            mat_int_by_material = {}
            for _, row in rc_mats.iterrows():
                mat_name = row["material"]
                region_values = row.drop("material")
                # Align region names
                aligned = pd.Series(0.0, index=region_coords)
                for reg in region_coords:
                    if reg in region_values.index:
                        aligned[reg] = float(region_values[reg])
                mat_int_by_material[mat_name] = aligned

            # Look up paving shares if available
            if paving_key in paving_shares_dict:
                paving_shares_df = paving_shares_dict[paving_key]

                # Process base materials
                for mat_name in BASE_MATERIALS:
                    if (mat_name in paving_shares_df.index
                            and mat_name in mat_int_by_material):
                        share_row = paving_shares_df.loc[mat_name]
                        aligned_share = pd.Series(0.0, index=region_coords)
                        for reg in region_coords:
                            if reg in share_row.index:
                                aligned_share[reg] = float(share_row[reg])
                        # Material intensity = share × mat_int per region
                        out_mat = mat_name if mat_name != "metal" else "steel"
                        mat_result[out_mat] += (aligned_share
                                                * mat_int_by_material.get(mat_name, 0.0))

                # Process pavingstone → decompose into brick, stone, concrete
                if "pavingstone" in paving_shares_df.index:
                    ps_share_row = paving_shares_df.loc["pavingstone"]
                    aligned_ps_share = pd.Series(0.0, index=region_coords)
                    for reg in region_coords:
                        if reg in ps_share_row.index:
                            aligned_ps_share[reg] = float(ps_share_row[reg])
                    for target_mat, factor in PAVINGSTONE_FACTORS.items():
                        ps_intensity = aligned_ps_share * factor
                        if target_mat in mat_int_by_material:
                            mat_result[target_mat] += (ps_intensity
                                                       * mat_int_by_material[target_mat])

                # Process aggregate materials
                for agg_mat in ["asphalt", "brick", "concrete", "pavingstone", "stone"]:
                    if (agg_mat in paving_shares_df.index
                            and "aggregate" in mat_int_by_material):
                        share_row = paving_shares_df.loc[agg_mat]
                        aligned_share = pd.Series(0.0, index=region_coords)
                        for reg in region_coords:
                            if reg in share_row.index:
                                aligned_share[reg] = float(share_row[reg])
                        mat_result["aggregate"] += (aligned_share
                                                    * mat_int_by_material["aggregate"])

                # Process unpaved construction material → aggregate
                if ("unpaved - construction" in paving_shares_df.index
                        and "aggregate" in mat_int_by_material):
                    unp_share_row = paving_shares_df.loc["unpaved - construction"]
                    aligned_unp_share = pd.Series(0.0, index=region_coords)
                    for reg in region_coords:
                        if reg in unp_share_row.index:
                            aligned_unp_share[reg] = float(unp_share_row[reg])
                    mat_result["aggregate"] += (aligned_unp_share
                                                * mat_int_by_material["aggregate"])

            else:
                # No paving shares available for this type (e.g. unpaved motorways
                # don't exist in Materials-paving.xlsx) — leave MI at zero.
                pass

        # Scale: original v5 applies x10^6 conversion factor to combined_mi
        for m in OUTPUT_MATERIALS:
            mat_result[m] = mat_result[m] * 1000000

        # V5 line 2096-2099: 80% of aggregate is permanent (non-replaceable subgrade)
        # Store full aggregate MI for permanent tracking before applying 0.2 factor
        mi_per_type_full_agg = mat_result["aggregate"].copy()  # full 100% aggregate MI
        mat_result["aggregate"] = mat_result["aggregate"] * 0.2  # only 20% through DSM

        mi_per_type[t] = mat_result
        mi_per_type[f"_permanent_agg_{t}"] = mi_per_type_full_agg * 0.8  # 80% permanent

    # --- Load additional MI data for bridges, tunnels, parking, rail ---
    mi_bt = pd.read_excel(scenario_data_dir / "Materials_infra_bt.xlsx", index_col=[0, 1, 2])
    mi_infra = pd.read_excel(scenario_data_dir / "Materials_infra.xlsx", index_col=[0, 1])

    # --- Build MI for bridge/tunnel types (from Materials_infra_bt.xlsx) ---
    # Bridge/tunnel MI: region-specific, per road subtype
    bt_mi_map = {}  # type_name -> {material: value_per_region}
    for bt_type in bt_active_types:
        # Parse type: e.g. "rural_bridge_motorway" -> (rural, bridge, motorway)
        parts = bt_type.split("_")
        loc = parts[0]       # rural/urban
        elem = parts[1]      # bridge/tunnel
        road = parts[2]      # motorway/primary/...

        bt_cat = f"road_{elem}s_{road}"  # e.g. "road_bridges_motorway"
        mat_vals = {m: pd.Series(0.0, index=region_coords) for m in EXTENDED_MATERIALS}

        # Iterate regions in outer loop, try years in inner loop (use first available year per region)
        for reg in region_coords:
            for yr in [2000, 2050]:
                try:
                    row = mi_bt.loc[(yr, reg, bt_cat)]
                    for m in EXTENDED_MATERIALS:
                        if m in row.index:
                            mat_vals[m][reg] = float(row[m]) if not pd.isna(row[m]) else 0.0
                    break  # Use first available year for this region
                except KeyError:
                    continue
        bt_mi_map[bt_type] = mat_vals

    # --- Build MI for parking types (same as local roads, v5 approach) ---
    parking_mi_map = {}
    for ptype in parking_types:
        local_road_type = PARKING_LOCAL_ROAD_MAP[ptype]
        if local_road_type in mi_per_type:
            # Copy the local road MI (output_materials → extended_materials)
            # Note: road MI already has 20% aggregate and 80% stored separately
            mat_vals = {m: pd.Series(0.0, index=region_coords) for m in EXTENDED_MATERIALS}
            for m in OUTPUT_MATERIALS:
                if m in mi_per_type[local_road_type]:
                    mat_vals[m] = mi_per_type[local_road_type][m].copy()
            parking_mi_map[ptype] = mat_vals
            # Also copy the permanent aggregate from the local road type
            perm_key_road = f"_permanent_agg_{local_road_type}"
            perm_key_parking = f"_permanent_agg_{ptype}"
            if perm_key_road in mi_per_type:
                mi_per_type[perm_key_parking] = mi_per_type[perm_key_road].copy()
        else:
            parking_mi_map[ptype] = {m: pd.Series(0.0, index=region_coords)
                                     for m in EXTENDED_MATERIALS}

    # --- Build MI for rail types (from Materials_infra.xlsx) ---
    # Rail MI is NOT region-specific in the data file (same for all regions)
    rail_mi_map = {}
    for rtype in rail_element_types:
        cat = RAIL_MI_CATEGORY.get(rtype, "standard_rail")
        mat_vals = {m: pd.Series(0.0, index=region_coords) for m in EXTENDED_MATERIALS}
        for yr in [2000, 2050]:
            try:
                row = mi_infra.loc[(yr, cat)]
                for m in EXTENDED_MATERIALS:
                    col_name = m if m != "brick" else "bricks"
                    if col_name in row.index:
                        val = float(row[col_name]) if not pd.isna(row[col_name]) else 0.0
                        mat_vals[m] = pd.Series(val, index=region_coords)
                break
            except KeyError:
                continue
        rail_mi_map[rtype] = mat_vals

    # --- Build combined MI DataArray for ALL types ---
    # Use extended_materials for all types (roads get zeros for copper, zinc, etc.)
    mi_data = np.zeros((len(final_all_types), len(EXTENDED_MATERIALS), len(region_coords)))

    for i, t in enumerate(final_all_types):
        # Determine which MI source to use
        # Strip obsolete_ prefix to find active counterpart
        active_t = t
        if t.startswith("obsolete_"):
            active_t = t.replace("obsolete_", "", 1)

        if active_t in mi_per_type:
            # Road type - use existing road MI (map output_materials to extended_materials)
            for j, m in enumerate(EXTENDED_MATERIALS):
                if m in mi_per_type[active_t]:
                    mi_data[i, j, :] = mi_per_type[active_t][m].values
        elif active_t in bt_mi_map:
            # Bridge/tunnel type
            for j, m in enumerate(EXTENDED_MATERIALS):
                if m in bt_mi_map[active_t]:
                    mi_data[i, j, :] = bt_mi_map[active_t][m].values
        elif active_t in parking_mi_map:
            # Parking type
            for j, m in enumerate(EXTENDED_MATERIALS):
                if m in parking_mi_map[active_t]:
                    mi_data[i, j, :] = parking_mi_map[active_t][m].values
        elif active_t in rail_mi_map:
            # Rail type
            for j, m in enumerate(EXTENDED_MATERIALS):
                if m in rail_mi_map[active_t]:
                    mi_data[i, j, :] = rail_mi_map[active_t][m].values

    da_mi = xr.DataArray(
        mi_data,
        coords=[final_all_types, EXTENDED_MATERIALS, region_coords],
        dims=["Type", "material", "Region"],
    )
    # Cohort range must match the stock time range (from FIRST_YEAR_GRID)
    years = list(range(FIRST_YEAR_GRID, end_year + 1))
    da_mi = da_mi.expand_dims({"Cohort": years})

    # --- Permanent Aggregate MI (80% non-replaceable subgrade) ---
    # v5 lines 2096-2099, 3244, 3352: 80% of aggregate is permanent
    # Apply 80/20 split on aggregate MI for ALL types (active + obsolete)
    agg_idx = EXTENDED_MATERIALS.index("aggregate")
    perm_agg_data = np.zeros((len(all_active_types), len(region_coords)))

    for i, t in enumerate(all_active_types):
        type_idx = final_all_types.index(t)
        # For road types, mi_per_type already stored 20% aggregate; read permanent value
        perm_key = f"_permanent_agg_{t}"
        if perm_key in mi_per_type:
            perm_agg_data[i, :] = mi_per_type[perm_key].values
        else:
            # Non-road active types: split the full aggregate MI
            full_agg_mi = mi_data[type_idx, agg_idx, :].copy()
            perm_agg_data[i, :] = full_agg_mi * 0.8
            mi_data[type_idx, agg_idx, :] = full_agg_mi * 0.2

    # Also reduce aggregate MI for obsolete types to match their active counterparts (20%)
    for obs_t in all_obsolete_types:
        if obs_t in final_all_types:
            obs_idx = final_all_types.index(obs_t)
            active_t = obs_to_active_map.get(obs_t, obs_t)
            if active_t in mi_per_type:
                # Road obsolete: already has 20% aggregate from mi_per_type
                pass
            else:
                # Non-road obsolete: reduce to 20%
                full_agg = mi_data[obs_idx, agg_idx, :].copy()
                mi_data[obs_idx, agg_idx, :] = full_agg * 0.2

    # Rebuild da_mi with 20% aggregate through DSM
    da_mi = xr.DataArray(
        mi_data,
        coords=[final_all_types, EXTENDED_MATERIALS, region_coords],
        dims=["Type", "material", "Region"],
    )
    da_mi = da_mi.expand_dims({"Cohort": years})
    da_mi = prism.Q_(da_mi, "kg / km**2")

    # Permanent aggregate MI DataArray for active types (used in infrastructure.py)
    da_perm_agg_mi = xr.DataArray(
        perm_agg_data,
        coords=[all_active_types, region_coords],
        dims=["Type", "Region"],
    )

    return da_mi, da_perm_agg_mi
