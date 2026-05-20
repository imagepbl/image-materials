"""Build Weibull and folded-normal lifetime parameter arrays (section 10 second half).

Roads (active + obsolete), bridges, tunnels, and parking use Weibull
distributions with shape/scale parameters from ``infra_lifetime_weibull.xlsx``.
Rail elements use folded-normal distributions whose mean transitions
linearly from ``start_life`` to ``target_life`` between 2025 and 2050,
configured by ``operational_lifetime.csv``.

Both arrays are returned in the shape expected by
``imagematerials.survival.ScipySurvival``: ``(ScipyParam, Time, Type)``.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from imagematerials.infrastructure.constants import LIFETIME_RC_MAP

# Bridge class map (v5)
BRIDGE_BC_MAP = {
    "urban_bridge": {"cycle": 1, "informal": 1, "local": 3, "motorway": 6,
                      "pedestrian": 1, "primary": 5, "secondary": 4, "tertiary": 3},
    "rural_bridge": {"cycle": 1, "informal": 1, "local": 1, "motorway": 4,
                      "pedestrian": 1, "primary": 3, "secondary": 2, "tertiary": 2},
}
# Tunnels use same mapping as bridges
TUNNEL_BC_MAP = {
    "urban_tunnel": BRIDGE_BC_MAP["urban_bridge"],
    "rural_tunnel": BRIDGE_BC_MAP["rural_bridge"],
}

# Parking lifetime: urban=rc3, rural=rc2
PARKING_RC_MAP = {
    "urban_paved_parking": 3, "urban_unpaved_parking": 3,
    "rural_paved_parking": 2, "rural_unpaved_parking": 2,
}

# v5: stdev_mult=5, mean transitions from start_life to target_life between 2025-2050
STDEV_MULT = 5


def build_lifetimes(scenario_data_dir: Path, combined_types, bt_active_types, bt_obs_types,
                    parking_types, parking_obs_types, rail_element_types, rail_obs_types,
                    extended_time_coords):
    """Return a dict ``{"weibull": DataArray, "folded_norm": DataArray}``."""
    weibull = pd.read_excel(scenario_data_dir / "infra_lifetime_weibull.xlsx", index_col=0)
    operational_lt = pd.read_csv(scenario_data_dir / "operational_lifetime.csv", index_col=0)

    # --- Weibull lifetimes: roads, bridges/tunnels, parking ---
    rc10_shape = weibull.loc["Road Class 10", "shape"]
    rc10_scale = weibull.loc["Road Class 10", "scale"]
    bc9_shape = weibull.loc["Bridge Class 9", "shape"]
    bc9_scale = weibull.loc["Bridge Class 9", "scale"]

    weibull_types = []
    weibull_shapes = []
    weibull_scales = []

    # 1. Road types (active + obsolete)
    for t in combined_types:
        weibull_types.append(t)
        if t.startswith("obsolete_"):
            weibull_shapes.append(rc10_shape)
            weibull_scales.append(rc10_scale)
        else:
            parts = t.split("_")
            category = f"{parts[0]}_{parts[1]}"
            road_type = parts[2]
            rc = LIFETIME_RC_MAP.get(category, {}).get(road_type, 1)
            rc_label = f"Road Class {rc}"
            weibull_shapes.append(weibull.loc[rc_label, "shape"])
            weibull_scales.append(weibull.loc[rc_label, "scale"])

    # 2. Bridge/tunnel types (active + obsolete)
    for t in bt_active_types:
        weibull_types.append(t)
        parts = t.split("_")  # e.g. urban_bridge_motorway
        loc_elem = f"{parts[0]}_{parts[1]}"
        road_type = parts[2]
        bc_map = BRIDGE_BC_MAP if "bridge" in t else TUNNEL_BC_MAP
        bc = bc_map.get(loc_elem, {}).get(road_type, 1)
        bc_label = f"Bridge Class {bc}"
        if bc_label in weibull.index:
            weibull_shapes.append(weibull.loc[bc_label, "shape"])
            weibull_scales.append(weibull.loc[bc_label, "scale"])
        else:
            weibull_shapes.append(2.0)
            weibull_scales.append(100.0)

    for t in bt_obs_types:
        weibull_types.append(t)
        weibull_shapes.append(bc9_shape)
        weibull_scales.append(bc9_scale)

    # 3. Parking types (active + obsolete)
    for t in parking_types:
        weibull_types.append(t)
        rc = PARKING_RC_MAP.get(t, 2)
        rc_label = f"Road Class {rc}"
        weibull_shapes.append(weibull.loc[rc_label, "shape"])
        weibull_scales.append(weibull.loc[rc_label, "scale"])

    for t in parking_obs_types:
        weibull_types.append(t)
        weibull_shapes.append(rc10_shape)
        weibull_scales.append(rc10_scale)

    # Build Weibull lifetime DataArray
    c_array = np.array(weibull_shapes, dtype=float)
    scale_array = np.array(weibull_scales, dtype=float)
    param_array = np.stack([c_array, scale_array], axis=0)
    da_lt_weibull = xr.DataArray(
        param_array,
        dims=["ScipyParam", "Type"],
        coords={"ScipyParam": ["c", "scale"], "Type": weibull_types},
        attrs={"loc": 0},
    )
    # Expand Time dimension for ScipySurvival
    da_lt_weibull = da_lt_weibull.expand_dims(Time=extended_time_coords)

    # --- 4. Rail types use FoldedNormal distribution ---
    fn_types = rail_element_types + rail_obs_types

    fn_c_data = np.zeros((len(extended_time_coords), len(fn_types)))
    fn_scale_data = np.zeros((len(extended_time_coords), len(fn_types)))

    for j, t in enumerate(fn_types):
        active_t = t.replace("obsolete_", "", 1) if t.startswith("obsolete_") else t
        # Map to operational_lt index
        lt_key_map = {
            "total_rail": "rail", "total_HST": "rail",
            "total_urban_rail": "urban_rail",
            "rail_bridge": "rail_bridge", "rail_tunnel": "rail_tunnel",
        }
        lt_key = lt_key_map.get(active_t, "rail")
        if t.startswith("obsolete_"):
            lt_key = "obs_rail"

        if lt_key in operational_lt.index:
            start_life = float(operational_lt.loc[lt_key, "start"])
            target_life = float(operational_lt.loc[lt_key, "end"])
        else:
            start_life, target_life = 30.0, 30.0

        for i, year in enumerate(extended_time_coords):
            if year < 2025:
                mean = start_life
            elif year <= 2050:
                mean = start_life + (target_life - start_life) * ((year - 2025) / 25.0)
            else:
                mean = target_life
            stdev = mean * STDEV_MULT
            # FoldedNormal scipy params: c = mean/stdev, scale = stdev, loc = 0
            fn_c_data[i, j] = mean / stdev if stdev > 0 else 1.0
            fn_scale_data[i, j] = stdev

    fn_param = np.stack([fn_c_data, fn_scale_data], axis=-1)  # (Time, Type, 2)
    fn_param = np.transpose(fn_param, (2, 0, 1))  # (ScipyParam, Time, Type)
    da_lt_foldnorm = xr.DataArray(
        fn_param,
        dims=["ScipyParam", "Time", "Type"],
        coords={"ScipyParam": ["c", "scale"],
                "Time": extended_time_coords,
                "Type": fn_types},
        attrs={"loc": 0},
    )

    return {"weibull": da_lt_weibull, "folded_norm": da_lt_foldnorm}
