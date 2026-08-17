"""Functions for implementing circular economy measures."""
import logging

import numpy as np
import prism
import xarray as xr

from imagematerials.concepts import create_region_graph
from imagematerials.constants import IMAGE_REGIONS
from imagematerials.util import apply_change_per_region


def ce_measures_residential_housing(total_m2_housing_per_cap: xr.DataArray, population: xr.DataArray,
                                    circular_economy_config: dict):
    """Implement circular economy measures for residential housing.

    Parameters
    ----------
    total_m2_housing_per_cap:
        The total amount of m2 per capita for all residential housing.
    population:
        Population data per Area (rural/urban) and Total values
    circular_economy_config:
        Configuration of the circular economy.

    Returns
    -------
    total_m2_housing_per_cap:
        Updated values for the total amount of floorspace per capita.

    """
    region_knowledge_graph = create_region_graph()
    regions = total_m2_housing_per_cap.coords["Region"].values

    buildings_config = circular_economy_config["base"]["buildings"]
    base_year = buildings_config["base_year"]
    floor_pc_2020 = buildings_config["residential"]["2020"]["useful_floor_pc"]

    floor_pc_2020_xr = xr.DataArray(
        list(floor_pc_2020.values()),
        coords={"Region": list(floor_pc_2020.keys())},
        dims=["Region"],
        name="floor_pc_2020",
    )

    regions_mapped = list(region_knowledge_graph.find_relations_inverse(regions,
                                                                        floor_pc_2020.keys()))
    floor_pc_2020_mapped = region_knowledge_graph.rebroadcast_xarray(
    floor_pc_2020_xr, output_coords=regions_mapped, dim="Region")
    floor_pc_2020_mapped = prism.Q_(floor_pc_2020_mapped, "m^2/person")
    target_vals = floor_pc_2020_mapped

    population_rururb = population.sel(Area=["Rural", "Urban"])
    population_total = population.sel(Area="Total")

    # Population-weighted current per-capita total at base_year
    # We use the 2020 anchor to compute the scaling factor, which is then applied to all years.
    current_vals = (
        (total_m2_housing_per_cap * population_rururb).sel(Time=2020).sum(dim=["Area", "Type"]) 
        / population_total.sel(Time=2020)
    ).drop_vars(["Area"])

    scaling_factors  = target_vals / current_vals   # (Region,) dimensionless

    # --- Time-decaying correction for large scaling factors only ---
    # Factors at/below 'threshold' stay constant across all years.
    # Factors above 'threshold' ramp linearly from their base_year value down to
    # 'threshold' by 'decay_end_year' (we trust the 2020 anchor, but don't
    # let a scaling factor compound into the far future).
    # this was implemented because for China the scaling factor is very high in 2020, which would lead to ~70m2/cap results in 2100
    # This implementation allows for floorspace 'harmonisation' (base scenario) to material studies in 2020, while  avoiding unrealistic results in 2100

    threshold = 1.5
    decay_end_year = 2100

    # The floor_region is the value that the scaling factor will converge to by decay_end_year
    floor_region = xr.where(scaling_factors > threshold, threshold, scaling_factors)  # (Region,)

    # The weight ramps linearly from 0 at base_year to 1 at decay_end_year, and is clipped to [0,1].
    all_times = total_m2_housing_per_cap.Time.values
    weight = np.clip((all_times - base_year) / (decay_end_year - base_year), 0.0, 1.0)
    weight = xr.DataArray(weight, dims=["Time"], coords={"Time": all_times})

    # The factor_t is a linear interpolation between the scaling_factors and the floor_region, based on the weight.
    factor_t = scaling_factors * (1 - weight) + floor_region * weight        # (Region, Time)
    factor_t = xr.where(all_times <= base_year, scaling_factors, factor_t)  # full correction up to 2020

    # Multiplying the regional total by one factor scales every Type/Area identically,
    # preserving the within-region type mix.
    total_m2_housing_per_cap.loc[{"Region": regions_mapped}] = (
        total_m2_housing_per_cap.sel(Region=regions_mapped)
        * factor_t.sel(Region=regions_mapped)
    )
    
# narrow implementation for: 
#---> regions unchanged under 'base' --> 1) take per capita floorspace directly from IMAGE 
#---> regions changed under 'base' --> 1) take original reduction (SSP2 by 2060 vs narrow_act by 2060) in % terms and apply that to per capita 
#                                       floorspace after 'base'; trajectory from 2020 to 2060 follows the same trajectory in IMAGE
#                                     2) if region is >36 by 2060, implement reduction to 36 by 2100.
#                                        if region is <36 by 2060, assume growth to 36 by 2100         
                                       
    narrow_scen = None
    if 'narrow' in circular_economy_config.keys():
        narrow_scen = 'narrow'
    if 'narrow_activity' in circular_economy_config.keys():
        narrow_scen = 'narrow_activity'

    if narrow_scen is not None:
        nbuild = circular_economy_config[narrow_scen]["buildings"]
        narrow_base_year = nbuild["base_year"]
        target_year = nbuild["target_year"]          # 2060
        implementation_rate = nbuild["implementation_rate"]
        convergence_cap = 36.0                        # m²/cap (population-weighted total)
        convergence_year_end = 2100

        base_region_set = set(regions_mapped)

        # --- Phase 1: percentage reduction until target_year (2060) -----------
        # Only base regions get a reduction; build a full-coverage change array
        # (apply_change_per_region iterates over every region) with 0 elsewhere.
        residential_settings = nbuild["residential"]["m2_change_pc"]
        settings_xr = xr.DataArray(
            list(residential_settings.values()),
            coords={"Region": list(residential_settings.keys())},
            dims=["Region"],
            name="residential_settings",
        )
        settings_regions = list(region_knowledge_graph.find_relations_inverse(
            regions, residential_settings.keys()))
        settings_mapped = region_knowledge_graph.rebroadcast_xarray(
            settings_xr, output_coords=settings_regions, dim="Region")

        all_regions = list(total_m2_housing_per_cap.coords["Region"].values)
        narrow_regions = [r for r in all_regions if r in base_region_set]

        # --- Work in plain floats for the convergence arithmetic --------------
        if prism.U_(total_m2_housing_per_cap) is not None:
            total_m2_housing_per_cap = total_m2_housing_per_cap.pint.dequantify()
        pr = population_rururb.pint.dequantify() if prism.U_(population_rururb) is not None else population_rururb
        pt = population_total.pint.dequantify() if prism.U_(population_total) is not None else population_total

        all_times = total_m2_housing_per_cap.Time.values
        post_mask = all_times > target_year

        # Smootherstep (6x^5 - 15x^4 + 10x^3) ramp 0 -> 1 across target_year -> convergence_year_end to the cap
        # Chosen over linear/spline/sigmoid smoothing because it's the minimal,
        # parameter-free function that is monotonic and bounded in [0,1] (never
        # overshoots the cap) and continuous at both anchor years (no kinks in
        # the floorspace trajectory, which would otherwise show up as spikes in 
        # the stock -> flow series). 
        lin = np.clip((all_times - target_year) / (convergence_year_end - target_year), 0.0, 1.0)
        smooth = 6 * lin**5 - 15 * lin**4 + 10 * lin**3
        ramp = xr.DataArray(smooth, dims=["Time"], coords={"Time": all_times})

        def _region_pc(reg, t):
            """Population-weighted per-capita total (rural+urban) for region at time t."""
            fs = total_m2_housing_per_cap.sel(Time=t, Region=reg)        # (Area, Type)
            tot = (fs * pr.sel(Time=t, Region=reg)).sum(dim=["Area", "Type"])
            return float(tot / pt.sel(Time=t, Region=reg))

        # --- Phase 2: converge every reduced base region to 36 by 2100 --------
        # Take the 2060 (Area, Type) distribution, scale it so its
        # population-weighted total equals 36 -> that's the 2100 endpoint.
        for reg in narrow_regions:
            fs_2060 = total_m2_housing_per_cap.sel(Time=target_year, Region=reg)   # (Area, Type)
            pc_2060 = _region_pc(reg, target_year)
            if pc_2060 <= 0:
                continue
            fs_end = fs_2060 * (convergence_cap / pc_2060)   # scaled so weighted total = 36

            for t in all_times[post_mask]:
                r = float(ramp.sel(Time=t))
                total_m2_housing_per_cap.loc[{"Time": t, "Region": reg}] = (
                    fs_2060.values * (1 - r) + fs_end.values * r
                )

        # Re-attach units cleanly
        total_m2_housing_per_cap.attrs.pop("units", None)
        total_m2_housing_per_cap = prism.Q_(total_m2_housing_per_cap, "m^2/person")

        logging.debug(f"implemented '{narrow_scen}' for Residential Buildings (reduction + converge to 36)")
    else:
        logging.debug("implemented 'base' for Residential Buildings")

    return total_m2_housing_per_cap

# if 'narrow' in circular_economy_config.keys():
#     building_config = circular_economy_config["narrow"]["buildings"]
#     base_year = building_config["base_year"]
#     target_year = building_config["target_year"]

#     residential_scenario_settings = building_config['residential']['m2_change_pc']
#     implementation_rate = building_config['implementation_rate']

#     residential_scenario_settings_xr = xr.DataArray(
#         list(residential_scenario_settings.values()),
#         coords={"Region": list(residential_scenario_settings.keys())},
#         dims=["Region"],
#         name="residential_scenario_settings"
#     )

#     regions_mapped = list(region_knowledge_graph.find_relations_inverse(
#         regions, residential_scenario_settings.keys()))
#     residential_scenario_settings_xr_mapped = region_knowledge_graph.rebroadcast_xarray(
#         residential_scenario_settings_xr, output_coords=regions_mapped, dim="Region")

#     total_m2_housing_per_cap = apply_change_per_region(
#         total_m2_housing_per_cap, base_year, target_year,
#         residential_scenario_settings_xr_mapped, implementation_rate)
#     print("implemented 'narrow' for Residential Buildings")


def apply_circular_economy_commercial_floorspace(floorspace_commercial: xr.DataArray,
                                                 circular_economy_config: dict) -> xr.DataArray:
    """Implement circular economy measures for commercial floorspace.

    Parameters
    ----------
    floorspace_commercial:
        Total commercial floorspace.
    circular_economy_config:
        Configuration file of the circular economy measures.

    Returns
    -------
    floorspace_commercial:
        Updated floorspace for commercial targets with circular economy configuration.

    """

    print("FUNCTION CALLED")
    print(f"ce keys: {list(circular_economy_config.keys())}")
    region_knowledge_graph = create_region_graph()
    regions = floorspace_commercial.coords["Region"].values
    # floorspace_commercial in m^2/cap

    # Base scenario
    if 'base' in circular_economy_config.keys():
        buildings_config = circular_economy_config["base"]["buildings"]

        base_year = buildings_config["base_year"]
        target_year = buildings_config["target_year"]
        floor_pc_2020 = buildings_config["commercial"]["2020"]["useful_floor_pc"]

        floor_pc_2020_xr = xr.DataArray(
            list(floor_pc_2020.values()),
            coords={"Region": list(floor_pc_2020.keys())},
            dims=["Region"],
            name="floor_pc_2020"
        )
        floor_pc_2020_xr = prism.Q_(floor_pc_2020_xr, "m^2/person")

        regions_mapped = list(region_knowledge_graph.find_relations_inverse(
            regions, floor_pc_2020.keys()))
        floor_pc_2020_mapped = region_knowledge_graph.rebroadcast_xarray(
            floor_pc_2020_xr, output_coords=regions_mapped, dim="Region")
        target_vals = floor_pc_2020_mapped
        current_vals = floorspace_commercial.sel(Time=2020, Region=regions_mapped).sum(dim="Type")

        scaling_factors = xr.where(current_vals > 0, target_vals / current_vals, 1.0)

        floorspace_commercial.loc[{"Region": regions_mapped}] *= scaling_factors
        logging.debug("implemented 'base' for Commercial Buildings")

    ce_scen = None  # INITIALIZE ce_scen
    # Right after ce_scen is set
    print(f"ce_scen = {ce_scen}")

    if "narrow" in circular_economy_config.keys():
        ce_scen = "narrow"
    if "narrow_activity" in circular_economy_config.keys():
        ce_scen = "narrow_activity"
    # narrow_activity scenario
    if ce_scen in circular_economy_config.keys():
        commercial_ce_mode = circular_economy_config[ce_scen]["buildings"].get(
            "commercial_ce_mode", "relative")
        implementation_rate = circular_economy_config[ce_scen]['buildings']['implementation_rate']
        base_year = circular_economy_config[ce_scen]["buildings"]["base_year"]
        target_year = circular_economy_config[ce_scen]["buildings"]["target_year"]

        # --- Build the region-mapped relative-change array (shared by both modes) ---
        commercial_scenario_settings = circular_economy_config[ce_scen]["buildings"]\
            ['commercial']['m2_change_pc']

        print(f"mode = {commercial_ce_mode}")

        commercial_scenario_settings_xr = xr.DataArray(
            list(commercial_scenario_settings.values()),
            coords={"Region": list(commercial_scenario_settings.keys())},
            dims=["Region"],
            name="commercial_scenario_settings"
        )

        regions_mapped = list(region_knowledge_graph.find_relations_inverse(
            regions, commercial_scenario_settings.keys()))
        commercial_scenario_settings_xr_mapped = region_knowledge_graph.rebroadcast_xarray(
            commercial_scenario_settings_xr, output_coords=regions_mapped, dim="Region")

        region_coords = np.sort(commercial_scenario_settings_xr_mapped.coords["Region"]
                                .values.astype(int)).astype(str)
        commercial_scenario_settings_xr_mapped = region_knowledge_graph.rebroadcast_xarray(
            commercial_scenario_settings_xr_mapped, region_coords, dim="Region")

        if commercial_ce_mode == "relative":
            # ── Plain relative reductions only (no convergence, no exclusions) ──
            floorspace_commercial = apply_change_per_region(
                floorspace_commercial, base_year, target_year,
                commercial_scenario_settings_xr_mapped, implementation_rate)

            logging.debug(f"implemented '{ce_scen}' for Commercial Buildings (relative only)")

        elif commercial_ce_mode == "convergence":
            print(f"Called with ce keys: {list(circular_economy_config.keys())}")
            # ── Relative reductions + three-category convergence toward cap ──
            convergence_cap = float(circular_economy_config[ce_scen]["buildings"].get(
                "convergence_cap", 14.0))
            convergence_year_end = int(circular_economy_config[ce_scen]["buildings"].get(
                "convergence_year_end", 2100))
            low_threshold = float(circular_economy_config[ce_scen]["buildings"].get(
                "low_threshold", 5.0))

            # Dequantify to plain floats upfront — .loc item-assignment strips pint units,
            # so working with plain floats throughout avoids the DimensionalityError at the end.
            if prism.U_(floorspace_commercial) is not None:
                floorspace_commercial = floorspace_commercial.pint.dequantify()

            # Save baseline (plain float) before any narrow_activity changes
            baseline = floorspace_commercial.copy(deep=True)

            # Classify regions by 2020 per-capita commercial floorspace (plain float)
            total_pc_2020 = floorspace_commercial.sel(Time=2020).sum(dim="Type")

            # Helper: numeric region string -> name (1-indexed into IMAGE_REGIONS)
            def _reg_name(r):
                try:
                    return IMAGE_REGIONS[int(str(r)) - 1]
                except (ValueError, IndexError):
                    return str(r)

            low_regions = total_pc_2020.where(total_pc_2020 < low_threshold, drop=True)\
                .coords["Region"].values
            low_region_strs = set(str(r) for r in low_regions)

            logging.info(
                "Commercial CE: low regions (< %.1f m²/cap in 2020, no relative reduction): %s",
                low_threshold, [_reg_name(r) for r in low_regions]
            )

            # Phase 1: relative reductions until target_year (skip low regions)
            changes_for_convergence = commercial_scenario_settings_xr_mapped.copy()
            for r in low_region_strs:
                if r in changes_for_convergence.coords["Region"].values:
                    changes_for_convergence.loc[{"Region": r}] = 0.0

            floorspace_commercial = apply_change_per_region(
                floorspace_commercial, base_year, target_year,
                changes_for_convergence, implementation_rate)

            # Re-dequantify in case apply_change_per_region re-added pint units
            if prism.U_(floorspace_commercial) is not None:
                floorspace_commercial = floorspace_commercial.pint.dequantify()

            # apply_change_per_region uses groupby+concat which may reorder the Region
            # coordinate (xarray sorts lexicographically). Realign baseline so that all
            # subsequent .sel() calls work without AlignmentError.
            baseline = baseline.reindex_like(floorspace_commercial)

            # Prevent the narrow_activity scenario from exceeding the baseline
            # (use np.minimum on raw arrays to avoid xarray exact-alignment issues).
            floorspace_commercial.values = np.minimum(
                floorspace_commercial.values, baseline.values
            )

            # Phase 2: convergence after target_year (all comparisons on plain floats)
            total_pc_at_target = floorspace_commercial.sel(Time=target_year).sum(dim="Type")

            # Exclude low regions from above-cap classification (they are handled separately)
            above_cap_at_target = [
                r for r in total_pc_at_target.where(
                    total_pc_at_target > convergence_cap, drop=True).coords["Region"].values
                if str(r) not in low_region_strs
            ]

            mid_range_regions = [
                r for r in total_pc_at_target.coords["Region"].values
                if str(r) not in low_region_strs
                and str(r) not in set(str(r) for r in above_cap_at_target)
            ]

            logging.info("CE phase 2 — above cap at %s (linear decline to cap by %s): %s",
                          target_year, convergence_year_end,
                          [_reg_name(r) for r in above_cap_at_target])
            logging.info("CE phase 2 — low regions (follow baseline, linear cap at %s by %s): %s",
                          convergence_cap, convergence_year_end,
                          [_reg_name(r) for r in low_regions])
            logging.info("CE phase 2 — mid-range (linear transition to baseline, capped at %s by %s): %s",
                          convergence_cap, convergence_year_end,
                          [_reg_name(r) for r in mid_range_regions])

            all_times = floorspace_commercial.Time.values
            post_mask = all_times > target_year

            ramp = xr.DataArray(
                np.clip(
                    (all_times - target_year) / (convergence_year_end - target_year),
                    0.0, 1.0
                ),
                dims=["Time"],
                coords={"Time": all_times},
            )

            def _capped_end_vals(fs_end: np.ndarray) -> np.ndarray:
                """Scale type values so total does not exceed convergence_cap."""
                total = fs_end.sum()
                if total > convergence_cap and total > 0:
                    return fs_end * (convergence_cap / total)
                return fs_end.copy()

            # (a) Above cap at target_year — linear decline to cap by convergence_year_end
            for reg in above_cap_at_target:
                fs_at_target = floorspace_commercial.sel(Time=target_year, Region=reg).values
                total_at_target = fs_at_target.sum()
                type_shares = fs_at_target / total_at_target if total_at_target > 0 \
                    else np.zeros_like(fs_at_target)
                end_vals = convergence_cap * type_shares  # exactly at cap

                for t in all_times[post_mask]:
                    r = float(ramp.sel(Time=t))
                    floorspace_commercial.loc[{"Time": t, "Region": reg}] = \
                        fs_at_target * (1 - r) + end_vals * r

            # (b) Low regions — follow baseline, linearly capped if baseline exceeds cap.
            # Restore baseline for all years first, then apply a smooth linear ramp
            # from the baseline value at target_year toward the capped endpoint at
            # convergence_year_end (identical ramp logic to category a).
            for reg in low_regions:
                floorspace_commercial.loc[{"Region": reg}] = baseline.sel(Region=reg).values

            for reg in low_regions:
                fs_at_target = floorspace_commercial.sel(Time=target_year, Region=reg).values
                baseline_end = baseline.sel(Region=reg, Time=convergence_year_end).values
                end_vals = _capped_end_vals(baseline_end)

                for t in all_times[post_mask]:
                    r = float(ramp.sel(Time=t))
                    floorspace_commercial.loc[{"Time": t, "Region": reg}] = \
                        fs_at_target * (1 - r) + end_vals * r

            # (c) Mid-range — linear transition from CE value at target_year toward
            # the (capped) baseline value at convergence_year_end.
            for reg in mid_range_regions:
                fs_at_target = floorspace_commercial.sel(Time=target_year, Region=reg).values
                baseline_end = baseline.sel(Region=reg, Time=convergence_year_end).values
                end_vals = _capped_end_vals(baseline_end)

                for t in all_times[post_mask]:
                    r = float(ramp.sel(Time=t))
                    floorspace_commercial.loc[{"Time": t, "Region": reg}] = \
                        fs_at_target * (1 - r) + end_vals * r

            # Final clamp: ensure convergence interpolation never exceeds baseline
            # at any year (the linear ramp can overshoot baseline at intermediate years
            # when the baseline is non-monotonic).
            floorspace_commercial.values = np.minimum(
                floorspace_commercial.values, baseline.values
            )

            logging.debug(
                f"implemented '{ce_scen}' for Commercial Buildings (relative + convergence)")

        else:
            raise ValueError(
                f"Invalid commercial_ce_mode: '{commercial_ce_mode}'. "
                "Use 'relative' or 'convergence'."
            )
        # Re-attach unit cleanly. After dequantify(), pint-xarray stores "dimensionless"
        # in attrs["units"], which makes prism.Q_ call .pint.to() and raise a
        # DimensionalityError. Strip that attr first so Q_ does a fresh assignment.
        if prism.U_(floorspace_commercial) is not None:
            floorspace_commercial = floorspace_commercial.pint.dequantify()
        floorspace_commercial.attrs.pop("units", None)
        floorspace_commercial = prism.Q_(floorspace_commercial, "m^2/person")

    return floorspace_commercial.transpose("Time", "Region", "Type")


def circular_economy_measures_material_intensities_residential(
        xr_mat_res_intensities: xr.DataArray, circular_economy_config: dict) -> xr.DataArray:
    """Adjust material intensities for circular economy.

    Parameters
    ----------
    xr_mat_res_intensities:
        Material intensities before circular economy adjustments.
    circular_economy_config:
        Configuration for circular economy parameters.

    Returns
    -------
    xr_mat_res_intensities:
        Adjusted material intensities for circular economy.

    """
    # rename Cohort to Time for compatibility with apply_change_per_region function
    if "Cohort" in xr_mat_res_intensities.dims:
        xr_mat_res_intensities = xr_mat_res_intensities.rename({"Cohort": "Time"})

    ce_scen = None  # INITIALIZE ce_scen

    if "narrow_product" in circular_economy_config.keys():
        ce_scen = "narrow_product"

    # import parameters from config file
    target_year = circular_economy_config[ce_scen]['buildings']['target_year']
    base_year = circular_economy_config[ce_scen]['buildings']['base_year']
    implementation_rate = circular_economy_config[ce_scen]['buildings']['implementation_rate']
    mat_changes = circular_economy_config[ce_scen]['buildings']['material_intensity_change']

    region_knowledge_graph = create_region_graph()
    model_regions = list(xr_mat_res_intensities.coords["Region"].values) # regions in model
    materials_all = set(xr_mat_res_intensities.coords["material"].values) # all materials in data

    for mat in ("steel", "concrete", "aluminium"):              # in residential buildings we apply lightweighting to concrete and later derive the cement demand from that
        if mat not in mat_changes or mat not in materials_all:
            continue

        # 1) TOML -> 1-D DA over region names
        change_dict = mat_changes[mat]
        raw = xr.DataArray(
            list(change_dict.values()),
            coords={"Region": list(change_dict.keys())},
            dims=["Region"],
            name=f"material_intensity_change_{mat}",
        )

        # 2) map region names -> region codes, then align to model order
        regions_mapped = list(region_knowledge_graph.find_relations_inverse(
            model_regions, raw.coords["Region"].values))
        changes_mapped = region_knowledge_graph.rebroadcast_xarray(
            raw, output_coords=regions_mapped, dim="Region")
        changes_mapped = changes_mapped.sel(Region=model_regions).astype(float)

        # 3) apply once per material
        cur = xr_mat_res_intensities.sel(material=mat)
        updated = apply_change_per_region(cur, base_year, target_year, changes_mapped,
                                          implementation_rate)
        updated = updated.reindex(Region=cur.coords["Region"])
        xr_mat_res_intensities.loc[dict(material=mat)] = updated

    # rename back
    if "Time" in xr_mat_res_intensities.dims:
        xr_mat_res_intensities = xr_mat_res_intensities.rename({"Time": "Cohort"})

    logging.debug(f"implemented '{ce_scen}' for Residential Buildings (lightweighting)")
    return xr_mat_res_intensities


def circular_economy_measures_material_intensities_commercial(xr_mat_comm_intensities: xr.DataArray,
                                                              circular_economy_config: dict,
                                                              model_regions: list):
    """Adjust the commercial material intensities for circular economy measures.

    Parameters
    ----------
    xr_mat_comm_intensities:
        Material intensities for the commercial buildings.
    circular_economy_config:
        Configuration for the circular economy.
    model_regions:
        Regions that are in the model.

    Returns
    -------
    xr_mat_comm_intensities:
        Updated commercial material intensities.

    """
    # work array with Time dim
    xr_mat_comm_intensities = (xr_mat_comm_intensities.rename({"Cohort": "Time"})
              if "Cohort" in xr_mat_comm_intensities.dims else xr_mat_comm_intensities)
    
    ce_scen = None  # INITIALIZE ce_scen
    if "narrow" in circular_economy_config.keys():
        ce_scen = "narrow"
    if "narrow_product" in circular_economy_config.keys():
        ce_scen = "narrow_product"

    base_year = circular_economy_config[ce_scen]['buildings']['base_year']
    target_year = circular_economy_config[ce_scen]['buildings']['target_year']
    implementation_rate = circular_economy_config[ce_scen]['buildings']['implementation_rate']
    mat_changes = circular_economy_config[ce_scen]['buildings']['material_intensity_change']

    region_graph = create_region_graph()
    materials_all = list(xr_mat_comm_intensities.coords["material"].values) #

    updated_slices = []

    for mat in ("steel", "cement", "aluminium"):                    # in commercial buildings we apply lightweighting to cement instead of concrete
        if mat not in mat_changes or mat not in materials_all:
            continue
        cur = xr_mat_comm_intensities.sel(material=mat)

        # only apply for those present in TOML; others pass through unchanged
        if mat in mat_changes:
            change_dict = mat_changes[mat]
            raw = xr.DataArray(
                list(change_dict.values()),
                coords={"Region": list(change_dict.keys())},
                dims=["Region"],
                name=f"mi_change_pc_{mat}",
            )

            # map region names -> region codes; align to model order
            regions_mapped = list(region_graph.find_relations_inverse(model_regions,
                                                                      raw.coords["Region"].values))
            changes_mapped = region_graph.rebroadcast_xarray(raw, output_coords=regions_mapped,
                                                             dim="Region")
            changes_mapped = changes_mapped.sel(Region=model_regions).astype(float)

            # apply once per material
            updated = apply_change_per_region(cur, base_year, target_year, changes_mapped,
                                              implementation_rate)
            # keep Region order & dim order identical to cur
            updated = updated.reindex(Region=cur.coords["Region"]).transpose(*cur.dims)
        else:
            updated = cur

        # attach the material coord and collect
        updated_slices.append(updated.expand_dims(material=[mat]))

    xr_mat_updated = xr.concat(updated_slices, dim="material")

    # rename back to Cohort if needed
    xr_mat_comm_intensities = (xr_mat_updated.rename({"Time": "Cohort"})
                               if "Time" in xr_mat_updated.dims else xr_mat_updated)

    logging.debug("implemented 'narrow_product' for Commercial Buildings (lightweighting)")
    return xr_mat_comm_intensities
