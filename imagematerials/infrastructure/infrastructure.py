# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import numpy as np
import xarray as xr
import scipy
import warnings
from pathlib import Path

from imagematerials.infrastructure.preprocessing import get_preprocessing_data_infrastructure
from imagematerials.model import GenericStocks, MaterialIntensities
from imagematerials.factory import ModelFactory, Sector
from imagematerials.__main__ import export_model_netcdf, _convert_timevar
import prism

SCEN = "SSP2"

# Simulation horizon — declared locally, matching the pattern used in
# imagematerials.electricity.electricity. Adjust here if you want a different
# range for a standalone infrastructure run.
YEAR_START = 1971   # first year of the simulation
YEAR_END = 2100     # last year of the simulation
YEAR_OUT = 2100     # last year of reporting
path_current = Path(__file__).resolve()
path_base = path_current.parent.parent.parent # C:\IMAGE\image-materials

def process_da_to_df(da, metric_name, unit):
    if len(da.dims) == 0:
        return pd.DataFrame()
    df = da.to_dataframe(name='value').reset_index()
    df['metric'] = metric_name
    df['unit'] = unit
    return df


def run_infrastructure_simulation(prep_data=None, scenario=SCEN):
    """
    Build and simulate the infrastructure sector model.

    This is the integration-ready entry point: it produces a populated
    ModelFactory whose submodels expose TimeVariable outputs that can be
    consumed by downstream models (e.g. SumMaterials in run_integrated.py).

    Parameters
    ----------
    prep_data : dict, optional
        Preprocessing dictionary from get_preprocessing_data_infrastructure.
        If None, it is loaded from disk using `scenario`.
    scenario : str
        Scenario name, used only when prep_data is None.

    Returns
    -------
    main_model_factory : prism.ModelFactory
        The finished factory after simulation. Submodels:
            [0] GenericStocks       → inflow, outflow_by_cohort, stock_by_cohort
            [1] MaterialIntensities → inflow_materials, outflow_by_cohort_materials,
                                      stock_by_cohort_materials
    prep_data : dict
        The preprocessing dict used (returned for downstream reporting).
    """
    if prep_data is None:
        prep_data = get_preprocessing_data_infrastructure(path_base, scenario)

    print("\n--- Running Infrastructure Sector Model ---")
    time_start = int(prep_data["stocks"].coords["Time"].min().values)

    complete_timeline = prism.Timeline(time_start, YEAR_END, 1)
    # Simulate from time_start (1911) so DSM processes stock_tail years
    # and builds proper cohort age distribution (not all stock in one cohort)
    simulation_timeline = prism.Timeline(time_start, YEAR_END, 1)

    sec_infrastructure = Sector("infrastructure", prep_data, check_coordinates=False)

    # GenericStocks requires a Cohort coordinate definition matching complete_timeline
    if "Cohort" not in sec_infrastructure.coordinates:
        sec_infrastructure.coordinates["Cohort"] = list(range(time_start, YEAR_END + 1))

    main_model_factory = ModelFactory(
        [sec_infrastructure], complete_timeline
        ).add(GenericStocks, "infrastructure"
        ).add(MaterialIntensities, "infrastructure"
        ).finish()

    # Simulate
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        main_model_factory.simulate(simulation_timeline)
    print("Simulation finished.")

    # Attach a Time coordinate DataArray for convenience in reporting
    infra_model = main_model_factory.submodels[0]
    times = np.array(infra_model.Cohort)
    infra_model.time_coor = xr.DataArray(times, dims=["Time"], coords={"Time": times})

    return main_model_factory, prep_data


def export_infrastructure_report(main_model_factory, prep_data, output_dir=None):
    """
    Stand-alone reporting: extract metrics from a simulated infrastructure model,
    apply mass balance and permanent-aggregate corrections, and export to
    NetCDF (road_materials_detailed.nc) and wide-format Excel
    (road_materials_output_kt_detail.xlsx).

    This is NOT needed for integration with the rest of the image-materials
    model — it is purely for standalone analysis/reporting.
    """
    infra_model = main_model_factory.submodels[0]
    mat_model = main_model_factory.submodels[1]

    if output_dir is None:
        output_dir = path_current.parent / "output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Compute aggregated outputs (sum over cohorts)
    print("\nExtracting metrics...")

    time_slice = slice(YEAR_START, YEAR_END)
    time_coor_da = infra_model.time_coor
    
    # physical Units (km)
    da_inflow_km = _convert_timevar(infra_model.inflow, time_coor_da).sel(Time=time_slice)
    da_outflow_km = _convert_timevar(infra_model.outflow_by_cohort, time_coor_da).sum(dim='Cohort').sel(Time=time_slice)

    # Material Units (kg)
    da_inflow_mat = _convert_timevar(mat_model.inflow_materials, time_coor_da).sel(Time=time_slice)
    da_outflow_mat = _convert_timevar(mat_model.outflow_by_cohort_materials, time_coor_da).sel(Time=time_slice)

    # Mass balance: subtract obsolete inflow from active outflow (v5 lines 2793-2809)
    active_types = prep_data.get("active_types", [])
    obsolete_types = prep_data.get("obsolete_types", [])
    obs_to_active_map = prep_data.get("obsolete_to_active_map", {})

    # Filter active types to those actually in the model
    all_model_types = list(da_inflow_km.coords['Type'].values)
    active_types = [t for t in active_types if t in all_model_types]
    obsolete_types = [t for t in obsolete_types if t in all_model_types]

    # Select active-only views
    da_inflow_km_active = da_inflow_km.sel(Type=active_types)
    da_outflow_km_active = da_outflow_km.sel(Type=active_types).copy()
    da_inflow_mat_active = da_inflow_mat.sel(Type=active_types)
    da_outflow_mat_active = da_outflow_mat.sel(Type=active_types).copy()

    # Apply mass balance: subtract each obsolete inflow from its active counterpart's outflow
    if obsolete_types and obs_to_active_map:
        da_obs_inflow_km = da_inflow_km.sel(Type=obsolete_types)
        da_obs_inflow_mat = da_inflow_mat.sel(Type=obsolete_types)

        # Rename obsolete types to their active counterparts for alignment
        active_names = [obs_to_active_map.get(t, t) for t in obsolete_types]
        da_obs_inflow_km = da_obs_inflow_km.assign_coords(Type=active_names)
        da_obs_inflow_mat = da_obs_inflow_mat.assign_coords(Type=active_names)

        # Group by active type (multiple obsolete can map to same active)
        obs_km_grouped = da_obs_inflow_km.groupby('Type').sum()
        obs_mat_grouped = da_obs_inflow_mat.groupby('Type').sum()

        # Subtract only for types that have obsolete counterparts
        for active_t in obs_km_grouped.coords['Type'].values:
            if active_t in da_outflow_km_active.coords['Type'].values:
                da_outflow_km_active.loc[dict(Type=active_t)] = np.maximum(
                    da_outflow_km_active.sel(Type=active_t) - obs_km_grouped.sel(Type=active_t), 0)
                da_outflow_mat_active.loc[dict(Type=active_t)] = np.maximum(
                    da_outflow_mat_active.sel(Type=active_t) - obs_mat_grouped.sel(Type=active_t), 0)

    da_inflow_km = da_inflow_km_active
    da_outflow_km = da_outflow_km_active
    da_inflow_mat = da_inflow_mat_active
    da_outflow_mat = da_outflow_mat_active
    da_expansion_km = da_inflow_km - da_outflow_km
    da_stock_km = infra_model.stock_by_cohort.sum(dim='Cohort').sel(Time=time_slice, Type=active_types)

    da_expansion_mat = da_inflow_mat - da_outflow_mat
    da_stock_mat = mat_model.stock_by_cohort_materials.sel(Time=time_slice, Type=active_types)

    # --- Permanent aggregate (80%) — add to stock and inflow for reporting ---
    perm_agg_mi = prep_data.get("permanent_aggregate_mi", None)
    if perm_agg_mi is not None:
        # Physical stock — strip pint units for computation
        phys_stock_values = da_stock_km.values if hasattr(da_stock_km, 'values') else da_stock_km
        try:
            phys_stock_values = phys_stock_values.magnitude
        except AttributeError:
            pass
        phys_stock_plain = xr.DataArray(
            phys_stock_values,
            coords=da_stock_km.coords,
            dims=da_stock_km.dims
        )
        # Permanent agg stock = physical_stock × permanent_MI (per type, region)
        perm_agg_stock = phys_stock_plain * perm_agg_mi

        # Permanent agg inflow = increase in permanent stock (diff, clipped at 0)
        perm_agg_inflow = perm_agg_stock.diff(dim="Time").clip(min=0)
        # First year: permanent stock IS the initial inflow
        first_year_stock = perm_agg_stock.isel(Time=0)
        perm_agg_inflow = xr.concat([first_year_stock.expand_dims("Time"), perm_agg_inflow], dim="Time")

        # Add permanent aggregate to material stock and inflow
        perm_stock_expanded = perm_agg_stock.expand_dims({"material": ["aggregate"]})
        perm_inflow_expanded = perm_agg_inflow.expand_dims({"material": ["aggregate"]})

        # Strip pint units from material DataArrays for addition
        def _strip_pint(da):
            try:
                return da.pint.dequantify() * 1.0  # Remove pint wrapping
            except Exception:
                try:
                    vals = da.values
                    if hasattr(vals, 'magnitude'):
                        vals = vals.magnitude
                    return xr.DataArray(np.array(vals, dtype=float), coords=da.coords, dims=da.dims)
                except Exception:
                    return da

        da_stock_mat_plain = _strip_pint(da_stock_mat)
        da_inflow_mat_plain = _strip_pint(da_inflow_mat)
        da_outflow_mat_plain = _strip_pint(da_outflow_mat)

        perm_stock_aligned = perm_stock_expanded.reindex_like(da_stock_mat_plain, fill_value=0)
        perm_inflow_aligned = perm_inflow_expanded.reindex_like(da_inflow_mat_plain, fill_value=0)

        da_stock_mat = da_stock_mat_plain + perm_stock_aligned
        da_inflow_mat = da_inflow_mat_plain + perm_inflow_aligned
        da_outflow_mat = da_outflow_mat_plain
        da_expansion_mat = da_inflow_mat - da_outflow_mat

    # --- 2. Export aggregated NetCDF (no cohort dimension → compact) ---
    # Aggregate material stock/outflow over Cohort before saving
    da_stock_mat_agg = da_stock_mat.sum(dim='Cohort') if 'Cohort' in da_stock_mat.dims else da_stock_mat
    da_outflow_mat_agg = da_outflow_mat.sum(dim='Cohort') if 'Cohort' in da_outflow_mat.dims else da_outflow_mat

    nc_data = {}
    for da, name in [
        (da_inflow_mat, "material_inflow"),
        (da_outflow_mat_agg, "material_outflow"),
        (da_stock_mat_agg, "material_stock"),
        (da_expansion_mat, "material_expansion"),
        (da_inflow_km, "physical_inflow"),
        (da_outflow_km, "physical_outflow"),
        (da_stock_km, "physical_stock"),
        (da_expansion_km, "physical_expansion"),
    ]:
        # Strip pint units for clean NetCDF export
        try:
            vals = da.values
            if hasattr(vals, 'magnitude'):
                vals = vals.magnitude
            clean = xr.DataArray(np.array(vals, dtype=float), coords=da.coords, dims=da.dims)
        except Exception:
            clean = da
        nc_data[name] = clean

    nc_output_path = output_dir / "road_materials_detailed.nc"
    xr.Dataset(nc_data).to_netcdf(nc_output_path, engine="netcdf4")
    print(f"Saved aggregated output to {nc_output_path}")

    # --- 3. Export wide-format Excel matching v5 layout ---
    # v5 columns: regions, flow, sector, category, elements, materials, <year cols>
    dfs = []
    for da, flow_name in [
        (da_inflow_mat, "inflow"),
        (da_outflow_mat_agg, "outflow"),
        (da_stock_mat_agg, "stock"),
    ]:
        df = da.to_dataframe(name='value').reset_index()
        if 'Material' in df.columns:
            df.rename(columns={'Material': 'material'}, inplace=True)
        df['flow'] = flow_name
        dfs.append(df)

    long_df = pd.concat(dfs, ignore_index=True)
    long_df.rename(columns={'Time': 'year', 'Region': 'regions',
                            'material': 'materials', 'Type': 'elements'}, inplace=True)
    # Convert kg → kt
    long_df['value'] = long_df['value'] / 1e6
    long_df['sector'] = 'transport_infra'
    long_df['category'] = 'network'
    # Rename 'brick' → 'bricks' to match v5
    long_df['materials'] = long_df['materials'].replace({'brick': 'bricks'})

    # Add total_tram as zero-valued placeholder (declared in v5 but never computed)
    materials_list = long_df['materials'].dropna().unique()
    regions_list = long_df['regions'].unique()
    years_list = sorted(long_df['year'].unique())
    tram_rows = []
    for flow_name in ['inflow', 'outflow', 'stock']:
        for mat in materials_list:
            for reg in regions_list:
                tram_rows.append({
                    'regions': reg, 'flow': flow_name, 'sector': 'transport_infra',
                    'category': 'network', 'elements': 'total_tram',
                    'materials': mat, 'year': years_list[0], 'value': 0.0
                })
    long_df = pd.concat([long_df, pd.DataFrame(tram_rows)], ignore_index=True)

    # Pivot to wide format: years as columns
    wide_df = long_df.pivot_table(
        index=['regions', 'flow', 'sector', 'category', 'elements', 'materials'],
        columns='year', values='value', aggfunc='sum', fill_value=0.0
    ).reset_index()
    wide_df.columns.name = None  # remove pivot column name

    print(f"\nOutput: {wide_df.shape[0]} rows × {wide_df.shape[1]} cols")
    print("Materials:", ", ".join(sorted(wide_df['materials'].unique())))
    print("Elements:", len(wide_df['elements'].unique()))

    report_path = output_dir / "road_materials_output_kt_detail.xlsx"
    wide_df.to_excel(report_path, index=False, sheet_name='network')
    print(f"Saved v5-format output to {report_path}")


def main(export_report=True, scenario=SCEN):
    """
    Stand-alone entry point for the infrastructure sector.

    Runs the simulation and (optionally) exports the NetCDF + Excel reports.
    When infrastructure is run as part of the integrated model
    (see examples/run_integrated.py), only run_infrastructure_simulation()
    is needed — the reporting step is skipped.

    Parameters
    ----------
    export_report : bool
        If True (default), write road_materials_detailed.nc and
        road_materials_output_kt_detail.xlsx to imagematerials/infrastructure/output/.
        If False, just run the simulation and return the factory (useful for
        testing or for callers that only want the in-memory results).
    scenario : str
        Scenario name passed to preprocessing.

    Returns
    -------
    main_model_factory : prism.ModelFactory
    prep_data : dict
    """
    main_model_factory, prep_data = run_infrastructure_simulation(scenario=scenario)

    if export_report:
        export_infrastructure_report(main_model_factory, prep_data)
    else:
        print("Skipping standalone report export (simulation results available in returned factory).")

    return main_model_factory, prep_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the infrastructure sector model standalone."
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip the NetCDF/Excel report export (simulation only).",
    )
    parser.add_argument(
        "--scenario",
        default=SCEN,
        help=f"Scenario name (default: {SCEN}).",
    )
    args = parser.parse_args()
    main(export_report=not args.no_report, scenario=args.scenario)
