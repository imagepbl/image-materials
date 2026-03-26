# -*- coding: utf-8 -*-
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
path_current = Path(__file__).resolve()
path_base = path_current.parent.parent.parent # C:\IMAGE\image-materials

YEAR_START = 1971   # start year of the simulation period
YEAR_END = 2100     # end year of the calculations
YEAR_OUT = 2100     # year of output generation = last year of reporting

def process_da_to_df(da, metric_name, unit):
    if len(da.dims) == 0:
        return pd.DataFrame()
    df = da.to_dataframe(name='value').reset_index()
    df['metric'] = metric_name
    df['unit'] = unit
    return df

def main():
    prep_data = get_preprocessing_data_infrastructure(path_base, SCEN)

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
    
    # Save standard outputs
    infra_model = main_model_factory.submodels[0]
    times = np.array(infra_model.Cohort)
    infra_model.time_coor = xr.DataArray(times, dims=["Time"], coords={"Time": times})
    
    output_dir = path_current.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Export Raw NetCDF
    nc_output_path = output_dir / "road_materials_detailed.nc"
    export_model_netcdf(infra_model, nc_output_path)
    print(f"Saved raw generic output to {nc_output_path}")
    
    # 2. Export Detailed Reports requested by user
    print("\nExtracting metrics for Expansion, Maintenance, and Cohorts...")
    
    time_slice = slice(YEAR_START, YEAR_END)
    mat_model = main_model_factory.submodels[1]
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

    dfs = []
    # Note: 'value' column will contain output values, but first we drop 0s to save memory and space
    for da, metric, unit in [
        (da_inflow_km, "Total Inflow (Construction)", "km"),
        (da_outflow_km, "Total Outflow (Maintenance)", "km"),
        (da_expansion_km, "Net Expansion (New Stock)", "km"),
        (da_stock_km, "Active Stock", "km"),
        (da_inflow_mat, "Material Inflow", "kg"),
        (da_outflow_mat, "Material Outflow (Replaced)", "kg"),
        (da_expansion_mat, "Material Expansion", "kg"),
        (da_stock_mat, "Material Stock", "kg")
    ]:
        df = process_da_to_df(da, metric, unit)
        if not df.empty:
            df = df[df['value'] != 0].copy()
            dfs.append(df)
            
    final_report = pd.concat(dfs, ignore_index=True)
    
    # Normalize Material dimension name if it differs ('Material' vs 'material')
    if 'Material' in final_report.columns:
        final_report.rename(columns={'Material': 'material'}, inplace=True)
    
    # Standardize column casing
    final_report.rename(columns={'Time': 'year', 'Region': 'region'}, inplace=True)
    print("\nMaterials Found List:")
    if 'material' in final_report.columns:
        print(", ".join(final_report['material'].dropna().unique().astype(str)))
        
    report_path = output_dir / "road_materials_detailed_report.csv"
    final_report.to_csv(report_path, index=False)
    print(f"Saved highly detailed physical and material tracking to {report_path}")

    # 3. Export Cohorts Separation
    cohort_stock = infra_model.stock_by_cohort.sel(Time=time_slice).to_dataframe(name='stock_km').reset_index()
    cohort_stock = cohort_stock[cohort_stock['stock_km'] > 0]
    cohort_stock.rename(columns={'Time': 'year', 'Region': 'region'}, inplace=True)
    
    cohort_path = output_dir / "road_stock_by_age_cohort.csv"
    cohort_stock.to_csv(cohort_path, index=False)
    print(f"Saved cohort age distributions to {cohort_path}")

if __name__ == "__main__":
    main()
