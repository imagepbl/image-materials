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
    simulation_timeline = prism.Timeline(YEAR_START, YEAR_END, 1)

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
    da_expansion_km = da_inflow_km - da_outflow_km
    da_stock_km = infra_model.stock_by_cohort.sum(dim='Cohort').sel(Time=time_slice)
    
    # Material Units (kg)
    da_inflow_mat = _convert_timevar(mat_model.inflow_materials, time_coor_da).sel(Time=time_slice)
    da_outflow_mat = _convert_timevar(mat_model.outflow_by_cohort_materials, time_coor_da).sel(Time=time_slice)
    da_expansion_mat = da_inflow_mat - da_outflow_mat
    da_stock_mat = mat_model.stock_by_cohort_materials.sel(Time=time_slice)
    
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
