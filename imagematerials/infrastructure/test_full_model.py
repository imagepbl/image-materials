"""Test full infrastructure model run."""
import warnings
import traceback
import numpy as np
import xarray as xr
from pathlib import Path

from imagematerials.infrastructure.preprocessing import get_preprocessing_data_infrastructure
from imagematerials.model import GenericStocks, MaterialIntensities
from imagematerials.factory import ModelFactory, Sector
import prism

SCEN = "SSP2"
path_base = Path(r"C:\IMAGE\image-materials")
YEAR_START = 1971
YEAR_END = 2100

print("Running preprocessing...")
prep_data = get_preprocessing_data_infrastructure(path_base, SCEN)
print("Preprocessing done.")

print("\n--- Setting up infrastructure model ---")
time_start = int(prep_data["stocks"].coords["Time"].min().values)
print(f"Time range: {time_start} - {YEAR_END}")

complete_timeline = prism.Timeline(time_start, YEAR_END, 1)
simulation_timeline = prism.Timeline(YEAR_START, YEAR_END, 1)

sec_infrastructure = Sector("infrastructure", prep_data, check_coordinates=False)

if "Cohort" not in sec_infrastructure.coordinates:
    sec_infrastructure.coordinates["Cohort"] = list(range(time_start, YEAR_END + 1))

print("Building model factory...")
try:
    main_model_factory = ModelFactory(
        [sec_infrastructure], complete_timeline
    ).add(GenericStocks, "infrastructure"
    ).add(MaterialIntensities, "infrastructure"
    ).finish()
    print("Model factory built.")
except Exception as e:
    traceback.print_exc()
    raise

print("\n--- Running simulation ---")
try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        main_model_factory.simulate(simulation_timeline)
    print("Simulation finished successfully!")
except Exception as e:
    traceback.print_exc()
    raise

# Quick output check
infra_model = main_model_factory.submodels[0]
mat_model = main_model_factory.submodels[1]

print("\n--- Output summary ---")
stock = infra_model.stock_by_cohort.sum(dim="Cohort")
print(f"Total stock 2024 (all types, all regions): {float(stock.sel(Time=2024).sum()):.2f} km2")
print(f"Total stock 2050 (all types, all regions): {float(stock.sel(Time=2050).sum()):.2f} km2")
print(f"Total stock 2100 (all types, all regions): {float(stock.sel(Time=2100).sum()):.2f} km2")

# Check material flows for a sample year
from imagematerials.__main__ import _convert_timevar
times = np.array(infra_model.Cohort)
time_coor_da = xr.DataArray(times, dims=["Time"], coords={"Time": times})
da_inflow_mat = _convert_timevar(mat_model.inflow_materials, time_coor_da)
print(f"\nMaterial inflow 2024 total (all materials, all types, all regions): {float(da_inflow_mat.sel(Time=2024).sum()):.2e} kg")
print(f"Material inflow 2050 total: {float(da_inflow_mat.sel(Time=2050).sum()):.2e} kg")

# Check by material
for mat in da_inflow_mat.coords["material"].values:
    val = float(da_inflow_mat.sel(Time=2024, material=mat).sum())
    if val > 0:
        print(f"  {mat} inflow 2024: {val:.2e} kg ({val/1e9:.2f} kt)")

print("\nDone!")
