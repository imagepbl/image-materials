# TODO write this as normal .py script
import matplotlib.pyplot as plt
import xarray as xr
import pym

from imagematerials.factory import ModelFactory
from imagematerials.read_mym import read_mym_df
from imagematerials.rest_of.util import sum_inflows_for_all_sectors, calculate_cement_equivalent, sand_gravel_crushed_rock_equivalent
from imagematerials.constants import IMAGE_REGIONS

def load_model_runs(base_scen = 'SSP2_baseline', 
                    eff_scen = 'SSP2_baseline_with_re', 
                    base_directory = "model_results/", 
                    scenarios: list = ["SSP2_baseline", "SSP2_baseline_with_re"]):
    model_runs = {scenario_name: ModelFactory.load_pkl(f"{base_directory}{scenario_name}_model.pkl") for scenario_name in scenarios}
    baseline = model_runs.get(base_scen)
    resource_eff = model_runs.get(eff_scen)
    return baseline, resource_eff

def read_in_population_and_gdp_cap(base_directory = "../data/raw/image/", 
                                   scenario: str = "SSP2_baseline"):
    gdp_cap = read_mym_df(f'{base_directory}{scenario}/Socioeconomic/gdp_pc.scn')
    population = read_mym_df(f'{base_directory}{scenario}/Socioeconomic/pop.scn')
    gdp_cap.index.name = "time"
    population.index.name = "time"

    gdp_cap_xr = xr.DataArray(
        gdp_cap.loc[:, :26].values,
        dims=["time", "Region"],
        coords={"time": gdp_cap.index, "Region": IMAGE_REGIONS}
    )

    population_xr = xr.DataArray(
        population.loc[:, :26].values,
        dims=["time", "Region"],
        coords={"time": population.index, "Region": IMAGE_REGIONS}
    )

    gdp = gdp_cap_xr * population_xr
    return gdp

def read_gompertz_coefs():
    # Read gompertz calues from standard run 
    gompertz_original = xr.open_dataarray('../data/raw/rest-of/gompertz_values/coefs_gompertz.nc')
    return gompertz_original


def calcualte_resource_efficiency_measures():
    gdp = read_in_population_and_gdp_cap()
    gompertz_original = read_gompertz_coefs()
    baseline, resource_eff = load_model_runs()
    # sum all inflows for both runs
    list_sum_sectors_all = ["buildings", "vehicles", "elc_gen", "elc_grid_lines", "elc_grid_add", "elc_stor_phs", "elc_stor_other"]
    total_inflow_base = sum_inflows_for_all_sectors(baseline, 'inflow_materials', list_sum_sectors_all)
    total_inflow_eff = sum_inflows_for_all_sectors(resource_eff, 'inflow_materials', list_sum_sectors_all)

    for material in gompertz_original.material.values:
        if material in ['clay', 'limestone']: 
            # for limestone and clay take sand_gravel_crushed_rock as a proxy
            inflow = sand_gravel_crushed_rock_equivalent(total_inflow_base, 
                                                         include_rest_bool = False).sum('Type')
            inflow_eff = sand_gravel_crushed_rock_equivalent(total_inflow_eff, 
                                                             include_rest_bool=False).sum('Type')
            inflow = inflow.assign_coords(material='sand')
            inflow_eff = inflow_eff.assign_coords(material='sand')
            # overwrite material name
            inflow = inflow.assign_coords(material=material)
            inflow_eff = inflow_eff.assign_coords(material=material)
        elif material == 'sand':
            # replace with sand_gravel_crushed_rock_equivalent
            inflow = sand_gravel_crushed_rock_equivalent(total_inflow_base, 
                                                         include_rest_bool = False).sum('Type')
            inflow_eff = sand_gravel_crushed_rock_equivalent(total_inflow_eff, 
                                                             include_rest_bool=False).sum('Type')
            inflow = inflow.assign_coords(material='sand')
            inflow_eff = inflow_eff.assign_coords(material='sand')
        elif material == 'cement':
            inflow = calculate_cement_equivalent(total_inflow_base, 
                                                 include_rest_of=False).sum('Type')
            inflow_eff = calculate_cement_equivalent(total_inflow_eff, 
                                                    include_rest_of=False).sum('Type')
        else:
            inflow = total_inflow_base.sel(material=material).sum('Type')
            inflow_eff = total_inflow_eff.sel(material=material).sum('Type')

        # Material intensities per GDP
        material_intensity_base = inflow / gdp
        material_intensity_eff = inflow_eff / gdp

        # calculate the percentual resource efficiency
        resource_eff_procentual = (material_intensity_base - material_intensity_eff) / material_intensity_base * 100
        # calculate the reduction per year
        resource_eff_procentual_year = resource_eff_procentual.diff(dim='time').fillna(0)

        # save all materials from material_intensity_base in a new joint xarray under the material dimension
        if material == 'aluminium':
            # start to save in new xarray
            all_materials_resource_eff_procentual = resource_eff_procentual
            all_materials_resource_eff_per_year = resource_eff_procentual_year
        else:
            all_materials_resource_eff_procentual = xr.concat([all_materials_resource_eff_procentual, resource_eff_procentual], dim='material')
            all_materials_resource_eff_per_year = xr.concat([all_materials_resource_eff_per_year, resource_eff_procentual_year], dim='material')

    # manipuate gompertz coefs
    a0 = gompertz_original.sel(coef='a')
    # per-year multiplier (choose sign correctly: reductions -> 1 - r_frac)
    per_year_multiplier = 1 - (all_materials_resource_eff_procentual / 100.0) 

    # rename coordinate per_year_multiplier from time to Time
    per_year_multiplier = per_year_multiplier.rename({'time':'Time'})

    a0_expanded = a0.copy()
    a_t = (a0_expanded * per_year_multiplier)
    gompertz_eff = gompertz_original.copy()

    # apply a 20-year rolling mean to smoothen & extend to edges
    a_t_rolling_20 = a_t.rolling(Time=20, center=False, min_periods=1).mean()

    gompertz_eff.loc[dict(coef='a')] = a_t_rolling_20
    # save new gompertz coefs
    gompertz_eff.to_netcdf('../data/raw/rest-of/gompertz_values/coefs_gompertz_eff.nc')