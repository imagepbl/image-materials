"""Weight related preprocessing functions"""
import logging
import xarray as xr
import pandas as pd

from imagematerials.vehicles.constants import (
    TONNES_TO_KGS,
    years_range
)
from imagematerials.vehicles.modelling_functions import interpolate
from imagematerials.vehicles.preprocessing.util import (
    get_ship_capacity,
    xarray_conversion
)
from imagematerials.vehicles.modelling_functions import (
    scenario_change
)


def get_weights(data_path: str, general_data_path: str, circular_economy_config: dict):
    """Get vehicle weights from CSV and squeeze it in the right format.

    Parameters
    ----------
    data_path
        Path to the directory containing scenario data
    general_data_path
        Path to the directory containing general data
    circular_economy_config
        Dictionary with config for circular economy scenario

    Returns
    -------
        A xarray DataArray with vehicle lifetimes

    Notes
    -----
    TODO: immediately read into xarray, bypass pandas
    """
    # Weight of a single vehicle of each type in kg (simple and typical)
    vehicle_weight_kg_simple: pd.DataFrame = pd.read_csv(
        data_path.joinpath("vehicle_weight_kg_simple.csv"),
        index_col=0
    )
    vehicle_weight_kg_typical: pd.DataFrame = pd.read_csv(
        data_path.joinpath("vehicle_weight_kg_typical.csv"),
        index_col=[0, 1]
    )

    # complete & interpolate the vehicle weight data
    vehicle_weights_simple = interpolate(pd.DataFrame(vehicle_weight_kg_simple))

    vehicle_weights_typical = \
        vehicle_weight_kg_typical.rename_axis('mode', axis=1).stack().unstack(['mode', 'type'])
    vehicle_weights_typical = interpolate(pd.DataFrame(vehicle_weights_typical))

    # Apply lightweighting if part of scenario
    ce_scen = None  # INITIALIZE ce_scen

    if "narrow" in circular_economy_config.keys():
        ce_scen = "narrow"
    if "narrow_product" in circular_economy_config.keys():
        ce_scen = "narrow_product"
    if "resource_efficient" in circular_economy_config.keys():
        ce_scen = "resource_efficient"

    if ce_scen in ["resource_efficient", "narrow_product", "narrow"]:
                # Verify both are defined, otherwise raise error
        if not ('weight_change_pc' in circular_economy_config[ce_scen]['vehicles'].get('road', {}) and \
                'weight_change_pc' in circular_economy_config[ce_scen]['vehicles'].get('non-road', {})):
            raise ValueError(f"Both 'road' and 'non-road' weight_change_pc must be defined in '{ce_scen}' scenario")
        
        config = circular_economy_config[ce_scen]['vehicles']
        target_year = config['target_year']
        base_year = config['base_year']
        non_road_weight_change_pc = config['non-road']['weight_change_pc']
        road_weight_change_pc = config['road']['weight_change_pc']
        implementation_rate = config['implementation_rate']

        vehicle_weights_simple = scenario_change(
            vehicle_weights_simple, base_year, target_year, 
            non_road_weight_change_pc, implementation_rate)
        
        if isinstance(vehicle_weights_typical.columns, pd.MultiIndex):
            weight_change_pc_expanded = {}
            for mode, pct in road_weight_change_pc.items():
                # all (mode, drivetrain) columns
                subcols = [c for c in vehicle_weights_typical.columns if c[0] == mode]
                for c in subcols:
                    weight_change_pc_expanded[c] = pct
        else:
            weight_change_pc_expanded = road_weight_change_pc

        vehicle_weights_typical = scenario_change(
            vehicle_weights_typical, base_year, target_year,
            weight_change_pc_expanded, implementation_rate
        )
        logging.debug(f"implemented '{ce_scen}' for Vehicles (lightweighting)")

    vehicle_weights_simple = xarray_conversion(vehicle_weights_simple, (["Cohort"], ["Type"],))
    vehicle_weights_typical = xarray_conversion(vehicle_weights_typical, (["Cohort"], ["Type", "SubType"], {"Type": ["Type", "SubType"]}))
    vehicle_weights = xr.concat((vehicle_weights_simple, vehicle_weights_typical), dim="Type")

    ship_weights = _get_ship_weights(general_data_path)
    return xr.concat((vehicle_weights, ship_weights), dim="Type")


def _get_ship_weights(general_data_path: str):
    # weight of boats as a percentage of the capacity (%) fixed Data is
    # based on Ecoinvent report 14 on Transport (section 8.4.1)
    weight_boats: pd.DataFrame = pd.read_csv(
        general_data_path.joinpath("ships", "weight_percofcap_boats.csv"),
        index_col="t"
    ).sort_index(axis=0)

    weight_frac_boats_yrs = interpolate(weight_boats, change='no')

    # capacity of boats is in tonnes, the weight - expressed as a
    # fraction of the capacity - calculated in in kgs here
    cap_of_boats_yrs = get_ship_capacity(general_data_path)
    ship_weights = weight_frac_boats_yrs * cap_of_boats_yrs * TONNES_TO_KGS

    ship_weights = xarray_conversion(ship_weights, (["Cohort"], ["Type"],))

    # Fix coordinates
    ship_weights.coords["Type"] = [f"{x} Ships" for x in ship_weights.coords["Type"].values]
    return ship_weights
