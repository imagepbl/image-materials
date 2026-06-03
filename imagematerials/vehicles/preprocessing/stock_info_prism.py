import logging
import xarray as xr
import numpy as np
import pandas as pd
import prism


from imagematerials.concepts import create_region_graph
from imagematerials.constants import IMAGE_REGIONS
from imagematerials.read_mym import read_mym_df
from imagematerials.vehicles.constants import (
    END_YEAR,
    LIGHT_COMMERCIAL_VEHICLE_SHARE,
    LOAD_FACTOR,
    MEGA_TO_TERA,
    PKMS_TO_VKMS,
    REGIONS,
    SHIPS_YEARS_RANGE,
    START_YEAR,
    all_modes,
    mile_adjustment,
    years_range
)
from imagematerials.vehicles.modelling_functions import (
    apply_change_per_region,
    interpolate,
    scenario_change,
)
from imagematerials.vehicles.preprocessing.shares import get_vehicle_shares
from imagematerials.vehicles.preprocessing.util import (
    get_passengerkms,
    get_ship_capacity,
    get_tonkms,
    xarray_conversion
)

def get_vehicle_stock_info(data_path: str, standard_data_path,
                           circular_economy_config):

    load: pd.DataFrame = pd.read_csv(
        standard_data_path.joinpath("load_pass_and_tonnes.csv"))
    
    # Percentage of the maximum load that is on average
    loadfactor: pd.DataFrame = pd.read_csv(
        standard_data_path.joinpath("loadfactor_percentages.csv"))

    # Km/year of all the vehicles (buses & cars have region-specific files)
    mileages: pd.DataFrame = pd.read_csv(
        data_path.joinpath("kilometrage_per_year.csv"), index_col="t")

    # first year of operation per vehicle-type
    first_year_vehicle: pd.DataFrame = pd.read_csv(
        standard_data_path.joinpath("first_year_vehicle.csv"))

    # Percentage of tonne-/passengerkilometres
    market_share: pd.DataFrame = pd.read_csv(
        standard_data_path.joinpath("fraction_tkm_pkm.csv"))

    # Adjust & interpolate mileages (same operations as for kilometrage
    # for cars and buses.
    mileages = mileages.reindex(
        years_range).interpolate(limit_direction='both')
    

    kilometrage_cars: pd.DataFrame = pd.read_csv(
        data_path.joinpath("kilometrage.csv"),index_col="t")
    # kilometrage of midi-buses in kms/yr
    kilometrage_midi_bus: pd.DataFrame = pd.read_csv(
        data_path.joinpath("kilometrage_midi.csv"),index_col="t")
    # kilometrage of regular buses in kms/yr
    kilometrage_bus: pd.DataFrame = pd.read_csv(
        data_path.joinpath("kilometrage_bus.csv"),index_col="t")
    
    # Interpolate kilometrage & mileage data (over the range of original
    # IMAGE data years, so the generic interpolation function is not used
    # here as that would extend the historic tail)
    kilometrage_cars = kilometrage_cars.reindex(
        years_range).interpolate(limit_direction='both')
    kilometrage_bus = kilometrage_bus.reindex(
        years_range).interpolate(limit_direction='both')
    kilometrage_midi_bus = kilometrage_midi_bus.reindex(
        years_range).interpolate(limit_direction='both')
    
    # to xarry
    kilometrage_cars.columns = IMAGE_REGIONS
    kilometrage_cars_xr = xr.DataArray(
    kilometrage_cars.to_numpy(),
    dims=("Time", "Region"),
    coords={"Time": kilometrage_cars.index.to_numpy(), "Region": kilometrage_cars.columns.to_numpy()},
    name="kilometrage")

    ce_scen = None  # INITIALIZE ce_scen
    if "narrow_product" in circular_economy_config.keys():
        ce_scen = "narrow_product"

    if ce_scen == "narrow_product":
        target_year = circular_economy_config[ce_scen]['vehicles']['target_year']
        base_year = circular_economy_config[ce_scen]['vehicles']['base_year']
        mileage_increase = circular_economy_config[ce_scen]['vehicles']['mileage']
        implementation_rate = circular_economy_config[ce_scen]['vehicles']['implementation_rate']

        mileages = scenario_change(
            mileages, base_year, target_year, 
            mileage_increase, implementation_rate)
        
        logging.debug(f"implemented '{ce_scen}' for Vehicles (mileage/kilometrage increase)")

    # reformatting lifetime data (because input is not yet region-specific)

    first_year_vehicle_regionalized = pd.DataFrame(
        0, index=first_year_vehicle.index, columns=IMAGE_REGIONS
    )

    return (load, loadfactor, 
            first_year_vehicle_regionalized, 
            market_share, mileages, 
            kilometrage_cars_xr, kilometrage_midi_bus, 
            kilometrage_bus)

