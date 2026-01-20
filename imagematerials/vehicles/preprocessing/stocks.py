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
    tkms_to_nr_of_vehicles_fixed
)
from imagematerials.vehicles.preprocessing.shares import get_vehicle_shares
from imagematerials.vehicles.preprocessing.util import (
    get_passengerkms,
    get_ship_capacity,
    get_tonkms,
    xarray_conversion
)


def get_vehicle_stocks(data_path: str, standard_data_path, climate_data_path, climate_policy_config,
                       circular_economy_config, knowledge_graph_vehicle):
    shares = get_vehicle_shares(climate_data_path, climate_policy_config,
                                knowledge_graph_vehicle)
    tonkms_Mtkms = get_tonkms(climate_data_path, climate_policy_config)
    passengerkms_Tpkms = get_passengerkms(climate_data_path, climate_policy_config)
    # TODO: add description again here!
    load: pd.DataFrame = pd.read_csv(
        standard_data_path.joinpath("load_pass_and_tonnes.csv")
    )
    # Percentage of the maximum load that is on average
    loadfactor: pd.DataFrame = pd.read_csv(
        standard_data_path.joinpath("loadfactor_percentages.csv")
    )

    # Km/year of all the vehicles (buses & cars have region-specific files)
    mileages: pd.DataFrame = pd.read_csv(
        data_path.joinpath("kilometrage_per_year.csv"),
        index_col="t"
    )

    # first year of operation per vehicle-type
    # 1807 was originally in the dataframe
    first_year_vehicle: pd.DataFrame = pd.read_csv(
        standard_data_path.joinpath("first_year_vehicle.csv")
    )

    # Percentage of tonne-/passengerkilometres
    market_share: pd.DataFrame = pd.read_csv(
        standard_data_path.joinpath("fraction_tkm_pkm.csv")
    )

    # Adjust & interpolate mileages (same operations as for kilometrage
    # for cars and buses.
    mileages = mileages.reindex(
        years_range
    ).interpolate(limit_direction='both')

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

    # TODO
    # TODO: exchange for proper region labels defined elsewhere
    # TODO: look at vehicle type names
    total_nr_vehicles_simple = pd.DataFrame(
        index=years_range,
        columns=pd.MultiIndex.from_product([all_modes,
                                            list(range(1, REGIONS + 3))],
                                           names=["Type", "Region"]
                                           )
    )
    total_nr_vehicles_simple.index.name = "time"

    # Fill the vehicle types that need no conversion    
    for label in ["Passenger Planes", "Trains", "High Speed Trains", "Bikes" ]:
        total_nr_vehicles_simple[label] = tkms_to_nr_of_vehicles_fixed(
            passengerkms_Tpkms[label].unstack(),
            mileages[label],
            load[label].values[0],
            loadfactor[label].values[0]
        )

    # Trucks are calculated differently because the IMAGE model does not account for LCV trucks, which concerns a large
    # portion of the material requirements of road freight
    # the total trucks Tkms remain the same, but a LCV fraction is substracted, and the remainder is re-assigned to
    # medium and heavy trucks according to their original ratio
    trucks_total_tkm = tonkms_Mtkms["Medium Freight Trucks"].unstack() + tonkms_Mtkms["Heavy Freight Trucks"].unstack()
    trucks_LCV_tkm = trucks_total_tkm * LIGHT_COMMERCIAL_VEHICLE_SHARE
    MFT_percshare_tkm = tonkms_Mtkms["Medium Freight Trucks"].unstack() / trucks_total_tkm  # the MFT fraction of the total
    HFT_percshare_tkm = tonkms_Mtkms["Heavy Freight Trucks"].unstack() / trucks_total_tkm   # the HFT fraction of the total
    trucks_min_LCV = trucks_total_tkm - trucks_LCV_tkm
    trucks_MFT_tkm = trucks_min_LCV.mul(MFT_percshare_tkm)             # Used in loop below
    trucks_HFT_tkm = trucks_min_LCV.mul(HFT_percshare_tkm)

    # demand for freight planes is reduced by 50% because about half of
    # the air freight is transported as cargo on passenger planes
    # TODO: this is clearly off: value of market_share["Passenger Planes"].values[0] == 1
    # fix for now hardcoded - fix when rewriting this file
    # air_freight_tkms = tonkms_Mtkms["Freight Planes"].unstack() * market_share["Passenger Planes"].values[0]
    air_freight_tkms = tonkms_Mtkms["Freight Planes"].unstack() * 0.5

    # Fill the vehicle types that need conversion from Mega-ton-kms
    # original ton kilometers are in Mega-ton-kms, div by MEGA_TO_TERA to
    # harmonize with pkms which are in Tera pkms
    # Define the input data for vehicles requiring conversion
    vehicle_data = {
        "Heavy Freight Trucks": trucks_HFT_tkm,
        "Medium Freight Trucks": trucks_MFT_tkm,
        "Light Commercial Vehicles": trucks_LCV_tkm,
        "Freight Planes": air_freight_tkms,
        "Freight Trains": tonkms_Mtkms['Freight Trains'].unstack(),
        "Inland Ships": tonkms_Mtkms['Inland Ships'].unstack()
    }

    # Handle the vehicle types that need conversion from Mega km to Tera km
    for key, df in vehicle_data.items():
        total_nr_vehicles_simple[key] = tkms_to_nr_of_vehicles_fixed(
            df / MEGA_TO_TERA,
            mileages[key],
            load[key].values[0],
            loadfactor[key].values[0]
        )

    # Adding cars and buses
    total_nr_vehicles_simple = _calculate_buses_cars_stocks(
        total_nr_vehicles_simple, data_path, climate_data_path, climate_policy_config,
        circular_economy_config, load, loadfactor, market_share, passengerkms_Tpkms
    )

    # Adding ships
    total_nr_vehicles_simple = \
        _calculate_ships_stocks(total_nr_vehicles_simple, standard_data_path, tonkms_Mtkms)

    # %% Interpolate to complete data for the entire model period (including a historic tail to set up the dynamic
    # stock calculations)

    # reformatting lifetime data (because input is not yet region-specific)
    idx = pd.IndexSlice  # needed for slicing multi-index
    first_year_vehicle_regionalized = pd.DataFrame(
        0, index=first_year_vehicle.index, columns=total_nr_vehicles_simple.columns
    )
    for vehicle in list(total_nr_vehicles_simple.columns.levels[0]):
        first_year_vehicle_regionalized.loc[:, idx[vehicle, :]] = \
            [first_year_vehicle[vehicle].values[0] for region in
             list(total_nr_vehicles_simple.columns.levels[1])]

    # Doing the interpolation & assigning a starting point for the vehicle
    # stock time series based on first_year_vehicle
    total_nr_vehicles_simple = interpolate(total_nr_vehicles_simple,
                                           first_year_vehicle_regionalized,
                                           change='no')

    stocks = xarray_conversion(total_nr_vehicles_simple, (["Time"], ["Type", "Region"],))

    region_coords = np.sort(stocks.coords["Region"].values.astype(int)).astype(str)[:-2]
    share_coords = set()
    for cur_type in shares.Type.values:
        share_coords.add(cur_type.split(" - ")[0])
    output_coords_type = [x for x in stocks.Type.values if x not in share_coords] + \
        list(shares.coords["Type"].values)

    # Use the shares to get the stocks for each of the subtypes.
    stocks = knowledge_graph_vehicle.rebroadcast_xarray(stocks,
                                                        output_coords=region_coords,
                                                        dim="Region")
    stocks = knowledge_graph_vehicle.rebroadcast_xarray(stocks,
                                                        output_coords_type,
                                                        dim="Type",
                                                        shares=shares)
    stocks = prism.Q_(stocks, "count")

    knowledge_graph_region = create_region_graph()
    return knowledge_graph_region.rebroadcast_xarray(stocks, output_coords=IMAGE_REGIONS,
                                                     dim="Region")

def _calculate_ships_stocks(total_nr_vehicles_simple, standard_data_path, tonkms_Mtkms):
    cap_of_boats_yrs = get_ship_capacity(standard_data_path)

    # number of boats in the global merchant fleet (2005-2018) changing
    # Data by EQUASIS
    nr_of_boats: pd.DataFrame = pd.read_csv(
        standard_data_path.joinpath("ships", "number_of_boats.csv"),
        index_col="t"
    ).sort_index(axis=0)

    # loadfactor of boats (fraction) fixed Data is based on Ecoinvent
    # report 14 on Transport (Table 8-19)
    loadfactor_boats: pd.DataFrame = pd.read_csv(
        standard_data_path.joinpath("ships", "loadfactor_boats.csv"),
        index_col="t"
    ).sort_index(axis=0)

    # mileage of boats in km/yr (per ship) fixed Data is based on
    # Ecoinvent report 14 on Transport (Table 8-19)
    mileage_boats: pd.DataFrame = pd.read_csv(
        standard_data_path.joinpath("ships", "mileage_kmyr_boats.csv"),
        index_col="t"
    ).sort_index(axis=0)

    # For INTERNATIONAL SHIPPING the number of vehicles is calculated
    # differently

    # pre-calculate the shares of the boats based on the number of boats,
    # before adding history/future
    # TODO: boats vs. ships?
    share_of_boats = nr_of_boats.div(nr_of_boats.sum(axis=1), axis=0)

    # does change based on trend in original data
    share_of_boats_yrs = interpolate(share_of_boats, change='yes')
    # could be 'yes' based on better data
    loadfactor_boats_yrs = interpolate(loadfactor_boats, change='no')
    mileage_boats_yrs = interpolate(mileage_boats, change='no')

    # normalize the share of boats to 1 & adjust the capacity & mileage for
    # smaller ships
    share_of_boats_yrs = share_of_boats_yrs.div(share_of_boats_yrs.sum(axis=1), axis=0)
    mileage_boats_yrs = mileage_boats_yrs.mul(mile_adjustment, axis=1)

    # now derive the number of ships for 4 ship types in four steps:
    # 1) get the share of the ship types in the Tkms shipped (what % of total
    # tkms shipped goes by what ship type?)
    share_of_boats_tkm_all = share_of_boats_yrs * cap_of_boats_yrs  * loadfactor_boats_yrs * mileage_boats_yrs
    share_of_boats_tkm = share_of_boats_tkm_all.div(share_of_boats_tkm_all.sum(axis=1), axis=0)

    # for comparison, we find the difference of the known and the calculated nr of ships (global total)
    # in the period 2005-2018
    # TODO: is this used anywhere?
    diff_ships = pd.DataFrame().reindex_like(nr_of_boats)

    # TODO: it seems this calculation should be able to be simplified
    for size in ["Small", "Medium", "Large", "Very Large"]:
        total_nr_vehicles_simple[f"{size} Ships"] = tonkms_Mtkms["international shipping"].unstack().mul(
            share_of_boats_tkm[size].loc[START_YEAR:], axis=0).mul(MEGA_TO_TERA).div(
                cap_of_boats_yrs[size].loc[START_YEAR:], axis=0).div(
        mileage_boats_yrs[size].loc[START_YEAR:], axis=0)
        diff_ships[size] = total_nr_vehicles_simple[f"{size} Ships"].loc[SHIPS_YEARS_RANGE,   # TODO Make years flexible
            28].div(nr_of_boats[size])

    return total_nr_vehicles_simple


def _calculate_buses_cars_stocks(total_nr_vehicles_simple, data_path, climate_data_path,
                                 climate_policy_config, circular_economy_config, load, loadfactor,
                                 market_share, passengerkms_Tpkms):
    # Average End-of-Life of vehicles in years, this file also contains
    # the setting for the choice of distribution and other lifetime
    # related settings (standard devition, or alternative
    # parameterisation)
    # kilometrage of passenger cars in kms/yr
    kilometrage: pd.DataFrame = pd.read_csv(
        data_path.joinpath("kilometrage.csv"),
        index_col="t"
    )
    # kilometrage of midi-buses in kms/yr
    kilometrage_midi_bus: pd.DataFrame = pd.read_csv(
        data_path.joinpath("kilometrage_midi.csv"),
        index_col="t"
    )
    # kilometrage of regular buses in kms/yr
    kilometrage_bus: pd.DataFrame = pd.read_csv(
        data_path.joinpath("kilometrage_bus.csv"),
        index_col="t"
    )

    # The loadfactor of passenger vehicles (occupation in nr of people/
    # vehicle) in reference to the base loadfactor (see constants above)
    loadfactor_car_data: pd.DataFrame = read_mym_df(climate_data_path.joinpath(
        climate_policy_config['data_files']['transport']['passenger']['load']
    )).rename(columns={"DIM_1": "region"})

    # preprocessing of the IMAGE files & files with additional
    # assumptions on vehcile materials (renaming, removing 27th region,
    # adding labels etc.)

    # Interpolate kilometrage & mileage data (over the range of original
    # IMAGE data years, so the generic interpolation function is not used
    # here as that would extend the historic tail)
    kilometrage = kilometrage.reindex(
        years_range
    ).interpolate(limit_direction='both')
    kilometrage_bus = kilometrage_bus.reindex(
        years_range
    ).interpolate(limit_direction='both')
    kilometrage_midi_bus = kilometrage_midi_bus.reindex(
        years_range
    ).interpolate(limit_direction='both')

    # get a list with region names
    # TODO: turn this into a proper mapping based on ...
    region_list = list(kilometrage.columns.values)

    # select loadfactor for cars
    # loadfactor for cars (in persons per vehicle) * LOAD_FACTOR to
    # correct with te TIMER reference
    loadfactor_car_data.rename(columns={"DIM_1": "region"}, inplace=True)
    car_loadfactor = loadfactor_car_data[["time", "region", 5]].pivot_table(
        index="time", columns="region"
    ).droplevel(level=0, axis=1) * LOAD_FACTOR

    # To avoid car load (person/vehicle) values ever going below 1,
    # replace all values below 1 with 1
    car_loadfactor = car_loadfactor.apply(lambda x:
                                          [y if y >= 1 else 1 for y in x])
    # remove years beyond LAST_YEAR
    car_loadfactor = car_loadfactor.loc[list(range(START_YEAR, END_YEAR + 1)), :]
    car_loadfactor.columns = region_list

    # increase mileages\kilometrages
    if 'narrow' in circular_economy_config.keys():
        target_year = circular_economy_config['narrow']['vehicles']['target_year']
        base_year = circular_economy_config['narrow']['vehicles']['base_year']
        mileage_increase = circular_economy_config['narrow']['vehicles']['mileage']
        region_mileage = circular_economy_config['narrow']['vehicles']['region_mileage']
        implementation_rate = circular_economy_config['narrow']['vehicles']['implementation_rate']

        # Cars are saved seperatly since they are not defined in the
        # general kilometrage dataframe
        kilometrage = apply_change_per_region(
            kilometrage, base_year, target_year, 
            region_mileage['Cars'], implementation_rate
        )
        kilometrage_bus = apply_change_per_region(
            kilometrage_bus, base_year, target_year, 
            mileage_increase['Midi Buses'], implementation_rate
        )
        kilometrage_midi_bus = apply_change_per_region(
            kilometrage_midi_bus, base_year, target_year, 
            mileage_increase['Regular Buses'], implementation_rate
        )

        logging.debug("implemented 'narrow' for Vehicles (increase mileage)")

    # Buses are adjusted to account for the higher material intensity of
    # mini-buses. All in tera pkms.
    bus_regl_pkms = passengerkms_Tpkms["bus"].unstack() * market_share["Regular Buses"].values[0]
    bus_midi_pkms = passengerkms_Tpkms["bus"].unstack() * market_share["Midi Buses"].values[0]

    # Select tkms of passenger cars (which will be adjusted to represent
    # 5 types: ICE, HEV, PHEV, BEV & FCV)
    car_pkms = passengerkms_Tpkms["Cars"].unstack()
    # exclude region 27 & 28 (empty & global total), mind
    # that the columns represent generation technologies
    # in tera pkms
    car_pkms = car_pkms.drop([27, 28], axis=1)
    car_pkms.columns = region_list

    # passenger cars and buses are calculated separately (due to regional &
    # changing mileage & load), first the totals
    # now in kms
    car_total_vkms = car_pkms.div(car_loadfactor) * PKMS_TO_VKMS
    # total number of cars
    car_total_nr = car_total_vkms.div(kilometrage)
    car_total_nr[["Extra Column", "Extra Column 2"]] = 1
    # remove region labels (for use in functions later on)
    car_total_nr.columns = list(range(1, REGIONS + 3))
    total_nr_vehicles_simple["Cars"] = car_total_nr

    # for buses do the same, but first remove region 27 & 28 (empty & world total) & kilometrage column names
    # bus_regl_pkms  = bus_regl_pkms.drop([27, 28], axis=1)
    # TODO: remove/change this hack!
    kilometrage_bus[["Extra Column", "Extra Column 2"]] = 1
    kilometrage_bus.columns = list(range(1, REGIONS + 3))
    bus_regl_vkms = bus_regl_pkms.div(load["Regular Buses"].values[0]
                                      * loadfactor["Regular Buses"].values[0]) * PKMS_TO_VKMS
    # now in kms
    total_nr_vehicles_simple["Regular Buses"] = bus_regl_vkms.div(
        kilometrage_bus)

    # bus_midi_pkms  = bus_midi_pkms.drop([27, 28], axis=1)
    # TODO: remove/change this hack!
    kilometrage_midi_bus[["Extra Column", "Extra Column 2"]] = 1
    kilometrage_midi_bus.columns = list(range(1, REGIONS + 3))
    bus_midi_vkms = bus_midi_pkms.div(load["Midi Buses"].values[0] * loadfactor["Midi Buses"].values[0]) * PKMS_TO_VKMS
    # now in kms
    total_nr_vehicles_simple["Midi Buses"] = bus_midi_vkms.div(kilometrage_midi_bus)

    return total_nr_vehicles_simple
