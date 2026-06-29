#%%
"""Battery related preprocessing functions"""
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from importlib.resources import files

import prism

from imagematerials.electricity.utils import (
    logistic, 
    quadratic,
    interpolate_xr,
)
from imagematerials.concepts import create_electricity_graph, create_region_graph, create_vehicle_graph
from imagematerials.constants import IMAGE_REGIONS
from imagematerials.electricity.constants import (
    STANDARD_SCEN_EXTERNAL_DATA,
    YEAR_FIRST_GRID,
    SENS_ANALYSIS,
)
from imagematerials.vehicles.constants import (
    EV_BATTERY_TYPES,
)

def get_preprocessing_data_evbattery(
        path_base: str,
        climate_policy_config: dict,
        circular_economy_config: dict
    ):
    """ Load, process, and transform input data required for the electric vehicle (EV) battery module.

    This function reads external datasets related to battery technologies, including costs,
    material compositions, energy densities, and vehicle battery characteristics. It converts
    these datasets into xarray DataArrays with consistent dimensions and units, performs
    interpolations over time, and computes derived quantities such as technology market shares
    and vehicle-to-grid (V2G) participation parameters.

    The processed data are returned in a standardized dictionary format for use in the EV battery module.

    Parameters
    ----------
    path_base : str
        Base directory containing input data folders for electricity and vehicle datasets (data/raw/).
    climate_policy_config : dict
        Configuration dictionary specifying climate policy assumptions (currently not used directly).
    circular_economy_config : dict
        Configuration dictionary specifying circular economy assumptions (currently not used directly).

    Returns
    -------
    dict
        Dictionary containing preprocessed data as xarray DataArrays and auxiliary objects:
        - "shares": Market shares of EV battery technologies by vehicle type and time.
        - "weights": Battery weights per vehicle type and cohort.
        - "material_fractions": Material composition of EV batteries.
        - "energy_density": Energy density (kg/kWh) of EV battery technologies.
        - "vhc_fraction_v2g": Fraction of vehicles available for vehicle-to-grid (V2G).
        - "capacity_fraction_v2g": Fraction of battery capacity usable for V2G.
        - "knowledge_graph_elc": Electricity system knowledge graph.
        - "knowledge_graph_vhc": Vehicle system knowledge graph.

    Notes
    -----
    - All numerical data are converted to xarray DataArrays and assigned physical units using `prism`.
    - Market shares of battery technologies are computed using a multinomial logit model based on
      cost developments.
    - Vehicle types not associated with electric drivetrains are explicitly assigned zero battery shares.

    Dependencies
    ------------
    Requires external helper functions and global variables, including:
    - `calculate_storage_market_shares`
    - `interpolate_xr`
    - `normalize_selected_techs`
    - `create_vehicle_graph`, `create_electricity_graph`
    - `logistic`
    - `EV_BATTERY_TYPES`, `IMAGE_REGIONS`, `YEAR_FIRST_GRID`, `SENS_ANALYSIS`

    """
    # path_image_output = Path(path_base, "data", "raw", "image", scen_folder, "EnergyServices")
    path_external_data_standard_elc = Path(path_base, "electricity", "standard_data")
    path_external_data_standard_vhc = Path(path_base, "vehicles", "standard_data")

    units_file = files("imagematerials") / "units.txt"
    prism.unit_registry.load_definitions(units_file)

    year_start = YEAR_FIRST_GRID
    year_end = 2100

    ###########################################################################################################
    # Read in files #

    # 1. External Data ======================================================================================== 

    # material intensities in kg/kWh
    material_intensities = pd.read_csv(path_external_data_standard_elc / 'storage_and_ev_battery_material_intensities.csv', 
                                        usecols=["Time", "sub_technology", "material", "value"])

    # market shares in % of total storage capacity (excluding pumped hydropower storage; values: 0-1)
    market_shares = pd.read_csv(path_external_data_standard_vhc / 'ev_battery_market_shares.csv', 
                                usecols=["Time", *EV_BATTERY_TYPES])

    # energy density (kWh/kg)
    energy_density = pd.read_csv(path_external_data_standard_elc / "energy_density_kwh_per_kg.csv",
                                usecols=["Time", "sub_technology", "value"])

    # Using the 250 Wh/kg on the kWh of the various batteries a weight (in kg) of the battery per vehicle category is determined
    # TODO: where is this data from? kWh per battery type?
    battery_weights_data = pd.read_csv(path_external_data_standard_vhc / "ev_battery_weights_kg.csv", index_col=[0,1])

    # usable capacity of EV batteries for V2G applications (relative: fraction of the total battery capacity that can be used for V2G)
    if SENS_ANALYSIS == 'high_stor':
    # pessimistic sensitivity variant (meaning more additional storage is needed) -> smaller fraction of the EV capacities is usable as storage compared to the normal case
    #    capacity_usable_PHEV = 0.025   # 2.5% of capacity of PHEV is usable as storage (in the pessimistic sensitivity variant)
    #    capacity_usable_BEV  = 0.05    # 5  % of capacity of BEVs is usable as storage (in the pessimistic sensitivity variant)
        ev_capacity_fraction_v2g = pd.read_csv(path_external_data_standard_elc / "ev_battery_capacity_usable_for_v2g_variant_high_storage.csv")
    else:
        ev_capacity_fraction_v2g = pd.read_csv(path_external_data_standard_elc / "ev_battery_capacity_usable_for_v2g.csv")

    # fraction of EVs available for V2G (considering that not all EVs are capable of bi-directional loading, economic incentives are still missing, and not all owners are willing to provide V2G services)
    ev_fraction_v2g_data = pd.read_csv(path_external_data_standard_elc / "ev_fraction_available_for_v2g.csv", index_col=[0])


    ###########################################################################################################
    # Transform to xarray #

    # create list of all vehicle types (combinations of vehicles type (Cars, Medium Freight Trucks,...) and drive trains (ICE, BEV,...))
    # vehicle_list = [f"{super_type} - {sub_type}" for super_type, sub_type in product(typical_modes, drive_trains)]
    vehicle_list = ['Cars - BEV', 'Cars - FCV', 'Cars - HEV', 'Cars - ICE', 'Cars - PHEV',
        'Cars - Trolley', 'Heavy Freight Trucks - BEV',
        'Heavy Freight Trucks - FCV', 'Heavy Freight Trucks - HEV',
        'Heavy Freight Trucks - ICE', 'Heavy Freight Trucks - PHEV',
        'Heavy Freight Trucks - Trolley', 'Light Commercial Vehicles - BEV',
        'Light Commercial Vehicles - FCV', 'Light Commercial Vehicles - HEV',
        'Light Commercial Vehicles - ICE', 'Light Commercial Vehicles - PHEV',
        'Light Commercial Vehicles - Trolley', 'Medium Freight Trucks - BEV',
        'Medium Freight Trucks - FCV', 'Medium Freight Trucks - HEV',
        'Medium Freight Trucks - ICE', 'Medium Freight Trucks - PHEV',
        'Medium Freight Trucks - Trolley', 'Midi Buses - BEV',
        'Midi Buses - FCV', 'Midi Buses - HEV', 'Midi Buses - ICE',
        'Midi Buses - PHEV', 'Midi Buses - Trolley', 'Regular Buses - BEV',
        'Regular Buses - FCV', 'Regular Buses - HEV', 'Regular Buses - ICE',
        'Regular Buses - PHEV', 'Regular Buses - Trolley']
    vehicle_list_non_ev = ['Cars - ICE', 'Cars - FCV', 'Cars - Trolley', 
        'Regular Buses - ICE', 'Regular Buses - FCV', 
        'Regular Buses - Trolley', 'Midi Buses - ICE',
        'Midi Buses - FCV', 'Midi Buses - Trolley', 'Heavy Freight Trucks - ICE',
        'Heavy Freight Trucks - FCV',
        'Heavy Freight Trucks - Trolley', 'Medium Freight Trucks - ICE',
        'Medium Freight Trucks - FCV',
        'Medium Freight Trucks - Trolley', 'Light Commercial Vehicles - ICE',
        'Light Commercial Vehicles - FCV',
        'Light Commercial Vehicles - Trolley']
    vhc_knowledge_graph = create_vehicle_graph()

    # 1. Market Shares -----------------------------------------------------------------------------

    market_shares = market_shares.melt(
        id_vars="Time",
        var_name="Type",
        value_name="market_shares"
    )
    market_shares_da = (
        market_shares.set_index(["Time", "Type"])["market_shares"]
        .to_xarray()
        .rename({"Time": "Cohort", "Type": "BatteryType"})
    )
    market_shares_da = prism.Q_(market_shares_da, "dimensionless")
    market_shares_da.name = "EVBatteryMarketShares"

    # 2. battery weigths -----------------------------------------------------------------------

    battery_weights_da = (
        battery_weights_data
        .rename_axis(index={"time": "Cohort", "type": "Drivetrain"})
        .to_xarray()                        # Convert the pandas DataFrame to xarray with dims: Cohort, Drivetrain, Vehicle
        .to_array("Vehicle")                # Move the DataFrame columns into an explicit xarray dimension
        .rename("BatteryWeights")
    )
    # Combine Vehicle and Drivetrain into a single dimension
    battery_weights_da = battery_weights_da.stack(Type=("Vehicle", "Drivetrain"))
    # combine Type string labels: ('Cars', 'BEV') -> 'Cars - BEV'
    new_type = [f"{v} - {d}" for v, d in battery_weights_da.indexes["Type"]]

    battery_weights_da = (
        battery_weights_da
        .reset_index("Type", drop=True)
        .assign_coords(Type=new_type) # Replace the stacked MultiIndex with the combined string labels
    )
    battery_weights_da = prism.Q_(battery_weights_da, "kg")
    # rebroadcast so that the Type coordinates have the correct sequence
    battery_weights_da = vhc_knowledge_graph.rebroadcast_xarray(battery_weights_da, output_coords=vehicle_list, dim="Type")


    # 3. material intensities -----------------------------------------------------------------------

    material_intensities_da = (
        material_intensities.set_index(["Time", "sub_technology", "material"])["value"]
        .to_xarray()
        .rename({"sub_technology": "BatteryType", "Time": "Cohort"})
    )
    material_intensities_da = prism.Q_(material_intensities_da, "kg/kWh")
    material_intensities_da.name = "EVBatteryMaterialIntensities"
    # Ensure correct dimension order
    material_intensities_da = material_intensities_da.transpose('material', 'Cohort', 'BatteryType')
    material_intensities_da = material_intensities_da.sel(BatteryType=EV_BATTERY_TYPES)


    # 4. energy density -----------------------------------------------------------------------------

    energy_density_da = (
        energy_density.set_index(["Time", "sub_technology"])["value"]
        .to_xarray()
        .rename({"sub_technology": "BatteryType", "Time": "Cohort"})
    )
    energy_density_da = prism.Q_(energy_density_da, "kWh/kg")
    energy_density_da.name = "EnergyDensity"
    energy_density_da = energy_density_da.sel(BatteryType=EV_BATTERY_TYPES) # Select only those technologies from the storage technologies that are suitable for EV & mobile applications


    # 5. EV fraction available for V2G --------------------------------------------------------------------
    # For this variable first the interpolations are done and then the conversion to xarray DataArray
    x = ev_fraction_v2g_data.reindex(range(ev_fraction_v2g_data.index[0],ev_fraction_v2g_data.index[-1]+1)).interpolate(method="linear")
    y = logistic(x, L=x.iloc[-1].values)
    # y = quadratic(x)
    ev_fraction_v2g = ev_fraction_v2g_data.reindex(range(year_start,year_end+1)).interpolate(method="linear") # create dataframe with full index; values before first data points will be Nans, between data points interpolated linearly, after last data point will be last known value
    ev_fraction_v2g.loc[:ev_fraction_v2g_data.index[0]] = 0 # set values before first data point to 0
    ev_fraction_v2g.loc[ev_fraction_v2g_data.index[0]:ev_fraction_v2g_data.index[1]] = y # set values between (originally) first and last data point to quadratic/logistic interpolation
    # Build xarray DataArray
    years = ev_fraction_v2g.index.astype(int).rename(None)
    techs = ev_fraction_v2g.columns
    data_array = ev_fraction_v2g.to_numpy()    # shape (Time, Type)
    data_array = data_array[:, :, np.newaxis]   # shape (Time, Type, 1)
    data_array = np.broadcast_to(data_array, data_array.shape[:2] + (len(IMAGE_REGIONS),)) # (Time, Type, Region) - add region dimension, though all regions have the same values for now
    vhc_fraction_v2g_da = xr.DataArray(
        data_array,
        dims=("Time", "Type", "Region"),
        coords={
            "Time": years,
            "Type": techs,
            "Region": IMAGE_REGIONS
        },
        name="VehicleFractionV2G"
    )
    vhc_fraction_v2g_da = prism.Q_(vhc_fraction_v2g_da, "fraction")
    
    # 6. capacity used for V2G -----------------------------------------------------------------------------
    # Note: xr_capacity_fraction_v2g must have same Type coords than xr_vhc_fraction_v2g (otherwise ElectricVehicleBatteries model will crash)
    # Build xarray DataArray
    techs = ev_capacity_fraction_v2g.columns
    data_array = ev_capacity_fraction_v2g.to_numpy().ravel()  # flatten to 1D
    capacity_fraction_v2g_da = xr.DataArray(
        data_array,
        dims=("Type"),
        coords={
            "Type": techs
        },
        name="CapacityFractionV2g"
    )
    capacity_fraction_v2g_da = prism.Q_(capacity_fraction_v2g_da, "fraction")


    ###########################################################################################################
    # Interpolations #

    # fix the market share of storage technologies before year_start and after 2050
    market_shares_da_interp = interpolate_xr(market_shares_da, year_start, year_end)
    market_shares_da_interp = market_shares_da_interp.expand_dims(Type=vehicle_list).copy() # add vehicle type dimension
    market_shares_da_interp.loc[dict(Type=vehicle_list_non_ev)] = 0 # set non-EV vehicle types to zero

    # 2. Battery Weights ----------------------------------------------------------------
    battery_weights_da_interp = interpolate_xr(battery_weights_da, year_start, year_end)

    # 3. Material Intensities ----------------------------------------------------------------
    material_intensities_da_interp = interpolate_xr(material_intensities_da, year_start, year_end)

    #  4. Energy Density ----------------------------------------------------------------
    energy_density_da_interp = interpolate_xr(energy_density_da, year_start, year_end)


    ###########################################################################################################
    # Prep_data File #

    # bring preprocessing data into a generic format for the model
    prep_data = {}
    prep_data["shares"] = market_shares_da_interp
    prep_data["weights"] = battery_weights_da_interp
    prep_data["material_intensities"] = material_intensities_da_interp
    prep_data["energy_density"] = energy_density_da_interp
    prep_data["vhc_fraction_v2g"] = vhc_fraction_v2g_da
    prep_data["capacity_fraction_v2g"] = capacity_fraction_v2g_da
    prep_data["knowledge_graph_elc"] = create_electricity_graph()
    prep_data["knowledge_graph_vhc"] = create_vehicle_graph()

    return prep_data