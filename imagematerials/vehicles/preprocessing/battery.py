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
    normalize_selected_techs,
    calculate_storage_market_shares
)
from imagematerials.concepts import create_electricity_graph, create_region_graph, create_vehicle_graph
from imagematerials.constants import IMAGE_REGIONS
from imagematerials.electricity.constants import (
    STANDARD_SCEN_EXTERNAL_DATA,
    YEAR_FIRST_GRID,
    SENS_ANALYSIS,
    EV_BATTERIES,
)


def get_preprocessing_data_evbattery(
    path_base: str,
    climate_policy_config: dict,
    circular_economy_config: dict,
    scenario: str,
    year_start: int,
    year_end: int,
    year_out: int
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
    scenario : str
        Scenario identifier used for selecting or organizing input data (not explicitly used here).
    year_start : int
        First year for interpolation of market shares and other time-dependent variables.
    year_end : int
        Last year of the simulation horizon (currently not explicitly used in this function).
    year_out : int
        Output year.

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
    - `EV_BATTERIES`, `IMAGE_REGIONS`, `YEAR_FIRST_GRID`, `SENS_ANALYSIS`

    """
    
    # path_image_output = Path(path_base, "data", "raw", "image", scen_folder, "EnergyServices")
    path_external_data_standard_elc = Path(path_base, "electricity", "standard_data")
    path_external_data_standard_vhc = Path(path_base, "vehicles", "standard_data")

    units_file = files("imagematerials") / "units.txt"
    prism.unit_registry.load_definitions(units_file)
    # prism.unit_registry.load_definitions(path_base / "imagematerials" / "units.txt")


    ###########################################################################################################
    # Read in files #
    
    # 1. External Data ======================================================================================== 

    # storage costs according to IRENA storage report & other sources in the SI ($ct/kWh electricity cycled)
    storage_costs = pd.read_csv(path_external_data_standard_elc / "storage_cost.csv", index_col=0).transpose()

    # assumed malus & bonus of storage costs (malus for advanced technologies, still under development; bonus for batteries currently used in EVs, we assume that a large volume of used EV batteries will be available and used for dedicated electricity storage, thus lowering costs), only the bonus remains by 2030
    costs_correction = pd.read_csv(path_external_data_standard_elc / "storage_malus.csv", index_col=0).transpose()

    # assumptions on the long-term price decline after 2050. the fraction of the annual growth rate (determined based on 2018-2030) that will be applied after 2030, ranging from 0.25 to 1 - 0.25 means the price decline is not expected to continue strongly, while 1 means that the same (2018-2030) annual price decline is also applied between 2030 and 2050)
    cost_decline_longterm_correction = pd.Series(pd.read_csv(path_external_data_standard_elc / "storage_ltdecline.csv",index_col=0,  header=None).transpose().iloc[0]) 

    # INVERS energy density (note that the unit is kg/kWh, the invers of energy density: storage capacity - mass required to store one unit of energy —> more mass per energy = worse performance)
    # TODO: read in directly IRENA data?
    energy_density = pd.read_csv(path_external_data_standard_elc / "storage_density_kg_per_kwh.csv",index_col=0).transpose()
    # lifetime of storage technologies (in yrs). The lifetime is assumed to be 1.5* the number of cycles divided by the number of days in a year (assuming diurnal use, and 50% extra cycles before replacement, representing continued use below 80% remaining capacity) OR the maximum lifetime in years, which-ever comes first 
    storage_lifetime = pd.read_csv(path_external_data_standard_elc / "storage_lifetime.csv",index_col=0).transpose()

    # material compositions (storage) in wt%
    storage_materials_data = pd.read_csv(path_external_data_standard_elc / 'storage_materials_dynamic.csv', usecols=lambda col: col != "unit")  # wt% of total battery weight for various materials, total battery weight is given by the density file above

    # Using the 250 Wh/kg on the kWh of the various batteries a weight (in kg) of the battery per vehicle category is determined
    # TODO: where is this data from? kWh per battery type?
    battery_weights_data = pd.read_csv(path_external_data_standard_vhc / "battery_weights_kg.csv", index_col=[0,1])

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
    vehicle_list_non_ev = ['Cars - ICE', 'Cars - HEV', 'Cars - FCV', 'Cars - Trolley', 
        'Regular Buses - ICE', 'Regular Buses - HEV', 'Regular Buses - FCV', 
        'Regular Buses - Trolley', 'Midi Buses - ICE', 'Midi Buses - HEV',
       'Midi Buses - FCV', 'Midi Buses - Trolley', 'Heavy Freight Trucks - ICE',
       'Heavy Freight Trucks - HEV', 'Heavy Freight Trucks - FCV',
       'Heavy Freight Trucks - Trolley', 'Medium Freight Trucks - ICE',
       'Medium Freight Trucks - HEV', 'Medium Freight Trucks - FCV',
       'Medium Freight Trucks - Trolley', 'Light Commercial Vehicles - ICE',
       'Light Commercial Vehicles - HEV', 'Light Commercial Vehicles - FCV',
       'Light Commercial Vehicles - Trolley']
    vhc_knowledge_graph = create_vehicle_graph()

    # 1. Market Shares -----------------------------------------------------------------------------

    # 1.1 storage costs
    years = storage_costs.index.astype(int) #.astype(int) to convert years from strings to integers
    techs = storage_costs.columns
    data_array = storage_costs.to_numpy()
    xr_storage_costs = xr.DataArray(
        data_array,
        dims=("Cohort", "BatteryType"),
        coords={
            "Cohort": years,
            "BatteryType": techs
        },
        name="StorageCosts"
    )
    xr_storage_costs = prism.Q_(xr_storage_costs, "USD_cent/kWh")

    # 1.2 storage costs correction (malus/bonus multiplicative factor) 
    years = costs_correction.index.astype(int)
    techs = costs_correction.columns
    data_array = costs_correction.to_numpy()
    xr_costs_correction = xr.DataArray(
        data_array,
        dims=("Cohort", "BatteryType"),
        coords={
            "Cohort": years,
            "BatteryType": techs
        },
        name="StorageCostsCorrection"
    )
    xr_costs_correction = prism.Q_(xr_costs_correction, "dimensionless")

    # 1.3 storage costs longterm decline factor
    techs = cost_decline_longterm_correction.index.rename(None)
    data_array = cost_decline_longterm_correction.to_numpy()
    xr_cost_decline_longterm_correction = xr.DataArray(
        data_array,
        dims=("BatteryType",),
        coords={
            "BatteryType": techs
        },
        name="StorageCostsDeclineLongterm"
    )
    xr_cost_decline_longterm_correction = prism.Q_(xr_cost_decline_longterm_correction, "fraction")


    # 2. battery weigths -----------------------------------------------------------------------

    xr_battery_weights = (
        battery_weights_data
        .rename_axis(index={"time": "Cohort", "type": "Drivetrain"})
        .to_xarray()                        # Convert the pandas DataFrame to xarray with dims: Cohort, Drivetrain, Vehicle
        .to_array("Vehicle")                # Move the DataFrame columns into an explicit xarray dimension
        .rename("BatteryWeights")
    )
    # Combine Vehicle and Drivetrain into a single dimension
    xr_battery_weights = xr_battery_weights.stack(Type=("Vehicle", "Drivetrain"))
    # combine Type string labels: ('Cars', 'BEV') -> 'Cars - BEV'
    new_type = [f"{v} - {d}" for v, d in xr_battery_weights.indexes["Type"]]

    xr_battery_weights = (
        xr_battery_weights
        .drop_vars(["Vehicle", "Drivetrain"]) # Remove now-redundant level coordinates
        .assign_coords(Type=new_type) # Replace the stacked MultiIndex with the combined string labels
    )
    xr_battery_weights = prism.Q_(xr_battery_weights, "kg")
    # rebroadcast so that the Type coordinates have the correct sequence
    xr_battery_weights = vhc_knowledge_graph.rebroadcast_xarray(xr_battery_weights, output_coords=vehicle_list, dim="Type")


    # 3. material intensities -----------------------------------------------------------------------
    
    # Reshape the DataFrame long format: Each material column (steel, aluminium, etc.) is converted into rows,
    # with 'material' indicating the material type and 'MaterialFractions' the corresponding value.
    storage_materials = storage_materials_data.melt(id_vars=['Cohort', 'Type'], var_name='material', value_name='MaterialFractions')
    # Make sure Cohort and Type are treated as categorical (to preserve order)
    storage_materials['Cohort'] = storage_materials['Cohort'].astype(int)
    storage_materials['Type'] = storage_materials['Type'].astype(str)
    # Create xarray DataArray directly
    xr_storage_materials = storage_materials.set_index(['material', 'Cohort', 'Type'])['MaterialFractions'].to_xarray()
    # Ensure correct dimension order
    xr_storage_materials = xr_storage_materials.transpose('material', 'Cohort', 'Type')
    xr_storage_materials = xr_storage_materials.rename({'Type': 'BatteryType'})
    xr_storage_materials = prism.Q_(xr_storage_materials, "fraction")
    xr_battery_materials = xr_storage_materials.sel(BatteryType=EV_BATTERIES)


    # 4. energy density -----------------------------------------------------------------------------
    years = energy_density.index.astype(int)
    energy_density.columns.name = None # remove header "kg/kWh" to avoid issues
    techs = energy_density.columns
    data_array = energy_density.to_numpy()
    xr_energy_density = xr.DataArray(
        data_array,
        dims=("Cohort", "BatteryType"),
        coords={
            "Cohort": years,
            "BatteryType": techs
        },
        name="EnergyDensity"
    )
    xr_energy_density = prism.Q_(xr_energy_density, "kg/kWh")
    xr_energy_density = xr_energy_density.sel(BatteryType=EV_BATTERIES) # Select only those technologies from the storage technologies that are suitable for EV & mobile applications


    # 5. EV fraction available for V2G --------------------------------------------------------------------
    # For this variable first the interpolations are done and then the conversion to xarray DataArray
    x = ev_fraction_v2g_data.reindex(range(ev_fraction_v2g_data.index[0],ev_fraction_v2g_data.index[-1]+1)).interpolate(method="linear")
    y = logistic(x, L=x.iloc[-1].values)
    # y = quadratic(x)
    ev_fraction_v2g = ev_fraction_v2g_data.reindex(range(YEAR_FIRST_GRID,year_out+1)).interpolate(method="linear") # create dataframe with full index; values before first data points will be Nans, between data points interpolated linearly, after last data point will be last known value
    ev_fraction_v2g.loc[:ev_fraction_v2g_data.index[0]] = 0 # set values before first data point to 0
    ev_fraction_v2g.loc[ev_fraction_v2g_data.index[0]:ev_fraction_v2g_data.index[1]] = y # set values between (originally) first and last data point to quadratic/logistic interpolation
    # Build xarray DataArray
    years = ev_fraction_v2g.index.astype(int).rename(None)
    techs = ev_fraction_v2g.columns
    data_array = ev_fraction_v2g.to_numpy()    # shape (Time, Type)
    data_array = data_array[:, :, np.newaxis]   # shape (Time, Type, 1)
    data_array = np.broadcast_to(data_array, data_array.shape[:2] + (len(IMAGE_REGIONS),)) # (Time, Type, Region) - add region dimension, though all regions have the same values for now
    xr_vhc_fraction_v2g = xr.DataArray(
        data_array,
        dims=("Time", "Type", "Region"),
        coords={
            "Time": years,
            "Type": techs,
            "Region": IMAGE_REGIONS
        },
        name="VehicleFractionV2G"
    )
    xr_vhc_fraction_v2g = prism.Q_(xr_vhc_fraction_v2g, "fraction")
    # xr_vhc_fraction_v2g = vhc_knowledge_graph.rebroadcast_xarray(xr_vhc_fraction_v2g, output_coords=vehicle_list, dim="Type")

    # 6. capacity used for V2G -----------------------------------------------------------------------------
    # Note: xr_capacity_fraction_v2g must have same Type coords than xr_vhc_fraction_v2g (otherwise ElectricVehicleBatteries model will crash)
    # Build xarray DataArray
    techs = ev_capacity_fraction_v2g.columns
    data_array = ev_capacity_fraction_v2g.to_numpy().ravel()  # flatten to 1D
    xr_capacity_fraction_v2g = xr.DataArray(
        data_array,
        dims=("Type"),
        coords={
            "Type": techs
        },
        name="CapacityFractionV2g"
    )
    xr_capacity_fraction_v2g = prism.Q_(xr_capacity_fraction_v2g, "fraction")


    ###########################################################################################################
    # Calculations #

    # 1. Market Shares -----------------------------------------------------------------------------

    # calc. storage market shares based on cost developments & multinomial logit model for the years 1970-2050
    storage_market_share = calculate_storage_market_shares(
        xr_storage_costs,
        xr_costs_correction,
        xr_cost_decline_longterm_correction,
        mnlogit_param=-0.2,
        t_start_interpolation=year_start,
        t_end_interpolation=2050
    )

    # fix the market share of storage technologies before year_start and after 2050
    storage_market_share_interp = interpolate_xr(storage_market_share, YEAR_FIRST_GRID, year_out)
    # Select only those technologies from the storage technologies that are suitable for EV & mobile applications
    # normalize the selection of EV battery technologies, so that total market share is 1 again (taking the relative share in the selected battery techs)
    market_share_EVs = normalize_selected_techs(storage_market_share_interp, EV_BATTERIES, dim_type="BatteryType") # TODO: this should be done differently as market shares of EV batteries probably differ from their market shares in total storage market
    market_share_EVs = market_share_EVs.expand_dims(Type=vehicle_list).copy() # add vehicle type dimension
    market_share_EVs.loc[dict(Type=vehicle_list_non_ev)] = 0 # set non-EV vehicle types to zero

    # 2. Battery Weights ----------------------------------------------------------------
    xr_battery_weights_interp = interpolate_xr(xr_battery_weights, YEAR_FIRST_GRID, year_out)

    # 3. Material Intensities ----------------------------------------------------------------
    xr_battery_materials_interp = interpolate_xr(xr_battery_materials, YEAR_FIRST_GRID, year_out)

    #  4. Energy Density ----------------------------------------------------------------
    xr_energy_density_interp = interpolate_xr(xr_energy_density, YEAR_FIRST_GRID, year_out)


    ###########################################################################################################
    # Prep_data File #

    # bring preprocessing data into a generic format for the model
    prep_data = {}
    prep_data["shares"] = market_share_EVs
    prep_data["weights"] = xr_battery_weights_interp
    prep_data["material_fractions"] = xr_battery_materials_interp
    prep_data["energy_density"] = xr_energy_density_interp
    prep_data["vhc_fraction_v2g"] = xr_vhc_fraction_v2g
    prep_data["capacity_fraction_v2g"] = xr_capacity_fraction_v2g
    prep_data["knowledge_graph_elc"] = create_electricity_graph()
    prep_data["knowledge_graph_vhc"] = create_vehicle_graph()

    return prep_data