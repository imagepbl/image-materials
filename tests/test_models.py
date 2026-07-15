"""Test the prism models."""

import inspect
from dataclasses import is_dataclass

import prism
import pytest
import xarray as xr
from pytest import mark

from imagematerials.concepts import knowledge_graph
from imagematerials.concepts import Node, KnowledgeGraph
from imagematerials.model import GenericMaterials, GenericStocks, MaterialIntensities, RestOf
from imagematerials.model import SharesInflowStocks, ElectricVehicleBatteries

@pytest.fixture(scope="module")
def coordinates():
    """Coordinate set to create fake datasets with."""
    return {
        "Region": ["region_a", "region_b"],
        "Type": ["vehicle_a", "vehicle_b"],
        "Cohort": [2000, 2001, 2002],
        "Time": [2000, 2001, 2002],
        "ScipyParam": ["c", "scale"],
        "SuperType": ["vehicle_super"],
    }

@pytest.fixture(scope="module")
def timelines():
    """Complete and simulation timelines."""
    return prism.Timeline(2000, 2002, 1), prism.Timeline(2001, 2002, 1)

@mark.parametrize(
    "model_class", [GenericStocks, GenericMaterials, MaterialIntensities, RestOf, 
                    SharesInflowStocks, ElectricVehicleBatteries]
)
def test_basic_model(model_class):
    """Test to check any model without specific tests and without running it."""
    assert is_dataclass(model_class), (f"{model_class} is not a data class, did you "
                                       "forget to add @prism.interface?")
    assert issubclass(model_class, prism.Model), (f"{model_class} is not a prism.Model, define your"
                                                  f" class as class {model_class}(prism.Model)")

    # Check whether the class variables / compute_values arguments agree with the input_data attr.
    signature = inspect.signature(model_class.compute_values)
    for var_name in model_class.input_data:
        if not (var_name in model_class.__annotations__ or var_name in signature.parameters):
            raise ValueError(f"{model_class} has '{var_name}' listed as input variable, "
                             "but this is not the case.")
        if var_name in model_class.__annotations__ and var_name in signature.parameters:
            raise ValueError(f"{model_class} has {var_name} both as static and dynamic input.")

    # Check if all class attributes are accounted for.
    for var_name, data_type in model_class.__annotations__.items():
        if isinstance(data_type, prism._typing.CoordsType):
            continue
        if isinstance(data_type, prism._time_variable.TimeVariableType):
            assert var_name in model_class.output_data
            continue
        if var_name in model_class.input_data:
            continue
        assert var_name in ["input_data", "output_data", "time", "transition_years"], (
            f"Unknown dataclass attribute '{var_name}' for '{model_class}'")

def _get_xarray(coordinates, *dims):
    """Create a quick array."""
    return xr.DataArray(1.0,
                         dims=dims,
                         coords={d: coordinates[d] for d in dims})

def test_generic_stocks(coordinates, timelines):
    """Test the GenericStocks model."""
    stocks = _get_xarray(coordinates, "Time", "Region", "Type")
    stocks = prism.Q_(stocks, "count")
    lt =  _get_xarray(coordinates, "Time", "Region", "Type", "ScipyParam")
    lt.attrs["loc"] = 0
    lifetimes = {"weibull": lt}
    complete_timeline, simulation_timeline = timelines
    model = GenericStocks(
        complete_timeline, stocks=stocks, lifetimes=lifetimes,
        knowledge_graph=knowledge_graph, set_unit_flexible="count",
        Region=coordinates["Region"], Type=coordinates["Type"],
        Cohort=coordinates["Cohort"], Time=coordinates["Time"])
    model.simulate(complete_timeline)
    for var_name in model.output_data:
        assert hasattr(model, var_name)

    for t in coordinates["Time"]:
        assert (model.stock_by_cohort.loc[t].sum("Cohort") == stocks.loc[t]).all()


@pytest.fixture(scope="module")
def test_knowledge_graph():
    """Create a test knowledge graph."""
    return KnowledgeGraph(
        Node("vehicle_super"),
        Node("vehicle_a", inherits_from="vehicle_super"),
        Node("vehicle_b", inherits_from="vehicle_super"),
    )
def test_shares_inflow_stocks(coordinates, timelines, test_knowledge_graph):
    """Test the SharesInflowStocks model."""
    stocks = _get_xarray(coordinates, "Time", "Region", "SuperType")
    stocks = prism.Q_(stocks, "count")

   
    shares = _get_xarray(coordinates, "Region", "Type", "Cohort")
    shares = shares / shares.sum("Type") 

    lt = _get_xarray(coordinates, "Time", "Region", "Type", "ScipyParam")
    lt.attrs["loc"] = 0
    lifetimes = {"weibull": lt}

    complete_timeline, simulation_timeline = timelines

    model = SharesInflowStocks(
        complete_timeline,
        stocks=stocks,
        shares=shares,
        lifetimes=lifetimes,
        knowledge_graph=test_knowledge_graph,
        set_unit_flexible="count",
        Region=coordinates["Region"],
        SuperType=coordinates["SuperType"],
        Type=coordinates["Type"],
        Cohort=coordinates["Cohort"],
        Time=coordinates["Time"]
    )

    model.simulate(complete_timeline)

    for var_name in ["outflow_by_cohort", "inflow", "stock_by_cohort"]:
        assert hasattr(model, var_name)
    
    for t in coordinates["Time"]:
        stock_by_type = model.stock_by_cohort.loc[t].sum("Cohort")
        stock_by_supertype = test_knowledge_graph.aggregate_sum(
            stock_by_type,
            stocks.coords["SuperType"].values,
            dim="Type"
        ).rename({"Type": "SuperType"})
        
        # With constant stock targets (all 1.0) and fast decay (Weibull c=1, scale=1),
        # inflow is always positive, so stock equals demand exactly.
        assert (stock_by_supertype == stocks.loc[t]).all()

def test_electric_vehicle_batteries(coordinates, timelines):
    """Test the ElectricVehicleBatteries model for critical calculation logic."""
    complete_timeline, simulation_timeline = timelines

    battery_types = ["NMC", "LFP"]
    material_types = ["copper", "lithium"]
    v2g_types = ["Cars - BEV"]

    road_vehicle_types = [
        'Cars - BEV', 'Cars - FCV', 'Cars - HEV', 'Cars - ICE',
        'Cars - PHEV', 'Cars - Trolley', 'Heavy Freight Trucks - BEV',
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
        'Regular Buses - PHEV', 'Regular Buses - Trolley',
    ]

    time_coords = coordinates["Time"]
    cohort_coords = coordinates["Cohort"]
    region_coords = coordinates["Region"]
 
    inflow_da = xr.DataArray(
        100.0, dims=("time", "Region", "Type"),
        coords={"time": time_coords, "Region": region_coords, "Type": road_vehicle_types})
    inflow_da = prism.Q_(inflow_da, "count")
    inflow = prism.TimeVariable.from_array(inflow_da)

    outflow_da = xr.DataArray(
        20.0, dims=("time", "Cohort", "Region", "Type"),
        coords={"time": time_coords, "Cohort": cohort_coords,
                "Region": region_coords, "Type": road_vehicle_types})
    outflow_da = prism.Q_(outflow_da, "count")
    outflow_by_cohort = prism.TimeVariable.from_array(outflow_da)

    # stock_by_cohort uses .loc[t] in the model, so plain Q_ with "Time" dim works
    stock_by_cohort = xr.DataArray(
        50.0, dims=("Time", "Cohort", "Region", "Type"),
        coords={"Time": time_coords, "Cohort": cohort_coords,
                "Region": region_coords, "Type": road_vehicle_types})
    stock_by_cohort = prism.Q_(stock_by_cohort, "count")

    # --- Battery-specific input data ---

    shares = xr.DataArray(
        0.5, dims=("BatteryType", "Type", "Cohort"),
        coords={"BatteryType": battery_types, "Type": road_vehicle_types,
                "Cohort": cohort_coords})
    shares = shares / shares.sum("BatteryType")

    # Battery weights: kg per vehicle
    weights = xr.DataArray(
        10.0, dims=("BatteryType", "Type", "Cohort"),
        coords={"BatteryType": battery_types, "Type": road_vehicle_types,
                "Cohort": cohort_coords})
    weights = prism.Q_(weights, "kg/count")  # kg per vehicle

    # Energy density: kg/kWh
    energy_density = xr.DataArray(
        10.0, dims=("BatteryType", "Type", "Cohort"),
        coords={"BatteryType": battery_types, "Type": road_vehicle_types,
                "Cohort": cohort_coords})
    energy_density = prism.Q_(energy_density, "kg/kWh")

    # Shares and material_fractions are dimensionless (fractions) — leave as plain arrays

    material_fractions = xr.DataArray(
    0.5, dims=("material", "BatteryType", "Type", "Cohort"),
    coords={"material": material_types, "BatteryType": battery_types,
            "Type": road_vehicle_types, "Cohort": cohort_coords})
    material_fractions = material_fractions / material_fractions.sum("material")

    vhc_fraction_v2g = xr.DataArray(
    0.8, dims=("Type", "Time", "Cohort"),
    coords={"Type": v2g_types,
            "Time": time_coords,
            "Cohort": cohort_coords})
    
    capacity_fraction_v2g = xr.DataArray(
        0.9, dims=("Type", "BatteryType"),
        coords={"Type": v2g_types, "BatteryType": battery_types})

    # --- Instantiate and run ---

    model = ElectricVehicleBatteries(
        complete_timeline,
        weights=weights,
        shares=shares,
        material_fractions=material_fractions,
        energy_density=energy_density,
        vhc_fraction_v2g=vhc_fraction_v2g,
        capacity_fraction_v2g=capacity_fraction_v2g,
        Type=road_vehicle_types,
        BatteryType=battery_types,
        Region=region_coords,
        Cohort=cohort_coords,
        material=material_types,
        Time=time_coords,
    )

    def get_inputs(time_value):
        return {
            'inflow': inflow,
            'stock_by_cohort': stock_by_cohort,
            'outflow_by_cohort': outflow_by_cohort,
        }

    model.simulate(complete_timeline, inputs=get_inputs)

    
    expected_outputs = ['stock_battery_kWh_v2g', 'inflow_battery_kWh', 'stock_battery_kWh', 'outflow_battery_kWh', 'inflow_battery_materials', 'stock_battery_materials', 'outflow_battery_materials']
    for var_name in expected_outputs:
        assert hasattr(model, var_name), f"Expected output {var_name} not found on model"
    
    t = time_coords[0]  # 2000
    n_types = len(road_vehicle_types)
    n_battery_types = len(battery_types)
    n_regions = len(region_coords)

    # Per (BatteryType, Type, Region): mass = 100 × 0.5 × 10 = 500 kg
    expected_mass_per_bt_type = 100.0 * 0.5 * 10.0
    # kWh = 500 / 10 = 50
    expected_kwh_per_bt_type = expected_mass_per_bt_type / 10.0
    expected_total_inflow_kwh = expected_kwh_per_bt_type * n_battery_types * n_types * n_regions

    inflow_kwh_total = prism.M_(model.inflow_battery_kWh[t].sum())
    assert inflow_kwh_total == pytest.approx(expected_total_inflow_kwh, rel=0.01), \
        f"Inflow kWh mismatch: expected {expected_total_inflow_kwh}, got {inflow_kwh_total}"

    # Material mass total = battery mass total (fractions sum to 1.0)
    expected_total_inflow_mass = expected_mass_per_bt_type * n_battery_types * n_types * n_regions
    material_mass_total = prism.M_(model.inflow_battery_materials[t].sum())
    assert material_mass_total == pytest.approx(expected_total_inflow_mass, rel=0.01), \
        f"Material mass mismatch: expected {expected_total_inflow_mass}, got {material_mass_total}"

    # V2G stock only includes V2G-capable types
    assert set(model.stock_battery_kWh_v2g.Type.values) == set(v2g_types)

    # V2G stock ≤ total stock
    total_stock_kwh = prism.M_(model.stock_battery_kWh[t].sum())
    v2g_stock_kwh = prism.M_(model.stock_battery_kWh_v2g.loc[dict(Time=t)].sum())
    assert v2g_stock_kwh <= total_stock_kwh