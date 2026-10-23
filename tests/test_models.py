"""Test the prism models."""

import inspect
from dataclasses import is_dataclass
from imagematerials.lifetimes import lifetimes_to_matrix
import prism
import pytest
import xarray as xr
from pytest import mark
import numpy as np

from imagematerials.concepts import knowledge_graph
from imagematerials.concepts import Node, KnowledgeGraph
from imagematerials.model import GenericMaterials, GenericStocks, MaterialIntensities, RestOf
from imagematerials.model import SharesInflowStocks, ElectricVehicleBatteries

from imagematerials.vehicles.constants import (
    EV_VEHICLE_TYPES
)

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
    stocks = _get_xarray(coordinates, "Time", "Region", "Type").rename({"Time": "time"})
    stocks = prism.Q_(stocks, "count")
    lt =  _get_xarray(coordinates, "Time", "Region", "Type", "ScipyParam").rename({"Time": "time"})
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

    assert isinstance(model.survival_matrix, prism.TimeVariable)
    assert "time" in model.survival_matrix.to_array().dims
    assert "Cohort" in [d.label for d in model.survival_matrix.dims]

    # age-0 survival should always be 1.0
    t0 = coordinates["Time"][0]
    assert np.allclose(model.survival_matrix[t0].sel(Cohort=t0).values, 1.0)

    raw = lifetimes_to_matrix(lifetimes, stocks.coords["Type"], knowledge_graph=knowledge_graph)
    t, t_future = coordinates["Time"][0], coordinates["Time"][2]
    assert np.allclose(
        model.survival_matrix[t_future].sel(Cohort=t).values,
        raw[t_future].sel(Cohort=t).values 
    )


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
    stocks = _get_xarray(coordinates, "Time", "Region", "SuperType").rename({"Time": "time"})
    stocks = prism.Q_(stocks, "count")

   
    shares = _get_xarray(coordinates, "Region", "Type", "Cohort")
    shares = shares / shares.sum("Type") 

    lt = _get_xarray(coordinates, "Time", "Region", "Type", "ScipyParam").rename({"Time": "time"})
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

    assert isinstance(model.survival_matrix, prism.TimeVariable)
    assert "Cohort" in [d.label for d in model.survival_matrix.dims]

    t0 = coordinates["Time"][0]
    assert np.allclose(model.survival_matrix[t0].sel(Cohort=t0).values, 1.0)

    types_to_model = next(iter(lifetimes.values())).coords["Type"].values
    raw = lifetimes_to_matrix(lifetimes, types_to_model, knowledge_graph=test_knowledge_graph)
    t, t_future = coordinates["Time"][0], coordinates["Time"][2]
    assert np.allclose(
        model.survival_matrix[t_future].sel(Cohort=t).values,
        raw[t_future].sel(Cohort=t).values
    )

def test_electric_vehicle_batteries(coordinates, timelines):
    """Test the ElectricVehicleBatteries model for critical calculation logic."""
    complete_timeline, simulation_timeline = timelines

    battery_types = ["high-nickel", "LFP"]
    material_types = ["copper", "lithium"]
    v2g_types = ["Cars - BEV"]

    time_coords = coordinates["Time"]
    cohort_coords = coordinates["Cohort"]
    region_coords = coordinates["Region"]
 
    inflow_da = xr.DataArray(
        100.0, dims=("time", "Region", "Type"),
        coords={"time": time_coords, "Region": region_coords, "Type": EV_VEHICLE_TYPES})
    inflow_da = prism.Q_(inflow_da, "count")
    inflow = prism.TimeVariable.from_array(inflow_da)

    outflow_da = xr.DataArray(
        20.0, dims=("time", "Cohort", "Region", "Type"),
        coords={"time": time_coords, "Cohort": cohort_coords,
                "Region": region_coords, "Type": EV_VEHICLE_TYPES})
    outflow_da = prism.Q_(outflow_da, "count")
    outflow_by_cohort = prism.TimeVariable.from_array(outflow_da)

    # stock_by_cohort uses .loc[t] in the model, so plain Q_ with "Time" dim works
    stock_by_cohort = xr.DataArray(
        50.0, dims=("time", "Cohort", "Region", "Type"),
        coords={"time": time_coords, "Cohort": cohort_coords,
                "Region": region_coords, "Type": EV_VEHICLE_TYPES})
    stock_by_cohort = prism.Q_(stock_by_cohort, "count")

    # --- Battery-specific input data ---

    shares = xr.DataArray(
        0.5, dims=("BatteryType", "Type", "Cohort"),
        coords={"BatteryType": battery_types, "Type": EV_VEHICLE_TYPES,
                "Cohort": cohort_coords})
    shares = shares / shares.sum("BatteryType")

    # Battery capacity: kWh/vehicle
    capacity = xr.DataArray(
            10.0, dims=("Type", "Cohort"),
            coords={"Type": EV_VEHICLE_TYPES,
                    "Cohort": cohort_coords})
    capacity = prism.Q_(capacity, "kWh/count")  # kg per vehicle

    material_intensities = xr.DataArray(
    5, dims=("material", "BatteryType", "Type", "Cohort"),
    coords={"material": material_types, "BatteryType": battery_types,
            "Type": EV_VEHICLE_TYPES, "Cohort": cohort_coords})
    material_intensities = prism.Q_(material_intensities, "kg/kWh")

    vhc_fraction_v2g = xr.DataArray(
    0.8, dims=("Type", "time", "Cohort"),
    coords={"Type": v2g_types,
            "time": time_coords,
            "Cohort": cohort_coords})
    
    capacity_fraction_v2g = xr.DataArray(
        0.9, dims=("Type", "BatteryType"),
        coords={"Type": v2g_types, "BatteryType": battery_types})

    # --- Instantiate and run ---

    model = ElectricVehicleBatteries(
        complete_timeline,
        battery_capacity=capacity,
        shares=shares,
        material_intensities=material_intensities,
        vhc_fraction_v2g=vhc_fraction_v2g,
        capacity_fraction_v2g=capacity_fraction_v2g,
        Type=EV_VEHICLE_TYPES,
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
    n_types = len(EV_VEHICLE_TYPES)
    n_battery_types = len(battery_types)
    n_regions = len(region_coords)

    # Per (BatteryType, Type, Region): mass = 100 vehicles × 0.5 shares × 10 kWh/vehicle = 500 kWh
    expected_kwh_per_bt_type = 100.0 * 0.5 * 10.0
    expected_total_inflow_kwh = expected_kwh_per_bt_type * n_battery_types * n_types * n_regions

    inflow_kwh_total = prism.M_(model.inflow_battery_kWh[t].sum())
    assert inflow_kwh_total == pytest.approx(expected_total_inflow_kwh, rel=0.01), \
        f"Inflow kWh mismatch: expected {expected_total_inflow_kwh}, got {inflow_kwh_total}"

    # V2G stock only includes V2G-capable types
    assert set(model.stock_battery_kWh_v2g.Type.values) == set(v2g_types)

    # V2G stock ≤ total stock
    total_stock_kwh = prism.M_(model.stock_battery_kWh[t].sum())
    v2g_stock_kwh = prism.M_(model.stock_battery_kWh_v2g.loc[dict(time=t)].sum())
    assert v2g_stock_kwh <= total_stock_kwh