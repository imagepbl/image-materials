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
from imagematerials.model import SharesInflowStocks
from imagematerials.vehicles.battery import Battery

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
                    SharesInflowStocks,Battery]
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
        assert var_name in ["input_data", "output_data", "time"], (
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
        assert (stock_by_supertype == stocks.loc[t]).all()