import prism
import pytest
import numpy as np
from pathlib import Path
from imagematerials.model import GenericStocks, GenericMaterials, MaterialIntensities
from imagematerials.factory import ModelFactory, Sector
from imagematerials.vehicles.preprocessing.util import simulate_timer_data
from imagematerials.util import read_climate_policy_config
# from imagematerials.vehicles.timermodel import TIMERMaterials
from imagematerials.vehicles.stocks_prism import VehicleStocks


@pytest.fixture(scope="session")
def pickle_path(tmp_path_factory):
    temp_path = tmp_path_factory.mktemp("data")
    return temp_path


@pytest.fixture(scope="module")
def time_start():
    return 1960  # prism.Q_(1960, 'year')


@pytest.fixture(scope="module")
def simulation_start():
    return 1971  # prism.Q_(1971, 'year')


@pytest.fixture(scope="module")
def simulation_end():
    return 1980  # prism.Q_(1980, 'year')


@pytest.fixture(scope="module")
def complete_timeline(time_start, simulation_end):
    return prism.Timeline(
        start=time_start,
        end=simulation_end,
        stepsize=1  # prism.Q_(1, 'year')
    )


@pytest.fixture(scope="module")
def simulation_timeline(simulation_start, simulation_end):
    return prism.Timeline(
        start=simulation_start,
        end=simulation_end,
        stepsize=1  # prism.Q_(1, 'year')
    )


@pytest.fixture(scope="module")
def vhc_model(vhc_prep_data, complete_timeline, simulation_timeline):
    sector = Sector("vhc", vhc_prep_data)
    factory = ModelFactory(sector, complete_timeline)
    model = factory.add(GenericStocks).add(GenericMaterials).finish()
    model.simulate(simulation_timeline)
    return model


def test_vehicle_stocks(vhc_model, pickle_path):
    model = vhc_model
    model.save_pkl(pickle_path / "test.pkl")
    model2 = ModelFactory.load_pkl(pickle_path / "test.pkl")
    assert model.stocks.equals(model2.stocks)


def test_buildings_stocks(bld_prep_data):
    time_start = 1960
    sim_end = 1980
    complete_timeline = prism.Timeline(time_start, sim_end, 1)
    simulation_timeline = prism.Timeline(1970, sim_end, 1)

    sector = Sector("bld", bld_prep_data)
    model = ModelFactory(sector, complete_timeline).add(GenericStocks).add(MaterialIntensities).finish()
    model.simulate(simulation_timeline)


def test_combined_stocks(bld_prep_data, vhc_prep_data):
    time_start = 1960
    sim_end = 1980
    complete_timeline = prism.Timeline(time_start, sim_end, 1)
    simulation_timeline = prism.Timeline(1970, sim_end, 1)

    sectors = [Sector("bld", bld_prep_data), Sector("vhc", vhc_prep_data)]
    factory = ModelFactory(sectors, complete_timeline)
    factory.add(GenericStocks, "vhc").add(GenericMaterials, "vhc")
    factory.add(GenericStocks, "bld").add(MaterialIntensities, "bld")
    model = factory.finish()
    model.simulate(simulation_timeline)
    assert len(model.stocks) == 2


def test_timer_interface(vhc_model, complete_timeline, simulation_timeline):
    # Run simulation both in standalone and simulated TIMER-coupled mode.
    # Ensure results are the same.

    # Get the raw data that otherwise would come from TIMER
    path_test_scenario = Path("data", "raw", "image", "SSP2_baseline")
    timer_data = simulate_timer_data(path_test_scenario)

    # Retrieve TIMER transport activity inputs from preprocessing helpers.
    passenger_kms = timer_data["passenger_kms"]
    tonne_kms = timer_data["tonne_kms"]
    load_car = timer_data["load_car"]

    def timer_input(t):
        return {
            "passenger_kms": passenger_kms.sel(Time=t),
            "tonne_kms": tonne_kms.sel(Time=t),
            "load_car": load_car.sel(Time=t)
        }

    # For now only testing the VehicleStocks
    timer_vehicle_model = VehicleStocks(complete_timeline, path_test_scenario)
    # Simulate TIMER's input using the timer_input function
    timer_vehicle_model.simulate(simulation_timeline, inputs=timer_input)
    
    # Compare outputs of both model runs
    compare_total_vehicles_stocks(vhc_model.stocks, timer_vehicle_model.total_vehicles, simulation_timeline)  # timer_vehicle_model.stocks

def compare_total_vehicles_stocks(standalone_stocks, timer_stocks, timeline):
    # Get simulation years to compare
    for time in range(timeline.start, timeline.end, timeline.stepsize):
        if time != timeline.end:
            continue
        for vehicle in timer_stocks.coords["Type"].values:
            # skip these for now as they are still need to be split up in suptypes in the new calculation
            # compare to 1e-6 to account for small discrepancies in data handling
            if vehicle not in ["Light Commercial Vehicles", "Cars", "Medium Freight Trucks", "Heavy Freight Trucks", "Regular Buses", "Midi Buses"]:
                compare_old = standalone_stocks.sel(Type = vehicle).loc[int(prism.M_(time))].sum()
                compare_new = timer_stocks.sel(Type=vehicle).sum()
                assert np.isclose(compare_old, compare_new), f"Discrepancy in {vehicle} calculation for time {time}: original total {standalone_stocks.sel(Type = vehicle).loc[int(prism.M_(time))].sum().values}, new total {timer_stocks.sel(Type=vehicle).sum().values}"
