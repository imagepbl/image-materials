import prism
from pathlib import Path
from imagematerials.model import GenericStocks, GenericMaterials, MaterialIntensities
from imagematerials.factory import ModelFactory, Sector
from imagematerials.vehicles.preprocessing.util import get_passengerkms, get_tonkms
from imagematerials.util import read_climate_policy_config
from imagematerials.vehicles.timermodel import TIMERVehicleMaterials


def test_vehicle_stocks(vhc_prep_data, tmpdir):
    time_start = 1960
    sim_end = 1980
    complete_timeline = prism.Timeline(time_start, sim_end, 1)
    simulation_timeline = prism.Timeline(1970, sim_end, 1)

    sector = Sector("vhc", vhc_prep_data)
    factory = ModelFactory(sector, complete_timeline)
    model = factory.add(GenericStocks).add(GenericMaterials).finish()
    model.simulate(simulation_timeline)
    model.save_pkl(tmpdir / "test.pkl")
    model2 = ModelFactory.load_pkl(tmpdir / "test.pkl")
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


def test_timer_interface(tmpdir):
    # Run simulation both in standalone and simulated TIMER-coupled mode.
    # Ensure results are the same.
    sim_end = prism.Q_(1980, 'year')
    complete_timeline = prism.Timeline(
        start=prism.Q_(1960, 'year'),
        end=sim_end,
        stepsize=prism.Q_(1, 'year')
    )
    simulation_timeline = prism.Timeline(
        start=prism.Q_(1970, 'year'),
        end=sim_end,
        stepsize=prism.Q_(1, 'year')
    )

    # Reuse model output from earlier test
    # TODO: make this work in a more clean way
    vehicles_model = ModelFactory.load_pkl(tmpdir / "test.pkl")

    # Get the raw data that otherwise would come from TIMER
    path_test_scenario = Path("data", "raw", "image", "SSP2_baseline")
    climate_policy_config = read_climate_policy_config(path_test_scenario)
    # Use own functions to retrieve data ...
    # passenger_kms = get_passengerkms(path_test_scenario, climate_policy_config)
    # tonkms = get_tonkms(path_test_scenario, climate_policy_config)
    # TODO: turn into TimeVariables
    # ...or use Prism built-in functionality:
    # TODO: check this
    passenger_kms = prism.TimeVariable(
        timeline=simulation_timeline,
        dims=(Region, Type),
        unit="km",
        file=path_test_scenario.joinpath(
            climate_policy_config['data_files']['transport']['passenger']['kilometers']
        )
    )
    tonkms = prism.TimeVariable(
        timeline=simulation_timeline,
        dims=(Region, Type),
        unit="km",
        file=path_test_scenario.joinpath(
            climate_policy_config['data_files']['transport']['freight']['Tkm']
        )
    )
    def timer_input(t):
        return {
            "passenger_kms": passenger_kms[t],
            "tonkms": tonkms[t]
        }

    timer_vehicle_model = TIMERVehicleMaterials(complete_timeline) # TODO: fix signature
    # Simulate TIMER's input using the timer_input function
    timer_vehicle_model.simulate(simulation_timeline, inputs=timer_input)
    
    # Compare outputs of both model runs
    # TODO: decide what to compare
    vehicles_model.vehicles.get("inflow_materials")
    timer_vehicle_model.get("inflow_materials")

