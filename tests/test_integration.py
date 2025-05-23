import prism
from imagematerials.model import GenericStocks, GenericMaterials, MaterialIntensities
from imagematerials.factory import ModelFactory


def test_vehicle_stocks(vhc_prep_data):
    time_start = 1960
    sim_end = 1980
    complete_timeline = prism.Timeline(time_start, sim_end, 1)
    simulation_timeline = prism.Timeline(1970, sim_end, 1)

    model = ModelFactory(vhc_prep_data, complete_timeline).add(GenericStocks).add(GenericMaterials).finish()
    model.simulate(simulation_timeline)

def test_buildings_stocks(bld_prep_data):
    time_start = 1960
    sim_end = 1980
    complete_timeline = prism.Timeline(time_start, sim_end, 1)
    simulation_timeline = prism.Timeline(1970, sim_end, 1)

    model = ModelFactory(bld_prep_data, complete_timeline).add(GenericStocks).add(MaterialIntensities).finish()
    model.simulate(simulation_timeline)


def test_combined_stocks(bld_prep_data, vhc_prep_data):
    time_start = 1960
    sim_end = 1980
    complete_timeline = prism.Timeline(time_start, sim_end, 1)
    simulation_timeline = prism.Timeline(1970, sim_end, 1)

    factory = ModelFactory({"bld": bld_prep_data, "vhc": vhc_prep_data}, complete_timeline)
    factory.add(GenericStocks, "vhc").add(GenericMaterials, "vhc")
    factory.add(GenericStocks, "bld").add(MaterialIntensities, "bld")
    model = factory.finish()
    model.simulate(simulation_timeline)
    assert len(model.stocks) == 2
