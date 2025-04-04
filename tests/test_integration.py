import prism
from imagematerials.model import GenericStocks, GenericMaterials
from imagematerials.factory import ModelFactory


def test_vehicle_stocks(vhc_prep_data):
    time_start = 1960
    sim_end = 1980
    complete_timeline = prism.Timeline(time_start, sim_end, 1)
    simulation_timeline = prism.Timeline(1970, sim_end, 1)

    model = ModelFactory(vhc_prep_data, complete_timeline).add(GenericStocks).add(GenericMaterials).finish()
    model.simulate(simulation_timeline)
