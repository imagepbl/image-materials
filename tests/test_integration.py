import prism
from imagematerials.model import GenericStocks, GenericMaterials, MaterialIntensities
from imagematerials.factory import ModelFactory, Sector


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
