import cProfile

from imagematerials.vehicles.__main__ import simulate_stocks
from imagematerials.vehicles.preprocessing import import_from_netcdf

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    simulate_stocks(import_from_netcdf("prep_vema.nc"))
    profiler.disable()
    profiler.dump_stats("all.prof")
