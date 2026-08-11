from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import prism
import warnings
import numpy as np
import time

from imagematerials.concepts import create_electricity_graph
from imagematerials.factory import ModelFactory
from imagematerials.model import GenericStocks, SharesInflowStocks, GenericMaterials, MaterialIntensities
from imagematerials.preprocessing import get_preprocessing_data

# from imagematerials.sensitivity_analysis.changedata import change_sector, ChangeAction, ChangeReplace
from imagematerials.sensitivity_analysis.changedata_test import change_sector, ChangeAction, ChangeReplace, ChangeRename, ChangeDelete
from imagematerials.sensitivity_analysis.monte_carlo import sample_values, load_material_intensities, load_lifetimes, process_material_intensities

from imagematerials.maintenance import Maintenance
from imagematerials.vehicles.constants import vehicles_modes_sensitivity_analysis
from imagematerials.electricity.constants import EPG_TECHNOLOGIES_FINAL
from imagematerials.electricity.utils import rebroadcast_type_dims
warnings.filterwarnings("ignore")

knowledge_graph_electricity = create_electricity_graph()

path_current = Path().resolve()
path_base = path_current.parent #.parent # base path of the project -> image-materials
path_base = Path(path_base, "data", "raw")


####################################################################################################
# Initialize sectors 

# vehicles -----------------------------------------------------------------------------------------

# Get the preprocessing data for the vehicles sector only once
vhc_sector = get_preprocessing_data(
    "vehicles", Path("..", "data", "raw"),
    climate_policy_scenario_dir = Path("..", "data", "raw", "image", "SSP2_baseline"), 
    circular_economy_scenario_dirs = None
)
# sensitivity analysis currently only implemented for road vehicle types (due to lack of data): select those from the preprocessing data
stocks = vhc_sector.all_data["stocks"]
stocks = stocks.sel(Type = vehicles_modes_sensitivity_analysis)

# load material intensities and use them to replace the material fractions in the preprocessing data
# use the standard values (column "values") - needs to be renamed to "sampled" though to be able to 
# use the function process_material_intensities() to convert the data to the correct format for the model
mi = load_material_intensities(path_base / "vehicles" / "standard_data" / "vehicles_material_intensities_long.csv", sector="vehicles")
mi = mi.rename(columns={'value': 'sampled'})
mi = process_material_intensities(mi, "vehicles")

# change vehicle sector to fit the set up of the sensitivity analysis
change_definition = {
        "maintenance_material_fractions": ChangeDelete(),
        "stocks": ChangeReplace(stocks),
        "weights": ChangeDelete(),
        "material_fractions": ChangeRename("material_intensities"),
        "material_intensities": ChangeReplace(mi),
    }
vhc_sector = change_sector(vhc_sector, change_definition, inplace=True)

# electricity - generation -------------------------------------------------------------------------
elc_sector = get_preprocessing_data(
    "electricity", Path("..", "data", "raw"),
    climate_policy_scenario_dir = Path("..", "data", "raw", "image", "SSP2_baseline"),
    circular_economy_scenario_dirs = None
)
# elc_sector is a list of preprocessing data for each electricity subsector

elc_sector_gen = next(item for item in elc_sector if item.name == "elc_gen")

####################################################################################################
# Run Monte-Carlo

time_start = 1998
time_end = 2100
complete_timeline = prism.Timeline(time_start, time_end, 1)
simulation_timeline = prism.Timeline(time_start, time_end, 1)

ranges_vehicles = load_material_intensities(path_base / "vehicles" / "standard_data" / "vehicles_material_intensities_long.csv", "vehicles")
ranges_generation = load_material_intensities(path_base / "electricity" / "standard_data" / "generation_material_intensities_long.csv", "electricity")

rng = np.random.default_rng(42)   # seed once for reproducibility


start = time.time()
all_output = {}
N = 10
for i in range(N):
    # vehicles ------------------------------------------------------------
    mi = sample_values(ranges_vehicles, rng=rng)
    mi = process_material_intensities(mi, "vehicles")
    change_definition_vehicles = {
        "material_intensities": ChangeReplace(mi),
    }
    new_vhc_sector = change_sector(vhc_sector, change_definition_vehicles, inplace=False)

    # electricity - generation --------------------------------------------
    mi = sample_values(ranges_generation, rng=rng)
    mi = process_material_intensities(mi, "electricity")
    
    change_definition_generation = {
        "material_intensities": ChangeReplace(mi),
    }
    new_elc_sector_gen = change_sector(elc_sector_gen, change_definition_generation, inplace=False)

    # model run ------------------------------------------------------------
    factory = ModelFactory(
        [new_vhc_sector, new_elc_sector_gen], complete_timeline
        ).add(GenericStocks, ["vehicles"]
        ).add(SharesInflowStocks, ["elc_gen"]
        ).add(MaterialIntensities, ["elc_gen",
                                    "vehicles"]
        )
    model = factory.finish()
    model.simulate(simulation_timeline)
    all_output[i] = model
    print(f"\rSimulation {i} completed.     ", end="")

end = time.time()
print(f"Total time for {N} simulations: {(end - start)/60:.1f} minutes.")