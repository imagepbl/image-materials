

import matplotlib.pyplot as plt
import warnings
from pathlib import Path
import warnings
from pint.errors import UnitStrippedWarning
warnings.simplefilter("ignore", UnitStrippedWarning)

from imagematerials.electricity.constants import STANDARD_SCEN_EXTERNAL_DATA

from imagematerials.model import (
    GenericStocks,
    MaterialIntensities,
    SharesInflowStocks,
    ElectricVehicleBatteries,
    EvBatteryLinkModule
)
from imagematerials.factory import ModelFactory
from imagematerials.preprocessing import get_preprocessing_data
import prism


# TODO absolute path of file "preprocessing.py" ? current solution can differ depending on IDE used
path_current = Path().resolve() #
path_base = path_current#.parent.parent # base path of the project -> image-materials
path_base = Path(path_base, "data", "raw") #



scenario_list = {#"SSP2_baseline":("SSP2_baseline", None),
                #  "SSP2_narrow":("SSP2_narrow", ["narrow"]),
                # "SSP2_slow":("SSP2_slow", ["slow"]),
                "SSP2_baseline":("SSP2_baseline", None)
                 }

t_start = 1971
t_end = 2060
complete_timeline = prism.Timeline(t_start, t_end, 1) #YEAR_END
simulation_timeline = prism.Timeline(t_start, t_end, 1)

all_output = {}

for scen_id, (climate_scen, circular_scen) in scenario_list.items():
    
    # Paths to scenario data
    print(climate_scen)
    print(circular_scen)
    if circular_scen is not None:
        climate_policy_scenario_dir = Path(path_base, "IMAGE_CircoMod", climate_scen)
        circular_economy_scenario_dirs = {
            scenario: Path(path_base, "circular_economy_scenarios", scenario) for scenario in circular_scen # path to circular economy scenario data
        }
    else:
        climate_policy_scenario_dir = Path(path_base, "image", climate_scen)
        circular_economy_scenario_dirs = None
    scenario = path_base /"electricity"/ STANDARD_SCEN_EXTERNAL_DATA

    elc_sector = get_preprocessing_data("electricity", path_base,
                                        climate_policy_scenario_dir, 
                                        circular_economy_scenario_dirs, cache = False) 
    # elc_sector is a list of preprocessing data for each electricity subsector

    elc_sector_gen = next(item for item in elc_sector if item.name == "elc_gen")

    factory = ModelFactory(
        elc_sector_gen, complete_timeline
        ).add(SharesInflowStocks
        ).add(MaterialIntensities
        )
    model = factory.finish()

    model.simulate(simulation_timeline)
    print(f"Finished {scen_id}")
    list(model.elc_gen)