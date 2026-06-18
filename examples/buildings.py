from pathlib import Path
import matplotlib.pyplot as plt

import prism

from imagematerials.factory import ModelFactory
from imagematerials.model import (
    StocksQuintiles,
    MaterialIntensitiesQuintiles,
)
from imagematerials.preprocessing import get_preprocessing_data

scenario_list = {"SSP2_baseline": ("SSP2_baseline", ["test_flags"])}

scenario_base_path = Path("data", "raw")
scenario_cp_path = scenario_base_path / "image"
scenario_ce_path = scenario_base_path / "circular_economy_scenarios"

time_start = 1971
complete_timeline = prism.Timeline(1971, 2060, 1)
simulation_timeline = prism.Timeline(1971, 2060, 1)

all_output = {}

for scen_id, (climate_scen, circular_scen) in scenario_list.items():
    climate_policy_scenario_dir = scenario_cp_path / climate_scen
    circular_economy_scenario_dirs = {
        scenario: scenario_ce_path / scenario for scenario in circular_scen
    }

    bld_sector = get_preprocessing_data("buildings", scenario_base_path,
                                        climate_policy_scenario_dir, 
                                        circular_economy_scenario_dirs) 


    factory = ModelFactory(
    [bld_sector], complete_timeline
    ).add(StocksQuintiles, ["buildings"]
    ).add(MaterialIntensitiesQuintiles, "buildings",
)
    model = factory.finish()

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model.simulate(simulation_timeline)

    print(f"Finished {scen_id}")