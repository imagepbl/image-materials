from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import prism
import warnings
import numpy as np
import time
import argparse
import pickle

from imagematerials.concepts import create_electricity_graph
from imagematerials.factory import ModelFactory
from imagematerials.model import GenericStocks, SharesInflowStocks, MaterialIntensities, ElectricVehicleBatteries
from imagematerials.preprocessing import get_preprocessing_data

from imagematerials.sensitivity_analysis.changedata import change_sector, ChangeAction, ChangeReplace, ChangeRename, ChangeDelete
from imagematerials.sensitivity_analysis.monte_carlo import sample_values, load_material_intensities, load_lifetimes, process_material_intensities

from imagematerials.vehicles.constants import vehicles_modes_sensitivity_analysis, EV_BATTERY_TYPES
from imagematerials.electricity.constants import EPG_TECHNOLOGIES_FINAL, TECH_STATIONARY_STORAGE
from imagematerials.electricity.utils import rebroadcast_type_dims
warnings.filterwarnings("ignore")

knowledge_graph_electricity = create_electricity_graph()

path_current = Path().resolve()
path_base = path_current # base path of the project -> image-materials
path_data = Path(path_base, "data", "raw")

path_mc_output = Path(path_current.parent, "analysis", "sensitivity", "data")



# ----------------------------------------------------------------------------------------------
# Command-line arguments
#
# Lets this script be launched as several independent OS processes (e.g. from separate terminal
# sessions) that together cover --n-total Monte Carlo runs, while staying reproducible for a
# given --seed / --n-processes combination.
#
# Example: 2 parallel processes covering 100 runs in total, run in two terminals:
#   python sensitivity.py --seed 42 --n-processes 2 --process-index 0 --n-total 100
#   python sensitivity.py --seed 42 --n-processes 2 --process-index 1 --n-total 100

# ----------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Monte Carlo sensitivity analysis (parallelizable across processes).")
parser.add_argument("--seed", type=int, default=42, help="base seed shared by all processes")
parser.add_argument("--n-processes", type=int, default=1, help="total number of parallel processes being run")
parser.add_argument("--process-index", type=int, default=0, help="index (0-based) of THIS process, must be < n-processes")
parser.add_argument("--n-total", type=int, default=10, help="total number of Monte Carlo runs across ALL processes combined")
parser.add_argument("--output-dir", type=Path, default=path_mc_output, help="directory to write pickled results to") #Path("mc_output")
parser.add_argument("--checkpoint-every", type=int, default=10, help="write a partial-results checkpoint pickle every N completed runs in this process (0 = only save once, at the end)")

cli_args = parser.parse_args()

if cli_args.process_index >= cli_args.n_processes:
    raise ValueError(f"--process-index ({cli_args.process_index}) must be smaller than --n-processes ({cli_args.n_processes})")

cli_args.output_dir.mkdir(parents=True, exist_ok=True)

def iteration_range(n_total, n_processes, process_index):
    """Split range(n_total) into n_processes contiguous, near-equal chunks and return the
    (start, end) indices [start, end) handled by `process_index`. Remainder iterations (if
    n_total doesn't divide evenly) are spread over the first chunks, so the chunks always
    sum to exactly n_total.

    """
    base, remainder = divmod(n_total, n_processes)
    chunk_sizes = [base + 1 if i < remainder else base for i in range(n_processes)]
    starts = np.cumsum([0] + chunk_sizes)

    return int(starts[process_index]), int(starts[process_index + 1])

iter_start, iter_end = iteration_range(cli_args.n_total, cli_args.n_processes, cli_args.process_index)


####################################################################################################
# Initialize sectors 

# vehicles -----------------------------------------------------------------------------------------

# Get the preprocessing data for the vehicles sector only once
vhc_sector = get_preprocessing_data(
    "vehicles", path_data,
    climate_policy_scenario_dir = Path(path_data, "image", "SSP2_baseline"), 
    circular_economy_scenario_dirs = None
)
# sensitivity analysis currently only implemented for road vehicle types (due to lack of data): 
# select those from the preprocessing data
stocks = vhc_sector.all_data["stocks"]
stocks = stocks.sel(Type = vehicles_modes_sensitivity_analysis)

# load material intensities and use them to replace the material fractions in the preprocessing data
# use the standard values (column "values") - needs to be renamed to "sampled" though to be able to 
# use the function process_material_intensities() to convert the data to the correct format for the model
mi = load_material_intensities(path_data / "vehicles" / "standard_data" / "vehicles_material_intensities_long.csv", sector="vehicles")
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

# EV batteries -------------------------------------------------------------------------------------

ev_battery_sector = get_preprocessing_data(
    "ev_battery", path_data,
    climate_policy_scenario_dir = Path(path_data, "image", "SSP2_baseline"),
    circular_economy_scenario_dirs = None
)

# electricity - generation -------------------------------------------------------------------------
elc_sector = get_preprocessing_data(
    "electricity", path_data,
    climate_policy_scenario_dir = Path(path_data, "image", "SSP2_baseline"),
    circular_economy_scenario_dirs = None
)
# elc_sector is a list of preprocessing data for each electricity subsector

elc_sector_generation = next(item for item in elc_sector if item.name == "elc_gen")
elc_sector_storage_phs = next(item for item in elc_sector if item.name == "elc_stor_phs")
elc_sector_storage_other = next(item for item in elc_sector if item.name == "elc_stor_other")


####################################################################################################
# Run Monte-Carlo

time_start = 1998
time_end = 2100
complete_timeline = prism.Timeline(time_start, time_end, 1)
simulation_timeline = prism.Timeline(time_start, time_end, 1)

ranges_vehicles = load_material_intensities(path_data / "vehicles" / "standard_data" / "vehicles_material_intensities_long.csv", "vehicles")
ranges_generation = load_material_intensities(path_data / "electricity" / "standard_data" / "generation_material_intensities_long.csv", "generation")
ranges_storage = load_material_intensities(path_data / "electricity" / "standard_data" / "storage_and_ev_battery_material_intensities_long.csv", "storage_other")
ranges_storage_other = ranges_storage[ranges_storage["Type"].isin(TECH_STATIONARY_STORAGE)]
ranges_storage_phs = ranges_storage[ranges_storage["Type"].isin(["PHS"])]
ranges_ev_battery = ranges_storage[ranges_storage["Type"].isin(EV_BATTERY_TYPES)]

rng = np.random.default_rng(42)   # seed once for reproducibility


# One independent, reproducible child stream PER MONTE CARLO ITERATION (not per process).
# child_seeds depends only on --seed and --n-total, never on how many processes are splitting
# the work. So iteration i always draws the exact same random numbers, whether it's iteration 37
# of a single sequential --n-processes 1, or iteration 37 handled by process 2 of an --n-processes 8.
# This is what makes the parallel output combinable into something identical to a single one-go run: 
# just union the per-process pickles by iteration.
seed_sequence = np.random.SeedSequence(cli_args.seed)
child_seeds = seed_sequence.spawn(cli_args.n_total)

start = time.time()
all_output = {}
# N = 2
for i in range(iter_start, iter_end): # N
    rng = np.random.default_rng(child_seeds[i])

    # vehicles ------------------------------------------------------------
    mi = sample_values(ranges_vehicles, rng=rng)
    mi = process_material_intensities(mi, "vehicles")
    change_definition_vehicles = {
        "material_intensities": ChangeReplace(mi),
    }
    new_vhc_sector = change_sector(vhc_sector, change_definition_vehicles, inplace=False)

    # EV - battery ------------------------------------------------------------
    mi = sample_values(ranges_ev_battery, rng=rng)
    mi = process_material_intensities(mi, "ev_battery").rename({'Type': 'BatteryType'})
    change_definition_ev_battery = {
        "material_intensities": ChangeReplace(mi),
    }
    new_ev_battery_sector = change_sector(ev_battery_sector, change_definition_ev_battery, inplace=False)

    # electricity - generation --------------------------------------------
    mi = sample_values(ranges_generation, rng=rng)
    mi = process_material_intensities(mi, "generation")
    
    change_definition_generation = {
        "material_intensities": ChangeReplace(mi),
    }
    new_elc_sector_generation = change_sector(elc_sector_generation, change_definition_generation, inplace=False)

    # electricity - storage - other --------------------------------------------
    mi = sample_values(ranges_storage_other, rng=rng)
    mi = process_material_intensities(mi, "storage_other")
    
    change_definition_storage_other = {
        "material_intensities": ChangeReplace(mi),
    }
    new_elc_sector_storage_other = change_sector(elc_sector_storage_other, change_definition_storage_other, inplace=False)

    # electricity - storage - PHS --------------------------------------------
    mi = sample_values(ranges_storage_phs, rng=rng)
    mi = process_material_intensities(mi, "storage_phs")
    
    change_definition_storage_phs = {
        "material_intensities": ChangeReplace(mi),
    }
    new_elc_sector_storage_phs = change_sector(elc_sector_storage_phs, change_definition_storage_phs, inplace=False)

    # model run ------------------------------------------------------------
    factory = ModelFactory(
        [new_vhc_sector,
         new_ev_battery_sector,
         new_elc_sector_generation,
         new_elc_sector_storage_phs,
         new_elc_sector_storage_other], 
         complete_timeline
        ).add(GenericStocks, ["vehicles", "elc_stor_phs"]
        ).add(SharesInflowStocks, ["elc_gen", "elc_stor_other"]
        ).add(ElectricVehicleBatteries, ["ev_battery"], input_sources={
                "stock_by_cohort": "vehicles",
                "inflow": "vehicles",
                "outflow_by_cohort": "vehicles"}
        ).add(MaterialIntensities, ["vehicles",
                                    "elc_gen",
                                    "elc_stor_phs",
                                    "elc_stor_other"]
        )
    model = factory.finish()
    model.simulate(simulation_timeline)

    # renaming Type coordinates (necessary due to work around within the sub-technology model for electricity generation)
    rebroadcast_type_dims(model.elc_gen, knowledge_graph_electricity, EPG_TECHNOLOGIES_FINAL)
    
    all_output[i] = {
            # MATERIAL inflows
            "inflow_materials": [model.vehicles["inflow_materials"], model.ev_battery["inflow_battery_materials"], model.elc_gen["inflow_materials"], model.elc_stor_phs["inflow_materials"], model.elc_stor_other["inflow_materials"]],
            # MATERIAL stocks
            "stock_by_cohort_materials": [model.vehicles["stock_by_cohort_materials"], model.ev_battery["stock_battery_materials"], model.elc_gen["stock_by_cohort_materials"], model.elc_stor_phs["stock_by_cohort_materials"], model.elc_stor_other["stock_by_cohort_materials"]], 
            # MATERIAL outflows
            "outflow_by_cohort_materials": [model.vehicles["outflow_by_cohort_materials"], model.ev_battery["outflow_battery_materials"], model.elc_gen["outflow_by_cohort_materials"], model.elc_stor_phs["outflow_by_cohort_materials"], model.elc_stor_other["outflow_by_cohort_materials"]],
        }
    print(f"\rSimulation {i} completed.     ", end="")

    if cli_args.checkpoint_every and (i - iter_start + 1) % cli_args.checkpoint_every == 0:
        checkpoint_path = cli_args.output_dir / f"mc_seed{cli_args.seed}_of{cli_args.n_processes}_part{cli_args.process_index}.partial.pkl"
        with open(checkpoint_path, "wb") as f:
            pickle.dump(all_output, f, protocol=pickle.HIGHEST_PROTOCOL)

end = time.time()
n_runs = iter_end - iter_start
print(f"\nTotal time for {n_runs} simulations (iterations {iter_start}-{iter_end - 1}): {(end - start)/60:.1f} minutes.")


output_path = cli_args.output_dir / f"mc_seed{cli_args.seed}_of{cli_args.n_processes}_part{cli_args.process_index}.pkl"
with open(output_path, "wb") as f:
    pickle.dump(all_output, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Saved results to {output_path}")
