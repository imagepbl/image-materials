Resource Efficiency and Circular Economy
==========================================

IMAGE-materials includes an optional set of **resource efficiency and circular economy (CE) measures** that can be layered on top of the standard sector preprocessing. These measures adjust intermediate preprocessing results (e.g. floorspace demand, product lifetimes, material intensities, vehicle mileage, end-of-life recycling rates) before they are used to build the stock and material models.

This page describes the flag-based system that controls which measures are applied, how the underlying configuration is organized, and how to add a new measure.

Why a flag-based system?
-------------------------

The current system separates two concerns:

1. **Which measures are switched on** — a small set of booleans, one per measure ("flag").
2. **The parameters of each measure** — target years, regional percentage changes, material intensity changes, etc.

This allows any combination of measures to be enabled independently, while keeping all parameter data in one place.

The two configuration files
----------------------------

Every scenario directory under ``data/raw/circular_economy_scenarios/`` contains exactly two files:

``resource_efficiency_flags.toml``
    A flat list of booleans, one per measure, grouped by sector (``[buildings]``, ``[vehicles]``, ``[electricity]``, ``[end_of_life]``, ``[rest_of]``). This is the file that decides **whether** a measure runs.

``circular_economy_data.toml``
    The parameters for **every** measure, regardless of whether it is enabled. Each block is keyed as ``[sector.FlagName]`` (or a sub-key thereof), so a flag's data is always self-contained and easy to find, e.g.::

        [buildings.FlagLightweightingResidential]
        base_year = 2025
        target_year = 2060
        implementation_rate = "linear"

        [buildings.FlagLightweightingResidential.material_intensity_change.steel]
        "Central Europe" = -8.5
        ...

Both files are read together, and a flag has no effect unless it is both ``true`` in the flags file **and** its corresponding block is present in the data file.

Loading the configuration
--------------------------

The two files are read via :func:`imagematerials.util.read_resource_efficiency_flags` and :func:`imagematerials.util.read_circular_economy_data`, and passed into ``get_preprocessing_data`` as ``resource_efficiency_flags_file`` and ``circular_economy_data_file``:

.. code-block:: python

       from pathlib import Path
       from imagematerials.preprocessing import get_preprocessing_data

       ce_scenario_path = Path("data", "raw", "circular_economy_scenarios", "slow")

       bld_sector = get_preprocessing_data(
           "buildings", Path("data", "raw"),
           climate_policy_scenario_dir,
           circular_economy_data_file=ce_scenario_path,
           resource_efficiency_flags_file=ce_scenario_path,
       )

Both arguments accept either a directory (containing a file with the expected name) or a direct path to the ``.toml`` file. Passing ``None`` for either disables all resource efficiency and circular economy measures.

Selecting a scenario is optional
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You do **not** have to select a resource efficiency / circular economy scenario to run the model. ``circular_economy_data_file=None`` and ``resource_efficiency_flags_file=None`` (the defaults) are a perfectly valid "no measures" configuration — every flag is treated as disabled. For a standard, non-CE run, the recommended and equivalent alternative is to point both arguments at the ``base`` preset, which is an explicit ``resource_efficiency_flags.toml`` with every flag set to ``false``:

.. code-block:: python

       ce_scenario_path = Path("data", "raw", "circular_economy_scenarios", "base")

       bld_sector = get_preprocessing_data(
           "buildings", Path("data", "raw"),
           climate_policy_scenario_dir,
           circular_economy_data_file=ce_scenario_path,
           resource_efficiency_flags_file=ce_scenario_path,
       )

Using ``"base"`` explicitly (rather than relying on ``None``) makes the scenario choice visible in the notebook/script, and keeps the same code path used for every other named scenario. The helper :func:`imagematerials.util.resolve_circular_economy_scenario` makes it easy to switch between named scenarios (including ``None``) by name, as used in the example notebooks:

.. code-block:: python

       from imagematerials.util import resolve_circular_economy_scenario

       ce_scenarios_base_path = Path("data", "raw", "circular_economy_scenarios")
       circular_economy_scenario_dir = resolve_circular_economy_scenario(
           ce_scenarios_base_path, "base")  # or "full_ce", "slow", ..., or None

.. warning::

   ``resource_efficiency_flags_file`` and ``circular_economy_data_file`` must point at configurations that agree with each other. If a flag is ``true`` but its sector/measure block is missing from the data file (for example because ``circular_economy_data_file=None`` was passed while the flags file still has flags set to ``true``), preprocessing will raise a ``KeyError`` for the missing block. Prefer pointing both arguments at the same scenario directory (or both at ``None``) rather than mixing sources.

How a flag is checked in code
------------------------------

Inside each sector's preprocessing module, :func:`imagematerials.util.flag_enabled` is used to check whether a given measure is switched on, and the corresponding block of ``circular_economy_config`` supplies its parameters:

.. code-block:: python

       from imagematerials.util import flag_enabled

       if flag_enabled(resource_efficiency_flags, "buildings", "FlagLightweightingResidential"):
           flag_config = circular_economy_config["buildings"]["FlagLightweightingResidential"]
           target_year = flag_config["target_year"]
           ...

``flag_enabled`` returns ``False`` (rather than raising) if ``resource_efficiency_flags`` is ``None`` or the flag/sector is missing, so measures are opt-in by default.

Mutually exclusive flags
--------------------------

Some flag pairs must never both be enabled at the same time, because the second measure would be applied on top of the (already adjusted) output of the first rather than on the base IMAGE trajectory it assumes. Currently this applies to:

- ``FlagFloorSpaceCalibrationResidential`` and ``FlagFloorSpaceReductionResidential``
- ``FlagFloorSpaceCalibrationCommercial`` and ``FlagFloorSpaceReductionCommercial``

This is enforced automatically: :func:`imagematerials.util.read_resource_efficiency_flags` calls :func:`imagematerials.util.validate_resource_efficiency_flags` on every flags file it reads, and raises a ``ValueError`` if a mutually-exclusive pair is both ``True``. The pairs are defined in ``imagematerials.util.MUTUALLY_EXCLUSIVE_FLAGS``; add new pairs there if a future measure has the same kind of conflict.

Provided scenario presets
---------------------------

``data/raw/circular_economy_scenarios/`` ships with several ready-made presets, each a directory containing matching ``resource_efficiency_flags.toml`` and ``circular_economy_data.toml`` files:

- ``base`` — every flag ``false``; no resource efficiency or circular economy measures applied.
- ``full_ce`` — every applicable flag ``true``; all compatible measures applied simultaneously. Note that ``FlagFloorSpaceCalibration*`` and ``FlagFloorSpaceReduction*`` are mutually exclusive (see below), so ``full_ce`` enables the reduction flags, not the calibration ones.
- ``narrow_product`` — lightweighting measures only (building material intensities, vehicle weight, generation/grid component weight).
- ``narrow_activity`` — floorspace/activity reduction measures only (residential and commercial floorspace).
- ``slow`` — lifetime extension measures only (buildings, vehicles, electricity generation/grid, and the matching end-of-life reuse rates).
- ``close`` — end-of-life recycling rate measures only.

These four presets reproduce the behaviour of the legacy ``narrow_product``, ``narrow_activity``, ``slow`` and ``close`` scenario folders from before the flag-based refactor, expressed in the new format. Combinations beyond these presets can be created by copying a ``resource_efficiency_flags.toml`` and toggling the flags you need — the ``circular_economy_data.toml`` file can usually be reused unchanged, since it defines parameters for every measure.

Where the measures are implemented
------------------------------------

Each sector has its own ``circular_economy_measures.py`` module under its ``preprocessing`` package, e.g.:

- :mod:`imagematerials.buildings.preprocessing.circular_economy_measures`
- :mod:`imagematerials.electricity.preprocessing.circular_economy_measures`
- vehicle and end-of-life measures live directly in ``imagematerials/vehicles/preprocessing/`` and ``imagematerials/eol/preprocessing.py`` respectively.

These functions are called from the sector's main preprocessing entry point (e.g. ``imagematerials/buildings/preprocessing/main.py``, ``floorspace.py``) after the "base" (non-circular) values have been computed, and adjust those values in place before they are returned as preprocessing output.

Adding a new measure
-----------------------

To add a new resource efficiency or circular economy measure to an existing sector:

1. Add a new boolean flag to the relevant sector block in every ``resource_efficiency_flags.toml`` you use (at minimum ``base`` and ``full_ce``), defaulting to ``false``.
2. Add a ``[sector.FlagYourNewMeasure]`` block with its parameters to ``circular_economy_data.toml``.
3. In the sector's ``circular_economy_measures.py``, guard your new logic with ``flag_enabled(resource_efficiency_flags, "sector", "FlagYourNewMeasure")`` and read parameters from ``circular_economy_config["sector"]["FlagYourNewMeasure"]``.
4. Make sure the function that implements the measure is actually called from the sector's preprocessing pipeline — a flag that is checked but never reached because its enclosing function isn't invoked will silently have no effect.

.. note::

   Some measures depend on data or intermediate arrays only being available at specific points in the preprocessing pipeline (for example, floorspace reduction measures need the base per-capita floorspace to already be computed). When adding a new measure, check where in the pipeline comparable existing measures are applied to reuse the same intermediate outputs, and be careful with array dimensions: several intermediate arrays (e.g. residential floorspace) carry extra dimensions such as ``Quintile`` that need to be handled explicitly (e.g. averaged or selected) rather than assumed away.
