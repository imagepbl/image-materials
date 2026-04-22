Vehicles
========

Vehicle preprocessing transforms transport activity data and technology assumptions into the harmonized inputs required by the dynamic stock and
materials models. The workflow is orchestrated by `vehicles.preprocessing.main.vehicles_preprocessing <../../../imagematerials/vehicles/preprocessing/main.py>`_ and combines scenario,
standard, climate-policy, and circular-economy settings. 

Scope
------------------
The vechicle sector includes passnger transport (cars, buses, rail and high speed rail, passenger planes) and freight transport (light commercial vehicles, medium and heavy trucks, inland and international shipping). 

For cars, 

types, regions, processes

Data Input 
------------------

* `vehicles/standard_data <../../../data/raw/vehicles/standard_data/>`_: fixed assumptions (loads, maintenance, first-year operation, load passenger versus freight, ship parameters)
* `vehicles/<scenario> <../../../data/raw/vehicles/>`_: scenario-dependent inputs (weights, lifetimes, material fractions, kilometrage)
* climate policy folder: IMAGE transport activity and technologyshare trajectories used to derive regional stocks and subtype splits
* circular economy config: optional adjustments for lifetime extension,	lightweighting, and increased intensity of use (e.g. kilometrage)

Preprocessing Structure
-----------------------

The preprocessing pipeline follows the modular structure in
``imagematerials.vehicles.preprocessing``:

* `util.py <../../../imagematerials/vehicles/preprocessing/util.py>`_
	- Reads and standardizes passenger-km and tonne-km inputs
	- Interpolates time series and converts tabular data to xarray-based model dimensions
	- Applies lifetime changes for circular economy ``slow`` scenarios
* `materials.py <../../../imagematerials/vehicles/preprocessing/materials.py>`_
	- Builds maintenance material coefficients (kg material per kg vehicle per year) and adds default values
	- Processes simple and subtype-specific material fraction trajectories
* `shares.py <../../../imagematerials/vehicles/preprocessing/shares.py>`_
	- Constructs drivetrain shares for typical road vehicles (cars, buses, freight classes)
	- Harmonizes shares across regions and years and rebroadcasts them to the vehicle knowledge graph
* `stocks.py <../../../imagematerials/vehicles/preprocessing/stocks.py>`_
	- Converts passenger-km and tonne-km demand into vehicle counts by type and	region
	- Includes specific handling for trucks (LCV split) and shipping
	- Applies circular economy utilization changes (``narrow_product`` mileage)
	- Uses subtype shares to split aggregate stocks and maps to IMAGE regions
* `weights.py <../../../imagematerials/vehicles/preprocessing/weights.py>`_
	- Processes simple and subtype-specific vehicle weights over cohorts
	- Applies lightweighting trajectories for circular economy scenarios (``narrow``, ``narrow_product``, ``resource_efficient``)
	- Adds ship weights from dedicated ship assumptions

Output
---------------

The ``vehicles_preprocessing`` entry point returns a dictionary with model-ready
objects used by the sector models:

* ``knowledge_graph``
* ``lifetimes``
* ``maintenance_material_fractions``
* ``material_fractions``
* ``stocks``
* ``weights``
* ``set_unit_flexible`` (set to ``count``)

Together, these outputs provide consistent dimensions (Type, SubType, Region, Time/Cohort), interpolated trajectories, and scenario-adjusted assumptions for vehicle stock and material flow calculations.

References
---------------