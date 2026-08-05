########
Vehicles
########

Vehicle preprocessing transforms transport activity data and technology assumptions into the harmonized inputs required by the dynamic stock and
materials models. The workflow is orchestrated by `vehicles.preprocessing.main.vehicles_preprocessing <../../../imagematerials/vehicles/preprocessing/main.py>`_ and combines scenario,
standard, socio economic scenario data from IMAGE, and circular-economy settings. 

Scope
----------
The vechicle sector includes passenger transport (cars, regular and midi buses, rail and high speed rail, passenger planes) and freight transport (light commercial vehicles, medium and heavy trucks, inland and international shipping). 
For cars, trucks, and busses, there is a subsplit into engine types (ICE, BEV, FCEV, PHEV). For international shipping, there is a split into ship types (small, medium, large, very large). 

The regional coverage includes all IMAGE regions. The time horizon extends from 1995 to 2100, with annual time steps.

Processes covered by the vehicle module are production of the vehicles and maintenance for a sletec number of transport modes (cars, buses, trains, high-speed trains, light commercial vehicles, trucks).


Preprocessing 
==================
The vehicle preprocessing transforms raw input data into a harmonized into a harmonized format for the dynamic stock and materials models. The workflow is orchestrated by `vehicles.preprocessing.main.vehicles_preprocessing <../../../imagematerials/vehicles/preprocessing/main.py>`_ and combines scenario, standard, climate-policy, and circular-economy settings.

Data Input 
----------

* `vehicles/standard_data <../../../data/raw/vehicles/standard_data/>`_: fixed assumptions (loads, maintenance, first-year operation, load passenger versus freight, ship parameters)
* `vehicles/<scenario> <../../../data/raw/vehicles/>`_: scenario-dependent inputs (weights, lifetimes, material fractions, kilometrage)
* `socio-economic scenariofolder <../../../data/image/SSP2_baseline/EnergyServices>`_: IMAGE transport activity and technologyshare trajectories used to derive regional stocks and subtype splits
* `circular economy config <../../../data/raw/circular_economy_scenarios/>`_: optional adjustments for lifetime extension,	lightweighting, and increased intensity of use (e.g. kilometrage)

Structure
----------

The preprocessing follows the modular structure in
`imagematerials.vehicles.preprocessing <../api/vehicles.html>`_:

The main output 'stocks' is calculated using two files where shares is used by stocks to apply technology split where needed.

* `stocks.py <../../../imagematerials/vehicles/preprocessing/stocks.py>`_ — `get_vehicle_stocks API <../api/vehicles.html>`_

  - Converts passenger-km and tonne-km demand into vehicle counts by type and region
  - Includes specific handling for trucks (LCV split) and shipping
  - Applies circular economy utilization changes (``narrow_product`` mileage)
  - Uses subtype shares to split aggregate stocks and maps to IMAGE regions

* `shares.py <../../../imagematerials/vehicles/preprocessing/shares.py>`_ — `get_vehicle_shares API <../api/vehicles.html>`_

  - Constructs drivetrain shares for typical road vehicles (cars, buses, freight classes)
  - Harmonizes shares across regions and years and rebroadcasts them to the vehicle knowledge graph

The output 'weights' is calculated using:

* `weights.py <../../../imagematerials/vehicles/preprocessing/weights.py>`_ — `get_weights API <../api/vehicles.html>`_

  - Processes simple and subtype-specific vehicle weights over cohorts
  - Applies lightweighting trajectories for circular economy scenarios (``narrow``, ``narrow_product``, ``resource_efficient``)
  - Adds ship weights from dedicated ship assumptions

Material and maintenance material fractions are calculated using:

* `materials.py <../../../imagematerials/vehicles/preprocessing/materials.py>`_ — `materials API <../api/vehicles.html>`_

  - Builds maintenance material coefficients (kg material per kg vehicle per year) and adds default values
  - Processes simple and subtype-specific material fraction trajectories

Lifetimes and other crosscutting functions are calculated using:

* `util.py <../../../imagematerials/vehicles/preprocessing/util.py>`_ — `utility preprocessing API <../api/vehicles.html>`_

  - Reads and standardizes passenger-km and tonne-km inputs
  - Interpolates time series and converts tabular data to xarray-based model dimensions
  - Applies lifetime changes for circular economy ``slow`` scenarios

Key Assumptions
----------------------
- all standard data is not region-specific, but applied globally 


Output
----------

The ``vehicles_preprocessing`` returns a dictionary with model-ready
objects used by the sector models:

* ``knowledge_graph``
* ``lifetimes``
* ``maintenance_material_fractions``
* ``material_fractions``
* ``stocks``
* ``weights``
* ``set_unit_flexible`` (set to ``count``)

Simulation
----------
The vehicle module uses the `GenericStocks class <../api/models_detail.html>`_ for stock modelling and the `GenericMaterials class <../api/models_detail.html>`_ for calculating material flows in simulation.

References
==================
Deetman, S. (2021). Stock-driven scenarios on global material demand: The story of a lifetime. https://hdl.handle.net/1887/3245696

Von Köckritz, L., Edelenbosch, O., Deetman, S., Arp, F., Brouwer, R., Schram, R., Zanon-Zotin, M., & Van Vuuren, D. (2026). Old is gold? Vehicle maintenance material demand of lifetime extension: dynamic stock modelling. Resources, Conservation and Recycling, 228, 108752. https://doi.org/10.1016/j.resconrec.2025.108752