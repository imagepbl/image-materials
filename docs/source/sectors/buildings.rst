#########
Buildings
#########

Buildings preprocessing transforms IMAGE floorspace and population data into the harmonized
inputs required by the dynamic stock and materials models. The workflow is orchestrated by
`buildings.preprocessing.main.buildings_preprocessing <../../../imagematerials/buildings/preprocessing/main.py>`_
and combines IMAGE scenario floorspace/population output, standard (fixed) building
assumptions, and circular-economy settings.

Scope
=====

The buildings sector covers **residential** buildings (Detached, Semi-detached, Apartment,
High-rise) and **commercial** (non-residential) buildings (Office, Retail+, Hotels+, Govt+).
Residential floorspace is further resolved by urban/rural area and by income quintile
(Q1-Q5), since floorspace per capita and housing-type shares vary strongly with income.

The regional coverage includes all 26 IMAGE regions. The time horizon extends from a
historic reconstruction starting in 1721 through IMAGE's simulation period to 2100, with
annual time steps — the long historic tail is needed so that the age structure of the
existing building stock is realistic at the start of the simulation period.

Processes covered by the buildings module are production (new construction) and end-of-life
outflow of buildings; maintenance materials are not yet modelled for this sector.

Preprocessing
=============

Data Input
----------

* `IMAGE EnergyServices output <../../../data/image/SSP2_baseline/EnergyServices>`_: residential
  floorspace by urban/rural area and income quintile (``res_FloorSpace.out``), used as the basis
  for residential floorspace.
* `IMAGE Socioeconomic output <../../../data/image/SSP2_baseline/Socioeconomic>`_: population by
  region, area and quintile (``Pop_q.out``) and service value added per capita
  (``sva_pc.scn``, the driver for commercial floorspace demand).
* `buildings/<scenario> <../../../data/raw/buildings/>`_: scenario-dependent inputs, including
  ``files_commercial`` (Gompertz curve parameters for commercial floorspace demand),
  ``files_lifetimes/<scenario>`` (Weibull/folded-normal lifetime parameters), and
  ``files_DB`` (average m²/capita by housing type, housing-type shares).
* `buildings/standard_data <../../../data/raw/buildings/standard_data/>`_: fixed assumptions,
  including historic population (used to anchor the pre-1971 population trajectory) and
  building material intensity tables (``Building_materials_rasmi.csv`` for residential,
  ``materials_commercial_rasmi.csv`` for commercial).
* `circular economy config <../../../data/raw/circular_economy_scenarios/>`_: optional
  adjustments for floorspace per capita (``base``, ``narrow``, ``narrow_activity``), lifetime
  extension (``slow``), and material lightweighting (``narrow_product``, ``resource_efficient``).

Structure
---------

The preprocessing follows the modular structure in
`imagematerials.buildings.preprocessing <../api/buildings.html>`_:

The main output ``stocks`` (floorspace) is calculated using two files:

* `floorspace.py <../../../imagematerials/buildings/preprocessing/floorspace.py>`_ — `floorspace API <../api/buildings.html>`_

  - Computes commercial floorspace per capita from service value added using fitted Gompertz
    curves, split across the four commercial subtypes (``compute_commercial_floor_m2_cap_sum``,
    ``compute_commercial_floor_m2_cap``)
  - Loads IMAGE residential floorspace per capita by urban/rural area and quintile
    (``get_floorspace_urban_rural``)
  - Extrapolates both residential and commercial floorspace back to 1721 using the average
    trend over IMAGE's first data decade, floored at a regional minimum
    (``extrapolate_floorspace``)
  - Combines average floorspace per capita, housing-type shares and population into total
    residential floorspace by housing type (``compute_housing_residential``,
    ``compute_housing_type``, ``compute_average_m2_capita``)
  - Applies circular economy floorspace-per-capita adjustments (``base``, ``narrow``,
    ``narrow_activity``) via calls into ``circular_economy_measures.py``

* `population.py <../../../imagematerials/buildings/preprocessing/population.py>`_ — `population API <../api/buildings.html>`_

  - Builds a continuous total-population series (``compute_population``) by combining
    historic population data (pre-1971) with IMAGE population output, avoiding an artificial
    jump in stock at the 1971 model start
  - Splits total population into urban/rural and further into income quintiles, using IMAGE
    quintile shares interpolated across the full timeline
  - Applies a linear historic transition of the urban share between 1700 and 1971, since
    IMAGE quintile splits are not defined before 1971

Lifetimes are calculated using:

* `lifetimes.py <../../../imagematerials/buildings/preprocessing/lifetimes.py>`_ — `lifetimes API <../api/buildings.html>`_

  - Reads Weibull (or optionally folded-normal) shape/scale parameters for residential and
    commercial buildings by region, area and building type, and interpolates them across the
    full time range
  - Converts to the ``(c, scale, loc)`` parameterization used by :func:`imagematerials.lifetimes.lifetimes_to_matrix`
  - Switches to the ``SSP2_2D_RE`` lifetime dataset when the ``slow`` circular economy scenario
    is active (lifetime extension)

Material intensities are calculated using:

* `materials.py <../../../imagematerials/buildings/preprocessing/materials.py>`_ — `materials API <../api/buildings.html>`_

  - Interpolates residential and commercial material-intensity tables (kg/m²) across cohorts,
    by region and building type
  - Applies a China-specific override for concrete intensity in the ``P100`` archetype
  - Switches to the resource-efficient material intensity dataset when the
    ``resource_efficient`` circular economy scenario is active
  - Applies circular economy lightweighting (``narrow_product``) via calls into
    ``circular_economy_measures.py``

Circular economy adjustments are centralized in:

* `circular_economy_measures.py <../../../imagematerials/buildings/preprocessing/circular_economy_measures.py>`_ — `circular economy API <../api/buildings.html>`_

  - See `CE measures calculations`_ below for details on each scenario.

Key Assumptions
---------------

- Commercial floorspace per capita is modelled as a saturating Gompertz function of service
  value added per capita, fitted separately for each of the four commercial subtypes.
- Residential floorspace is split into housing types (Detached, Semi-detached, Apartment,
  High-rise) on an m² basis using region-, area- and quintile-independent average m²/capita
  and housing-type-share assumptions.
- Historic floorspace (1721-1970) is extrapolated from the average annual growth trend across
  regions in IMAGE's first available decade (1971-1981), floored at each region's observed
  minimum (or an explicit minimum for commercial subtypes).
- Population before 1971 is taken from an external historic dataset and blended with IMAGE
  population from 1971 onward to avoid a discontinuity in stock build-up at model start.

Output
------

The ``buildings_preprocessing`` function returns a dictionary with model-ready objects used
by the sector models:

* ``stocks`` — floorspace, dims ``(time, Region, Type, Quintile)``, unit m²
* ``lifetimes`` — dict of scipy-parameter DataArrays (Weibull or folded-normal)
* ``material_intensities`` — dims ``(Cohort, Region, Type, material)``, unit kg/m²
* ``knowledge_graph``
* ``set_unit_flexible`` (unit of the floorspace stock, e.g. ``m^2``)

Simulation
----------

The buildings module uses the `StocksQuintiles class <../api/models_detail.html>`_ for stock
modelling and the `MaterialIntensitiesQuintiles class <../api/models_detail.html>`_ for
calculating material flows in simulation — the quintile-aware variants of
:class:`imagematerials.model.GenericStocks` and :class:`imagematerials.model.GenericMaterials`,
since residential floorspace (and therefore stock and material demand) is resolved by income
quintile in addition to Region, Type and Cohort. The added ``Quintile`` dimension is summed
away before results enter cross-sector aggregation (e.g. end-of-life reporting), since
downstream consumers track flows without the quintile split. 

CE measures calculations
=========================

To be added when new CE implementation is finalized

References
==========

- **RASMI dataset (material intensities).** Fishman, T., et al. RASMI: Global Building
  Material Intensity Database. Available via Zenodo.
- **IRP "Bend the Trend".** International Resource Panel (2024). Resource efficiency and
  climate change: material efficiency strategies for a low-carbon future — used as a
  reference for lightweighting and sufficiency penetration rates in the ``narrow_product``
  and ``narrow_activity`` scenarios.
- **ODYM-RECC lightweighting dataset.** Pauliuk, S., et al. Material efficiency strategies
  for buildings, dataset ``3_SHA_LightWeighting_Buildings_V2.2.xlsx``. Available via
  Zenodo: https://zenodo.org/records/4671644.
