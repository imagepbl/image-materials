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
  including historic population (used to anchor the pre-1971 population trajectory).
* `buildings/<scenario> <../../../data/raw/buildings/>`_: pre-built building material intensity
  tables (``Building_materials_rasmi.csv`` for residential,
  ``materials_commercial_rasmi_regionalized.csv`` for commercial). See
  `Material intensity sources`_ for how these are constructed.
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
  - Switches to the resource-efficient material intensity dataset when the
    ``resource_efficient`` circular economy scenario is active
  - Applies circular economy lightweighting (``narrow_product``) via calls into
    ``circular_economy_measures.py``

See `Material intensity sources`_ below for how the per-material MI tables are built.

Material intensity sources
--------------------------

The material-intensity (MI) tables consumed by the preprocessing are **pre-built** by
`data/raw/buildings/material_intensities/material_intensities.py <../../../data/raw/buildings/material_intensities/material_intensities.py>`_
(driven by the accompanying ``material_intensities.ipynb`` notebook) and written into every
scenario folder as:

* ``Building_materials_rasmi.csv`` — residential, indexed by (Year ∈ {2020, 2050}, Region 1-26,
  Building_type 1-4), one column per material
* ``Building_materials_rasmi_resource_efficient.csv`` — residential, resource-efficient variant
  (years {2020, 2030, 2050})
* ``materials_commercial_rasmi_regionalized.csv`` — commercial, indexed by (Year, Region, Material),
  one column per commercial type (selected in ``materials.py`` via
  ``USE_REGIONALIZED_COMMERCIAL_MI``)

Three data sources feed this build:

* **RASMI** (``MI_ranges_20230905.xlsx``) — an empirical global building MI database giving a
  percentile distribution (``p_0``-``p_100``) per material, per function (RS single-family,
  RM multi-family, NR non-residential), per structural system (C concrete, M masonry,
  S steel, T timber) and per RASMI region.
* **MaterialCities** (``MaterialCities_adm_gfa_ADM0.csv``) — gross floor area by country, split
  by function and structural system; aggregated to IMAGE regions and turned into per-region
  **structure-type shares** used to weight the RASMI structure dimension.
* **Deetman et al.** (``Building_materials_deetman.csv``, ``materials_commercial_old.csv``) — an
  older regionalised MI dataset, used only for aluminium.

Source and percentile per material:

.. list-table::
   :header-rows: 1
   :widths: 12 12 26 26 24

   * - Material
     - Source
     - Residential resolution
     - Commercial resolution
     - Time behaviour
   * - concrete
     - RASMI ``p_50``
     - per region & building type, structure-weighted
     - per region (NR function), structure-weighted
     - static; ``p_25`` in 2050 in the resource-efficient variant
   * - steel
     - RASMI ``p_50``
     - "
     - "
     - "
   * - wood
     - RASMI ``p_50``
     - "
     - "
     - "
   * - glass
     - RASMI ``p_50``
     - "
     - "
     - "
   * - brick
     - RASMI ``p_50``
     - "
     - "
     - "
   * - plastics
     - RASMI ``p_50``
     - "
     - "
     - "
   * - copper
     - RASMI ``p_75``
     - per region & building type, structure-weighted (``p_75``)
     - per region (NR function), structure-weighted (``p_75``)
     - static; no resource-efficiency reduction
   * - aluminium
     - Deetman × 0.64
     - Deetman regional value per building type
     - Deetman value per commercial type (not regionalised)
     - static; no resource-efficiency reduction

**Why copper and aluminium deviate from RASMI.** RASMI's imputation is only well constrained
where it has enough empirical records. The table below shows, per material, the number of
raw empirical datapoints behind the RASMI sheet and the **coefficient of variation (CV)** of
the ``p_50`` values across all 384 region/structure/function cells.

The CV is the standard deviation divided by the mean — a unitless measure of how much a
quantity varies relative to its typical size. A CV of ~0.5-1.0 means the median material
intensity genuinely differs from region to region and structure to structure (real signal).
A CV near zero means RASMI returns essentially the *same* value for every region, structure
and building type: its imputation has fallen back on a single global prior because there is
too little data to say anything region-specific.

.. list-table::
   :header-rows: 1
   :widths: 16 16 16 20

   * - Material
     - Raw datapoints
     - ``p_50`` CV
     - Interpretation
   * - concrete
     - 660
     - 0.49
     - well constrained
   * - wood
     - 611
     - 0.76
     - well constrained
   * - steel
     - 551
     - 0.90
     - well constrained
   * - glass
     - 364
     - 0.58
     - well constrained
   * - brick
     - 336
     - 0.99
     - constrained (high variance)
   * - plastics
     - 123
     - 0.012
     - degenerate — flat global prior (~1.2 kg/m²)
   * - aluminium
     - 93
     - 0.009
     - degenerate — flat global prior (~0.49 kg/m²)
   * - copper
     - 30
     - 0.008
     - degenerate — flat global prior (~0.18 kg/m²)

For concrete, steel, wood, glass and brick the RASMI ``p_50`` carries real regional and
typological signal and is used directly. For aluminium, copper and plastics the ``p_50`` is
a near-constant prior; used as-is it puts the building aluminium and copper stocks far below
the material-flow literature.

* **Copper** is taken from RASMI's ``p_75`` (~0.27 kg/m²) rather than the degenerate ``p_50`` —
  still on the RASMI methodology, but away from the collapsed median. Because ``p_25``/``p_50``
  are uninformative, no resource-efficiency reduction is applied to copper.
* **Aluminium** is taken from the older Deetman et al. dataset instead. The raw Deetman
  intensities are themselves on the high side (they put the aluminium stock ~1.5-1.6× above
  independent estimates), so a single global calibration factor
  (``DEETMAN_ALUMINIUM_CALIBRATION = 0.64``) is applied to every Deetman aluminium value. It
  is fitted to two benchmarks — USA buildings 2009 ≈ 155 kg/capita and Europe (WEU+CEU) 2013
  ≈ 49 Mt — which the calibrated stock reproduces to within ~3 %. Deetman's semi-detached
  (type 2) aluminium is a placeholder (one value, flat across all 26 regions), so it is taken
  from detached (type 1); the other three residential types keep their own regionalised
  Deetman values.
* **Plastics** is also thin in RASMI (~123 datapoints, near-constant ``p_50`` ≈ 1.2 kg/m²) and
  is currently kept on RASMI as-is — a known limitation.

**Structure weighting.** For every material taken from RASMI, ``weighted_structure_mi``
collapses the C/M/S/T structural dimension into a single per-region, per-type value by
taking a GFA-share-weighted average using the MaterialCities structure shares for that
region, renormalised over the structural systems permitted for that building type
(``housing_type_to_rasmi_building_structure``; e.g. apartments and high-rises are assumed
concrete or steel only). Regions with no MaterialCities coverage fall back to an unweighted
mean.

**Region and type mapping.** IMAGE's 26 regions map to RASMI's 32 regions via
``image_to_rasmi`` (one-to-many, e.g. IMAGE region 11 aggregates four European RASMI
regions). Residential building types 1/2 map to RASMI function RS and 3/4 to RM. RASMI
resolves only a single non-residential function (NR), so all four commercial types receive
the same per-region NR intensity.

Final MI ranges (2020, kg/m²) produced by this build:

.. list-table::
   :header-rows: 1
   :widths: 16 20 20

   * - Material
     - Residential
     - Commercial
   * - concrete
     - 321 - 1119
     - 540 - 1134
   * - steel
     - 2.4 - 60.0
     - 34.4 - 104.9
   * - brick
     - 68.6 - 625.3
     - 75.4 - 590.7
   * - wood
     - 18.3 - 70.5
     - 9.2 - 25.6
   * - glass
     - 1.5 - 5.7
     - 1.2 - 2.5
   * - aluminium
     - 0.8 - 4.4
     - 1.5 - 3.7
   * - plastics
     - 1.0 - 1.2
     - 1.2 - 1.2
   * - copper
     - 0.27
     - 0.27

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
  Material Intensity Database. Available via Zenodo. Primary MI source for concrete, steel,
  wood, glass, brick, plastics (``p_50``) and copper (``p_75``).
- **Deetman et al. (aluminium material intensities).** Deetman, S., et al. (2020). Modelling
  global material stocks and flows for residential and service sector buildings towards 2050.
  *Journal of Cleaner Production* 245, 118658. Source for the aluminium MI (calibrated by a
  global factor of 0.64 to match independent stock estimates).
- **IRP "Bend the Trend".** International Resource Panel (2024). Resource efficiency and
  climate change: material efficiency strategies for a low-carbon future — used as a
  reference for lightweighting and sufficiency penetration rates in the ``narrow_product``
  and ``narrow_activity`` scenarios.
- **ODYM-RECC lightweighting dataset.** Pauliuk, S., et al. Material efficiency strategies
  for buildings, dataset ``3_SHA_LightWeighting_Buildings_V2.2.xlsx``. Available via
  Zenodo: https://zenodo.org/records/4671644.
