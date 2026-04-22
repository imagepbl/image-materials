Infrastructure
==============

The infrastructure module of IMAGE Materials models the stocks and material flows
associated with **transport infrastructure**: roads (urban and rural, paved and unpaved,
classified by hierarchy), bridges, tunnels, parking, and rail (conventional rail, urban
rail, and high-speed rail). It follows the same dynamic stock modelling pattern as the
buildings and vehicles sectors: exogenous physical demand is combined with lifetime
distributions and material intensities to produce annual inflows, outflows and in-use
stocks of materials per region and infrastructure type.

The sector is implemented in :mod:`imagematerials.infrastructure`, with two primary
entry points:

- :func:`run_infrastructure_simulation` — builds the :class:`Sector`, assembles a
  :class:`GenericStocks → MaterialIntensities <imagematerials.factory.ModelFactory>`
  pipeline and runs the simulation.
- :func:`export_infrastructure_report` — optional standalone reporting step that writes
  the sector outputs to NetCDF and a wide-format Excel report. Not required for
  integration with other sectors. FOR TESTING PURPOSES

Scope
------------------

The infrastructure module represents the physical transportation infrastructure, and the materials embedded in it. Concretely it covers:

- **Roads**, classified by hierarchy (motorway, primary, secondary, tertiary, residential),
  by urban/rural context, and by paving status (paved/unpaved). Bridges and tunnels of
  each road class are tracked separately because of their distinctive material intensities
  and lifetimes.
- **Parking** area associated with road networks.
- **Rail**, including conventional rail, urban rail (metro/tram), and high-speed rail
  (HSR), each with their own bridges and tunnels.
  
- **Excluded**, road signings, barriers, earthworks, bus stops and stations, rail stops and stations, rail auxiliary infrastructure.

For each infrastructure type the module produces, by region and year:

- **Physical stocks, inflows and outflows** (area in km² for roads and parking or km for rail).
- **Material stocks, inflows and outflows** for twelve materials:
  asphalt, concrete, steel, brick, stone, wood, aggregate, copper, zinc,
  plastics, aluminium, and bronze.

The module is not a transport demand model: it uses service-demand signals
(vehicle-kilometres, GDP, population density) loaded upstream by from IMAGE, IMAGE-Timer and other sources and translates
them into physical infrastructure and its material footprint.


Data Input
------------------

The infrastructure module takes two kinds of input: **scenario drivers** produced by
IMAGE, and **external intensity - lifetimes - paving - type - indicators** provided as bundled data files.

**Scenario drivers (per region, per year):**

- Vehicle-kilometres (VKM) from the IMAGE TIMER-TRAVEL transport model — the main
  driver of road network demand.
- GDP per capita — used in regressions that determine road-class shares and
  paving shares.
- Population and urban population — used to split the network into urban and rural
  components.
- Human Development Index (HDI) — used to allocate road area across the road-class
  hierarchy.
- Urban area - used to calculate population density for urban and rural areas.

**External parameters:**

- Historical rail network length (km) by region and rail type, used to anchor the
  rail stock trajectory before the model takes over.
- Material intensity tables (``Materials-paving.xlsx`` for roads; ``Materials_infra.xlsx``
  and ``Materials_infra_bt.xlsx`` for rail, bridges and tunnels) giving
  kg-per-km² for each combination of infrastructure type, material, and region.
- Lifetime distribution parameters:
    - Weibull shape/scale parameters for roads, road obsolete types, bridges,
      tunnels and parking.
    - Folded-normal parameters for rail, urban rail, HSR and rail bridges/tunnels.

All scenario inputs are read as xarray DataArrays indexed by ``Region`` and ``Time``;
static parameters are read from Excel files in ``data/raw/infrastructure/``. CHANGE WHEN
LOCATION CHANGES


Preprocessing Structure
-----------------------

TO BE CHANGED WITH DIFFERENT STRUCTURE
Preprocessing is implemented in :mod:`imagematerials.infrastructure.preprocessing` and
exposed through the single function
:func:`get_preprocessing_data_infrastructure(path_base, scenario)`. It turns raw data
files into the dictionary that the model pipeline consumes.

The preprocessing proceeds in the following stages:

1. **Driver development.** Load VKM, GDP, population and HDI, harmonise them to
   the shared region and time coordinates defined in
   :mod:`imagematerials.constants`, and interpolate any gaps.

2. **Network sizing.** Apply elasticity/regression relationships to derive total road
   area (km²) from the demand drivers, separately for urban and rural segments.
   Allocate total road area across the road-class hierarchy as a function of HDI, and
   apply paving shares as a function of GDP.

3. **Bridges, tunnels and rail.** Bridges and tunnels are computed as ratios of the
   road network area per class. Rail stocks are derived from historical length series
   and projected forward using scenario drivers using IMAGE-timer Rail VKM development.

4. **Type consolidation.** The sector distinguishes *active* types (used in current
   simulations) from *obsolete* types (kept for mass-balance with legacy data). An
   obsolete-to-active type map is built so that legacy inflows can be reconciled with
   active-type outflows in the reporting step.
	TO BE ADDED - obsolete stocks processing

5. **Material intensities.** Construct an ``xr.DataArray`` of kg/km² indexed by
   ``(Type, material, Region, Cohort)``. Apply the **80/20 permanent-aggregate split**:
   80% of the aggregate material intensity is held as permanent, non-replaceable
   subgrade, while 20% flows through the dynamic stock model like other materials.

6. **Lifetimes.** Assemble Weibull and folded-normal parameter arrays suitable for
   :class:`imagematerials.survival.ScipySurvival`.

7. **Dictionary.** Return a dictionary with the following keys:

   +---------------------------+------------------------------------------+--------------+
   | Key                       | Dimensions                               | Unit         |
   +===========================+==========================================+==============+
   | ``stocks``                | ``(Type, Time, Region)``                 | km² or km    |
   +---------------------------+------------------------------------------+--------------+
   | ``material_intensities``  | ``(Type, material, Region, Cohort)``     | kg / km²     |
   +---------------------------+------------------------------------------+--------------+
   | ``lifetimes``             | nested dict of parameter DataArrays      | —            |
   +---------------------------+------------------------------------------+--------------+
   | ``active_types``          | list[str]                                | —            |
   +---------------------------+------------------------------------------+--------------+
   | ``obsolete_types``        | list[str]                                | —            |
   +---------------------------+------------------------------------------+--------------+
   | ``obsolete_to_active_map``| dict[str, str]                           | —            |
   +---------------------------+------------------------------------------+--------------+
   | ``permanent_aggregate_mi``| ``(Type, Region)``                       | kg / km²     |
   +---------------------------+------------------------------------------+--------------+
   | ``knowledge_graph``       | :class:`KnowledgeGraph`                  | —            |
   +---------------------------+------------------------------------------+--------------+

Output
---------------

The simulation produces time-varying outputs via the :class:`GenericStocks` and
:class:`MaterialIntensities` submodels attached to the infrastructure sector. Once
``main_model_factory.simulate(...)`` has run, they are available either as
``prism.TimeVariable`` objects on the submodels, or as ``xr.DataArray`` via the
sector's ``all_data`` dictionary (``main_model_factory.infrastructure.get(...)``).

**Physical flows (km² / year) for road infrastructure and (km / year) for rail infrastructure, from GenericStocks:**

- ``inflow`` — new infrastructure built each year by Type and Region.
- ``outflow_by_cohort`` — infrastructure retiring each year, broken down by the
  year it was built.
- ``stock_by_cohort`` — in-use stock, broken down by cohort.

**Material flows (kg / year), from MaterialIntensities:**

- ``inflow_materials`` — material demand for new construction.
- ``outflow_by_cohort_materials`` — material leaving the in-use stock.
- ``stock_by_cohort_materials`` — material embedded in the in-use stock.

All outputs are indexed by ``Region``, ``Type``, ``material`` (where applicable),
``Time`` and — for cohort-resolved quantities — ``Cohort``. This matches the coordinate
conventions used by the buildings and vehicles sectors so that cross-sector
aggregation (e.g. :class:`SumMaterials` in ``run_integrated.py``) can combine them.

**Dimensions**

- ``Region`` (26): the IMAGE world regions. NEED TO CHANGE FORMAT TO MATCH OTHER SECTORS
- ``Type`` (73): the active infrastructure types — all road classes (urban/rural ×
  paved/unpaved × motorway/primary/secondary/tertiary/residential/cycle), bridges
  and tunnels per road class, parking, and the rail types
  (``total_rail``, ``urban_rail``, ``HST``, ``rail_bridge``, ``rail_tunnel``).
- ``material`` (12): ``aggregate``, ``asphalt``, ``concrete``, ``steel``,
  ``brick``, ``stone``, ``wood``, ``copper``, ``zinc``, ``plastics``, ``aluminium``,
  ``bronze``.
- ``Time`` (130): simulation years from ``YEAR_START`` (1971) to ``YEAR_END`` (2100)
  inclusive, as defined in :mod:`imagematerials.constants`.

**Data variables**

+------------------------+--------------------------------------+-----------+
| Variable               | Meaning                              | Unit      |
+========================+======================================+===========+
| ``physical_inflow``    | New network built each year          | km² / yr  |
+------------------------+--------------------------------------+-----------+
| ``physical_outflow``   | Network retiring each year           | km² / yr  |
+------------------------+--------------------------------------+-----------+
| ``physical_stock``     | In-use network stock                 | km²       |
+------------------------+--------------------------------------+-----------+
| ``physical_expansion`` | Net change in stock                  | km² / yr  |
|                        | (``inflow − outflow``)               |           |
+------------------------+--------------------------------------+-----------+
| ``material_inflow``    | Material demand for new construction | kg / yr   |
+------------------------+--------------------------------------+-----------+
| ``material_outflow``   | Material leaving the in-use stock    | kg / yr   |
+------------------------+--------------------------------------+-----------+
| ``material_stock``     | Material embedded in the in-use      | kg        |
|                        | stock                                |           |
+------------------------+--------------------------------------+-----------+
| ``material_expansion`` | Net change in material stock         | kg / yr   |
+------------------------+--------------------------------------+-----------+

Both the mass-balance correction (obsolete-type inflows subtracted from
active-type outflows) and the permanent-aggregate add-back (the 80% subgrade
component re-introduced into material stocks and inflows) have already been
applied to the exported arrays, so downstream consumers can use the values
directly without further post-processing.


References
---------------

Data sources and scientific background for the infrastructure module:

- **IMAGE-Timer.** Girod, B., van Vuuren, D. P., & de Vries, B. (2013).
  *Influence of travel behavior on global CO₂ emissions*. Transportation Research
  Part A: Policy and Practice, 50, 183–197. Subsequent updates by Edelenbosch et al.
  provide the VKM inputs used here.

- **Material intensities and infrastructure assumptions.** van Engelenburg, M.,
  Deetman, S., Fishman, T., Behrens, P., & van der Voet, E. (2024).
  *TRIPI: A global dataset and codebase of the total resources in physical
  infrastructure encompassing road, rail, and parking*.
  Data in Brief, 54, 110387.

- **Full methodology and regression description.** van Engelenburg, M., Berrill, P.,
  Deetman, S., Fishman, T., Behrens, P., & van der Voet, E. (in prep., 2026).
  *Global urban and rural demand of resources in transportation infrastructure*.


