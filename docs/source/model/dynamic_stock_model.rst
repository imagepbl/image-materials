Dynamic Stock Model
===================

The dynamic stock model is the engine that turns an exogenous **stock demand**
(floorspace, vehicle fleet, generation capacity, ...) into the **inflow**
(new production) and **outflow** (end-of-life) time series that drive the
material flow calculations.

It is a *stock-driven, cohort-resolved* model: the total stock is prescribed for every year, and the
inflow is whatever is required to reach that stock given how the existing stock
ages out via the :doc:`survival matrix <survival_matrix>`.

This page states the equations as implemented in
:class:`imagematerials.model.GenericStocks` and
:class:`imagematerials.model.StocksQuintiles`, and notes where the implementation
deviates from the idealized formulation.


Notation
--------

.. list-table::
   :header-rows: 1
   :widths: 12 48 40

   * - Symbol
     - Meaning
     - Code
   * - :math:`t`
     - current year
     - ``t``
   * - :math:`c`
     - cohort year (year of installation)
     - ``Cohort``
   * - :math:`S(t, c)`
     - survival function: fraction of cohort :math:`c` still in use at year :math:`t`
     - ``survival_matrix[t].sel(Cohort=c)``
   * - :math:`\mathrm{St}(t)`
     - **demanded** total stock in year :math:`t` (exogenous driver)
     - ``stocks.loc[t]``
   * - :math:`\mathrm{St}_c(t)`
     - stock in year :math:`t` belonging to cohort :math:`c`
     - ``stock_by_cohort.loc[t, c]``
   * - :math:`I(t)`
     - inflow (additions) in year :math:`t`
     - ``inflow[t]``
   * - :math:`O_c(t)`
     - outflow in year :math:`t` from cohort :math:`c`
     - ``outflow_by_cohort[t]``

All quantities additionally carry ``Region`` and ``Type`` indices (and
``Quintile`` for :class:`~imagematerials.model.StocksQuintiles`). These are
element-wise throughout and are omitted below.

:math:`S` is a proper survival function: :math:`S(c, c) \le 1`, non-increasing in
:math:`t`, and :math:`S(t, c) = 0` for :math:`t < c`. It is built once for the
whole timeline by :func:`imagematerials.lifetimes.lifetimes_to_matrix`.


Idealized formulation
---------------------

**Cohort stock** — each surviving cohort decays according to its own
installation-year inflow:

.. math::

   \mathrm{St}_c(t) = I(c)\, S(t, c), \qquad t \ge c

**Stock balance** — the driver fixes the *sum* over cohorts:

.. math::

   \sum_{c \le t} \mathrm{St}_c(t) = \mathrm{St}(t)

**Inflow** — split the balance into the survivors of past inflow (cohorts
:math:`c < t`) and the new cohort (:math:`c = t`), then solve for :math:`I(t)`:

.. math::

   \mathrm{St}(t)
   = \underbrace{\sum_{c < t} I(c)\, S(t, c)}_{\text{survivors of past inflow}}
   + \underbrace{I(t)\, S(t, t)}_{\text{new cohort}}

.. math::

   I(t) = \frac{\mathrm{St}(t) - \sum_{c < t} I(c)\, S(t, c)}{S(t, t)},
   \qquad I(t) \ge 0

Dividing by :math:`S(t, t)` accounts for within-first-year mortality: if a
fraction :math:`1 - S(t, t)` of new units fails in their installation year, more
must be installed than the net gap in order to close it.

**Outflow** — conservation per cohort:

.. math::

   O_c(t) = \mathrm{St}_c(t - 1) - \mathrm{St}_c(t)
          = I(c)\, \bigl[ S(t - 1, c) - S(t, c) \bigr], \qquad c < t

.. math::

   O_t(t) = I(t)\, \bigl[ 1 - S(t, t) \bigr]

**Total mass balance** (holds automatically when the inflow is not clipped):

.. math::

   \mathrm{St}(t) - \mathrm{St}(t - 1) = I(t) - \sum_{c \le t} O_c(t)


Implementation
--------------

From :meth:`imagematerials.model.StocksQuintiles.compute_values` (and the
identical logic in :class:`~imagematerials.model.GenericStocks`):

**Inflow**

.. math::

   I(t) = \frac{\max\!\Bigl( 0,\; \mathrm{St}(t) - \sum_{c < t} \mathrm{St}_c(t) \Bigr)}{S(t, t)}

This is the boxed equation above. The running sum
:math:`\sum_{c < t} I(c)\, S(t, c)` is not recomputed each year; it is
accumulated in ``stock_by_cohort`` as each cohort is added, so
``stock_by_cohort.loc[t].sum("Cohort")`` at year :math:`t` picks up only past
cohorts (cohort :math:`t` has not been written yet).

**Cohort stock** — the entire future trajectory of the new cohort is written in
one shot at installation:

.. math::

   \mathrm{St}_t(t_{\text{future}}) = I(t)\, S(t_{\text{future}}, t),
   \qquad t_{\text{future}} \ge t

**Outflow**

.. math::

   O_c(t) = \mathrm{St}_c(t - 1) - \mathrm{St}_c(t) \quad (c \le t - 1),
   \qquad
   O_t(t) = I(t)\, \bigl[ 1 - S(t, t) \bigr]

For past cohorts the outflow is read back from the two stored stock endpoints,
which are both :math:`I(c)\, S(\cdot, c)` — so there is no numerical drift.


Materials layer
---------------

:class:`imagematerials.model.MaterialIntensitiesQuintiles` (and
:class:`~imagematerials.model.MaterialIntensities`) apply a cohort-specific
material intensity :math:`\mu_{m, c}` (kg of material :math:`m` per unit of stock
of cohort :math:`c`):

.. math::

   I^{\text{mat}}_m(t) = I(t)\, \mu_{m, t}

.. math::

   \mathrm{St}^{\text{mat}}_m(t)
   = \sum_{c \le t} \mathrm{St}_c(t)\, \mu_{m, c}
   = \sum_{c \le t} I(c)\, S(t, c)\, \mu_{m, c}

.. math::

   O^{\text{mat}}_m(t) = \sum_{c \le t} O_c(t)\, \mu_{m, c}

The :class:`~imagematerials.model.GenericMaterials` variant instead uses
:math:`\mu_{m, c} = w_c\, f_{m, c}`, i.e. stock weight times material mass
fraction.


Assumptions and caveats
-----------------------

1. **Non-negativity clip on inflow.** When the demanded stock
   :math:`\mathrm{St}(t)` falls faster than cohorts retire, the balance implies
   :math:`I(t) < 0`. The model sets :math:`I(t) = 0` instead, so in
   shrinking-stock years the realized total stock
   :math:`\sum_c \mathrm{St}_c(t)` **exceeds** the demanded
   :math:`\mathrm{St}(t)` and the total mass balance no longer closes exactly.
   This is standard MFA behavior (there is no mechanism to demolish serviceable
   stock early), but it means ``stock_by_cohort.sum("Cohort")`` should be used
   instead of ``stocks`` when a self-consistent stock is required.

2. **Division by** :math:`S(t, t)` **assumes** :math:`S(t, t) > 0`. For building
   and vehicle lifetime distributions :math:`S(t, t) \approx 1`, so the
   correction is negligible; for a distribution with large first-year mortality
   it materially inflates the inflow.

3. **Cohort trajectories are never revised.** ``stock_by_cohort`` is
   pre-allocated over the full timeline and each cohort's future is written once
   at installation. This is exact only because :math:`S` is a deterministic
   function of :math:`(t, c)` — there is no retrofit, no post-installation
   lifetime change, no re-derivation.

4. **Historic tail.** :meth:`imagematerials.model.GenericMainModel.compute_values`
   runs every pre-:math:`t` timestep once before the first simulated year, so
   that :math:`\sum_{c < t} \mathrm{St}_c(t)` is populated from a genuine cohort
   history rather than assuming the initial stock is a single cohort. Without
   this, early-year inflow and outflow are strongly biased. This is why the
   buildings sector reconstructs floorspace back to 1721.

5. **Quintile handling.** :class:`~imagematerials.model.StocksQuintiles` carries
   an extra ``Quintile`` dimension element-wise through every equation above; it
   is summed away before results enter cross-sector aggregation such as
   end-of-life reporting.


See Also
--------

- :doc:`survival_matrix` — how :math:`S(t, c)` is constructed from lifetime
  distributions
- :doc:`../models` — how the dynamic stock model fits into the overall workflow
- :doc:`../api/models_detail` — API reference for the model classes
