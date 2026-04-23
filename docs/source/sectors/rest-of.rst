##############
Rest-of sector
##############
 
**************
Resource Model
**************

The resource model is implemented in :mod:`imagematerials.rest_of.resource_model` (``ResourceModel`` class) and orchestrates all regression calculations for a given material. It performs the following steps:

1. Reads in total apparent consumption; if sectoral data is available, it reads that in and subtracts it to isolate the rest-of sector.
2. Calculates GDP per capita and consumption per capita as the regressors (:math:`X`).
3. Forms groups of IMAGE regions used for fitting (see :mod:`imagematerials.rest_of.correlation_materials`).
4. Fits all regression models to each group (via :func:`imagematerials.rest_of.projections_materials.estimate_models`).
5. Assigns either the best-fitting model (lowest RMSE) or a programmer-specified model to each region group (current implementation).

The ``ResourceModel`` objects are initiated in:

- Metals: :mod:`imagematerials.rest_of.metals_projections`. It covers: steel, aluminium and copper.
- Non-metallic minerals: :mod:`imagematerials.rest_of.nmm_projections`. It covers: cement, sand, limestone and clay.

Regression Classes
==================

The following regression classes are defined in this module:

OLS
---

These classes inherit from ``OLS_Model`` and apply linear regression on transformed data:

- ``Log_Log_Model``: :math:`\ln(y) = \ln(X)` — log-log regression
- ``Semi_Log_Model``: :math:`y = \ln(X)` — semi-log regression
- ``Log_Inverse_Model``: :math:`\ln(y) = 1/X` — log-inverse regression
- ``Log_Log_Inverse_Model``: :math:`\ln(y) = \ln(X) + 1/X` — log with log and inverse terms
- ``Log_Log_Square_Model``: :math:`\ln(y) = \ln(X) + \ln(X)^2` — log with quadratic log term

NLS
---

These classes inherit from ``NLS_Model`` and fit non-linear functions via ``scipy.optimize.curve_fit``.

.. note::

   For all NLS models except ``NLI_Model``, the regressor :math:`X` is scaled by the maximum value in the dataset before fitting:

   .. math::

      \tilde{X} = \frac{X}{X_{\max}}

   This normalises the input to the range :math:`(0, 1]`, which improves numerical stability of the non-linear solver and prevents the exponential terms from saturating prematurely. 
   Predictions are made using the same scaled :math:`\tilde{X}`.

- ``NLI_Model``: :math:`y = a \cdot e^{b/X} + c` — non-linear inverse
- ``GOMPERTZ_Model``: :math:`y = a \cdot e^{-b \cdot e^{-cX}}` — Gompertz growth curve
- ``LG_Model``: :math:`y = c / (1 + a \cdot e^{-bX})` — logistic growth
- ``BW_Model``: :math:`y = a - (a - b) \cdot e^{-cX}` — bounded Weibull-like saturation
- ``Log_Gauss_Saturate_Model``: :math:`y = a \cdot e^{-(\ln(X) - b)^2 / (2c^2)} + d` — log-Gaussian with saturation offset

After fitting, all models (both OLS and NLS) store the following statistics for later analysis:

- **Coefficients** — fitted parameter values
- **Standard errors** — per-parameter standard errors
- **p-values** — per-parameter significance (t-test)
- **95% confidence intervals** — lower and upper bounds per parameter
- **R²** — coefficient of determination (OLS only; ``NaN`` for NLS)
- **RMSE** — root mean squared error
- **Overall model p-value** — F-test significance of the full model

These are accessible via the ``stats_summary`` property (as a ``DataFrame``) and individual properties such as ``r2``, ``rmse``, ``p_value``, ``p_values``, ``std_errors``, and ``confidence_intervals``.


Preparation projection/Preprocessing
====================================

The rest-of module is recalibrated only when upstream sector inputs change (for example, updated buildings, vehicles, or electricity inflows that affect the rest-of subtraction).

Recalibration is triggered in preprocessing (``refit=True`` in :func:`imagematerials.rest_of.preprocessing.main.rest_of_preprocessing`). During this step:

1. Region-group regressions are re-estimated and the selected Gompertz fits are retained for each material/region.
2. The fitted parameters :math:`(a, b, c)` are exported to NetCDF (``coefs_gompertz.nc``).
3. The regressor scaling factor :math:`X_{\max}` is exported to NetCDF (``max_x_regressor.nc``).
4. Historic rest-of consumption references are exported to NetCDF (``diff_cons_all.nc`` and ``diff_cons_all_mean.nc``).

When the simulation starts, preprocessing reads these NetCDF inputs and provides them to :class:`imagematerials.model.RestOf`. The ``RestOf`` class then computes rest-of inflows from GDP per capita using the stored Gompertz parameters, with fallback to historic values where needed.

Simulation
==========

The simulation uses :class:`imagematerials.model.RestOf`, which uses:

- ``gompertz_coefs``
- ``gdp_per_capita``
- ``population``
- ``historic_diff_consumption_mean``
- ``historic_diff_consumption_total``

For each year :math:`t`, the class computes rest-of inflow in a tiered assignment order:

1. **Historic value first (highest priority)**: if a real historic value exists in ``historic_diff_consumption_total`` at year :math:`t`, that value is always used.
2. **Mean-historic fallback for missing parameters**: if no real historic value exists and Gompertz-parameters are not available (for example due to missing fitted coefficients), use ``historic_diff_consumption_mean (if this is available - otherwise leave nan)``.
3. **Transition-period smoothing**: if Gompertz parameters are available and the point is within the transition window after the historic period (default 20 years), linearly blend from the last historic value to the Gompertz estimate.
4. **Pure Gompertz outside transition**: if Gompertz parameters are available and the point is outside the transition window, use only the Gompertz function output.

This defines the runtime hierarchy as: observed historic data, then mean fallback where needed, then smoothed transition, and finally pure model-based values.