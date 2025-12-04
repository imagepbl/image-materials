Survival Matrix
===============

The survival matrix is a fundamental concept in IMAGE materials for modeling how stocks age and exit the system over time.

What is a Survival Matrix?
--------------------------

A survival matrix defines the probability that a stock of a given age will survive to the next time period. It is based on lifetime distributions and is essential for dynamic stock modeling.

Key Concepts
------------

**Cohort-based Aging**: Each cohort (age group) of stock has different survival probabilities based on their age and expected lifetime.

**Lifetime Distributions**: Survival matrices are computed from lifetime distributions that describe when stocks are expected to reach end-of-life. these include a mean lifetime and a distribution shape (e.g., normal, Weibull).

**Time-dependent Outflows**: The survival matrix determines how much stock exits the system at each time step due to aging and reaching end-of-life.

How Survival Matrices Work
---------------------------

1. **Initialization**: When new stock enters the system (inflow), it starts with a survival probability of 1.0
2. **Aging**: Each time step, the survival probability decreases based on the lifetime distribution
3. **Outflows**: Stock that "dies" (survival probability reaches 0) becomes outflow from the system
4. **Material Flows**: These outflows then feed into material calculations and end-of-life processing

Mathematical Foundation
-----------------------

The survival matrix :math:`S(t,c)` represents the fraction of cohort :math:`c` that survives at time :math:`t`:

.. math::

   S(t,c) = P(\text{lifetime} > t-c)

Where :math:`t-c` is the age of the cohort at time :math:`t`.

Implementation in IMAGE Materials
----------------------------------

IMAGE materials uses the :class:`imagematerials.survival.SurvivalMatrix` class to handle survival calculations. For detailed API documentation, see :doc:`SurvivalMatrix <api/models_detail>`.

The :class:`imagematerials.survival.ScipySurvival` class provides scipy-based distributions for computing survival probabilities. For detailed API documentation, see :doc:`ScipySurvival <api/models_detail>`.

Usage in Stock Models
---------------------

Survival matrices are primarily used in:

- :class:`imagematerials.model.GenericStocks`: Core stock dynamics
- :class:`imagematerials.model.SharesInflowStocks`: Stock dynamics with share-based inflows

The survival matrix determines:

- How much stock remains active at each time step
- When stock becomes outflow and available for end-of-life processing
- The aging behavior of different stock cohorts

Examples and Applications
-------------------------

Different sectors use different survival characteristics:

**Buildings**: Long lifetimes (50-100 years) with normal or Weibull distributions
**Vehicles**: Medium lifetimes (10-20 years) with normal distributions  

For detailed examples of survival matrix usage, see the sector-specific documentation:

- :doc:`Buildings <api/buildings>`
- :doc:`Vehicles <api/vehicles>`
- :doc:`Electricity <api/electricity>`

See Also
---------

- :doc:`Models Overview <models>` - How survival matrices fit into the overall modeling framework
- :doc:`API Reference <api/modules>` - Technical documentation for survival classes