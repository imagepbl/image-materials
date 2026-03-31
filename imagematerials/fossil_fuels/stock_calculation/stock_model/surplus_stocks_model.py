# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 15:54:38 2022

@author: thije
"""
import pandas as pd

idx = pd.IndexSlice


def stock_share_calc(stock, market_share, init_tech, techlist):

    # define new dataframes
    stock_cohorts = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [stock.columns, range(switchtime, endyear + 1)]
        ),
        columns=pd.MultiIndex.from_product(
            [techlist, range(switchtime - pre_time, endyear + 1)]
        ),
    )  # year, by year, by tech
    inflow_by_tech = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [stock.columns, range(switchtime, endyear + 1)]
        ),
        columns=techlist,
    )
    outflow_cohorts = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [stock.columns, range(switchtime, endyear + 1)]
        ),
        columns=pd.MultiIndex.from_product(
            [techlist, range(switchtime - pre_time, endyear + 1)]
        ),
    )  # year by year by tech

    # specify lifetime & other settings
    mean = storage_lifetime_interpol.loc[
        switchtime, init_tech
    ]  # select the mean lifetime for Lead-acid batteries in 1990
    stdev = (
        mean * stdev_mult
    )  # we use thesame standard-deviation as for generation technologies, given that these apply to 'energy systems' more generally
    survival_init = scipy.stats.foldnorm.sf(
        timeframe, mean / stdev, 0, scale=stdev
    )
    techlist_new = techlist
    techlist_new.remove(
        init_tech
    )  # techlist without Lead-acid (or other init_tech)

    # actual inflow & outflow calculations, this bit takes long!
    # loop over regions, technologies and years to calculate the inflow, stock & outflow of storage technologies, given their share of the inflow.
    for region in stock.columns:

        # pre-calculate the stock by cohort of the initial stock of Lead-acid
        multiplier_pre = (
            stock.loc[switchtime, region] / survival_init.sum()
        )  # the stock is subdivided by the previous cohorts according to the survival function (only allowed when assuming steady stock inflow)

        # pre-calculate the stock as lists (for efficiency)
        initial_stock_years = [
            np.flip(survival_init[0 : pre_time + 1]) * multiplier_pre
        ]

        for year in range(
            1, (endyear - switchtime) + 1
        ):  # then fill the columns with the remaining fractions
            initial_stock_years.append(
                initial_stock_years[0] * survival_init[year]
            )

        stock_cohorts.loc[
            idx[region, :],
            idx[init_tech, list(range(switchtime - pre_time, switchtime + 1))],
        ] = initial_stock_years  # fill the stock dataframe according to the pre-calculated stock
        outflow_cohorts.loc[
            idx[region, :],
            idx[init_tech, list(range(switchtime - pre_time, switchtime + 1))],
        ] = (
            stock_cohorts.loc[
                idx[region, :],
                idx[
                    init_tech,
                    list(range(switchtime - pre_time, switchtime + 1)),
                ],
            ].shift(1, axis=0)
            - stock_cohorts.loc[
                idx[region, :],
                idx[
                    init_tech,
                    list(range(switchtime - pre_time, switchtime + 1)),
                ],
            ]
        )

        # set the other stock cohorts to zero
        stock_cohorts.loc[idx[region, :], idx[techlist_new, pre_year_list]] = 0
        outflow_cohorts.loc[
            idx[region, :], idx[techlist_new, pre_year_list]
        ] = 0
        inflow_by_tech.loc[idx[region, switchtime], techlist_new] = (
            0  # inflow of other technologies in 1990 = 0
        )

        # except for outflow and inflow in 1990 (switchtime), which can be pre-calculated for Deep-cycle Lead Acid (@ steady state inflow, inflow = outflow = stock/lifetime)
        outflow_cohorts.loc[idx[region, switchtime], idx[init_tech, :]] = (
            outflow_cohorts.loc[idx[region, switchtime + 1], idx[init_tech, :]]
        )  # given the assumption of steady state inflow (pre switchtime), we can determine that the outflow is the same in switchyear as in switchyear+1
        inflow_by_tech.loc[idx[region, switchtime], init_tech] = (
            stock.loc[switchtime, region] / mean
        )  # given the assumption of steady state inflow (pre switchtime), we can determine the inflow to be the same as the outflow at a value of stock/avg. lifetime

        # From switchtime onwards, define a stock-driven model with a known market share (by tech) of the new inflow
        for year in range(switchtime + 1, endyear + 1):

            # calculate the remaining stock as the sum of all cohorts in a year, for each technology
            remaining_stock = 0  # reset remaining stock
            for tech in inflow_by_tech.columns:
                remaining_stock += stock_cohorts.loc[
                    idx[region, year], idx[tech, :]
                ].sum()

            # total inflow required (= required stock - remaining stock);
            inflow = max(
                0, stock.loc[year, region] - remaining_stock
            )  # max 0 avoids negative inflow, but allows for idle stock surplus in case the size of the required stock is declining more rapidly than it's natural decay

            stock_cohorts_list = []

            # enter the new inflow & apply the survival rate, which is different for each technology, so calculate the surviving fraction in stock for each technology
            for tech in inflow_by_tech.columns:
                # apply the known market share to the inflow
                inflow_by_tech.loc[idx[region, year], tech] = (
                    inflow * market_share.loc[year, tech]
                )
                # first calculate the survival (based on lifetimes specific to the year of inflow)
                survival = scipy.stats.foldnorm.sf(
                    np.arange(0, (endyear + 1) - year),
                    storage_lifetime_interpol.loc[year, tech]
                    / (storage_lifetime_interpol.loc[year, tech] * 0.2),
                    0,
                    scale=storage_lifetime_interpol.loc[year, tech] * 0.2,
                )
                # then apply the survival to the inflow in current cohort, both the inflow & the survival are entered into the stock_cohort dataframe in 1 step
                stock_cohorts_list.append(
                    inflow_by_tech.loc[idx[region, year], tech] * survival
                )

            stock_cohorts.loc[
                idx[region, list(range(year, endyear + 1))], idx[:, year]
            ] = list(map(list, zip(*stock_cohorts_list)))

    # separate the outflow (by cohort) calculation (separate shift calculation for each region & tech is MUCH more efficient than including it in additional loop over years)
    # calculate the outflow by cohort based on the stock by cohort that was just calculated
    for region in stock.columns:
        for tech in inflow_by_tech.columns:
            outflow_cohorts.loc[idx[region, :], idx[tech, :]] = (
                stock_cohorts.loc[idx[region, :], idx[tech, :]].shift(
                    1, axis=0
                )
                - stock_cohorts.loc[idx[region, :], idx[tech, :]]
            )

    return inflow_by_tech, stock_cohorts, outflow_cohorts
