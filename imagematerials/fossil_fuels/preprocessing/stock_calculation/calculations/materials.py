import numpy as np
import pandas as pd
from ..attributes import constants


def cohorts_to_materials_typical_np(
    inflow, outflow_cohort, stock_cohort, composition, out_time
):
    """
    This function translates inflow and cohort specific stock/outflow of fossil fuel infrastructure elements to describe the materials involved
    IN:  stock/flow data on infrastrcucture elements; dims: type, region, time (* time)   --> double time dimension for stock (by cohort) and outflow (by cohort) to prepare for changing material composition
    OUT: stock/flow data on individual materials;     dims: type, material, region, time  (time of the output is limited to a selection of years defined by out_time, to ensure output in comparable format for all fuels)
    """
    # type         # materials               # region         # time
    inflow_mat = np.zeros(
        (
            len(inflow),
            len(composition.columns),
            len(inflow[0]),
            len(inflow[0][0]),
        )
    )
    outflow_mat = np.zeros(
        (
            len(inflow),
            len(composition.columns),
            len(inflow[0]),
            len(inflow[0][0]),
        )
    )
    stock_mat = np.zeros(
        (
            len(inflow),
            len(composition.columns),
            len(inflow[0]),
            len(inflow[0][0]),
        )
    )

    stype_list = list(
        composition.index.unique(0)
    )  # .unique() keeps the original order, which is important here (same for materials)
    stype_count = list(range(0, len(inflow)))
    stype_dict = dict(zip(stype_count, stype_list))

    mater_list = list(composition.columns.unique())
    mater_count = list(range(0, len(composition.columns)))
    mater_dict = dict(zip(mater_count, mater_list))

    for stype in range(0, len(inflow)):
        # before running, check if the vehicle type is at all relevant in the vehicle (save calculation time)
        if stock_cohort[stype].sum() > 0.001:
            for material in range(0, len(mater_list)):
                composition_used = composition.loc[
                    stype_dict[stype], mater_dict[material]
                ]
                # before running, check if the material is at all relevant in the vehicle (save calculation time)
                if composition_used > 0.001:
                    for region in range(0, len(inflow[0])):
                        inflow_mat[stype, material, region, :] = (
                            inflow[stype, region, :] * composition_used
                        )
                        outflow_mat[stype, material, region, :] = np.multiply(
                            outflow_cohort[stype, region, :, :].T,
                            composition_used,
                        ).T.sum(axis=1)
                        stock_mat[stype, material, region, :] = np.multiply(
                            stock_cohort[stype, region, :, :], composition_used
                        ).sum(axis=1)
                else:
                    pass
        else:
            pass

    length_materials = len(composition.columns)
    length_time = len(inflow[0][0])

    # return as pandas dataframe, just onc
    index = pd.MultiIndex.from_product(
        [stype_list, mater_list, range(1, constants.NUM_REGIONS + 1)],
        names=["type", "material", "region"],
    )
    columns = list(
        range(
            (constants.END_YEAR + 1 - len(inflow[0][0])),
            constants.END_YEAR + 1,
        )
    )
    pd_inflow_mat = pd.DataFrame(
        inflow_mat.reshape(
            (len(inflow) * length_materials * constants.NUM_REGIONS),
            (length_time),
        ),
        index=index,
        columns=columns,
    )
    pd_outflow_mat = pd.DataFrame(
        outflow_mat.reshape(
            (len(inflow) * length_materials * constants.NUM_REGIONS),
            (length_time),
        ),
        index=index,
        columns=columns,
    )
    pd_stock_mat = pd.DataFrame(
        stock_mat.reshape(
            (len(inflow) * length_materials * constants.NUM_REGIONS),
            (length_time),
        ),
        index=index,
        columns=columns,
    )

    # Put the results in a dictionary for easier access later
    return {
        "inflow": pd_inflow_mat.loc[:, out_time],
        "outflow": pd_outflow_mat.loc[:, out_time],
        "stock": pd_stock_mat.loc[:, out_time],
    }


# Material calculations for stock with only 1 relevant sub-type
def cohorts_to_materials_dynamic_np(
    inflow, outflow_cohort, stock_cohort, weight, composition
):
    """
    for those vehicles with only 1 relevant sub-type, we calculate the material stocks & flows as:
    Nr * weight (kg) * composition (%)
    INPUT:  --------------
    inflow         = numpy array (time, regions)      - number of vehicles by region over time
    outflow_cohort = numpy array (region, time, time) - number of vehicles in outlfow by region over time, by built year
    stock_cohort   = numpy array (region, time, time) - number of vehicles in stock by region over time, by built year
    OUTPUT: --------------
    pd_inflow_mat  = pandas dataframe (index: material, time; column: regions) - inflow of materials in kg of material, by region & time
    pd_outflow_mat = pandas dataframe (index: material, time; column: regions) - inflow of materials in kg of material, by region & time
    pd_stock_mat   = pandas dataframe (index: material, time; column: regions) - inflow of materials in kg of material, by region & time
    """
    inflow_mat = np.zeros(
        (len(composition.columns), len(inflow[0]), len(inflow))
    )
    outflow_mat = np.zeros(
        (len(composition.columns), len(inflow[0]), len(inflow))
    )
    stock_mat = np.zeros(
        (len(composition.columns), len(inflow[0]), len(inflow))
    )

    for material in range(0, len(composition.columns)):
        # before running, check if the material is at all relevant in the vehicle (save calculation time)
        if composition.iloc[:, material].sum() > 0.001:
            for region in range(0, len(inflow[0])):
                composition_used = composition.iloc[:, material].values
                inflow_mat[material, region, :] = (
                    inflow[:, region] * weight
                ) * composition_used
                outflow_mat[material, region, :] = np.multiply(
                    np.multiply(outflow_cohort[region, :, :].T, weight),
                    composition_used,
                ).T.sum(axis=1)
                stock_mat[material, region, :] = np.multiply(
                    np.multiply(stock_cohort[region, :, :], weight),
                    composition_used,
                ).sum(axis=1)
        else:
            pass

    length_materials = len(composition.columns)
    length_time = len(inflow[0], [0])
    #   length_time      = END_YEAR + 1 - (END_YEAR + 1 - len(inflow))

    index = pd.MultiIndex.from_product(
        [composition.columns, range(END_YEAR + 1 - len(inflow), END_YEAR + 1)],
        names=["time", "type"],
    )
    pd_inflow_mat = pd.DataFrame(
        inflow_mat.transpose(0, 2, 1).reshape(
            (length_materials * length_time), REGIONS
        ),
        index=index,
        columns=range(1, len(inflow[0]) + 1),
    )
    pd_outflow_mat = pd.DataFrame(
        outflow_mat.transpose(0, 2, 1).reshape(
            (length_materials * length_time), REGIONS
        ),
        index=index,
        columns=range(1, len(inflow[0]) + 1),
    )
    pd_stock_mat = pd.DataFrame(
        stock_mat.transpose(0, 2, 1).reshape(
            (length_materials * length_time), REGIONS
        ),
        index=index,
        columns=range(1, len(inflow[0]) + 1),
    )

    return pd_inflow_mat, pd_outflow_mat, pd_stock_mat
