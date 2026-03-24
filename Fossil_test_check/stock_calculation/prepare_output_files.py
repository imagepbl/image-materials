from __future__ import annotations
import pandas as pd

flow_list = ['inflow', 'outflow', 'stock']

def prepend_index(original, prepend1, prepend2, prepend3):
    prepended1 = pd.concat([original],   keys=[prepend3], names=['stage'])
    prepended2 = pd.concat([prepended1], keys=[prepend2], names=['fuel'])
    prepended3 = pd.concat([prepended2], keys=[prepend1], names=['flow'])
    return prepended3

def prepend_stage(original_list, stage, fuel):
    #print(original_list)
    for item in range(0,len(original_list)):
        original_list[item] = prepend_index(original_list[item], flow_list[item], fuel, stage)
    output = pd.concat([original_list[0], original_list[1], original_list[2]])
    return output


def merge_material(data: dict, fuel: str):
    result = []
    for stage in data[fuel].keys(): # For coal: extraction, transport. for others: also pipelines
        dataset = []
        for flow in data[fuel][stage].keys():
            dataset.append(data[fuel][stage][flow])
        mat_extraction = prepend_stage(original_list=dataset, stage=stage, fuel=fuel)
        result.append(mat_extraction)
    return pd.concat(result)


def merge_total(data):
    result = []
    for fuel in data.keys():
        result.append(merge_material(data=data, fuel=fuel))
    return pd.concat(result)   
    
