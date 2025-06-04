# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 12:00:37 2024

@author: Arp00003
"""

#%% Auxiliary functions for creating a Sankey diagram


import numpy as np
import pandas as pd
import plotly.graph_objects as go

from typing import Any, Iterable

def create_node_dict(nodes: Iterable[Any]) -> dict[Any, int]:
    """
    Creates a dictionary with nodes being the keys and unique identifiers being 
    the values, used for a Sankey diagram.

    Parameters
    ----------
    nodes : Iterable[Any]
        Iterable of nodes.

    Returns
    -------
    dict[Any, int]
        Dictionary keyed by the provided nodes with values being unique 
        identifiers.

    """
    nodes_dict = {}
    index = 0
    for node in nodes:
        if node not in nodes_dict:
            nodes_dict[node] = index
            index += 1
    
    return nodes_dict

def index_mapper(*mappers):
    def func(index):
        if isinstance(index, tuple):
            return tuple(mapper(e) for e, mapper in zip(index, mappers))
        return mappers[0](index)
    
    return func

def convert_index_to_node_id(series: pd.Series, nodes_dict: dict[Any, int]):
    series.index = series.index.map(index_mapper(nodes_dict.get, nodes_dict.get))
    

def prepare_Sankey_lists(series: pd.Series, link_source: list = None, 
                         link_target: list = None, link_value: list = None) \
    -> tuple[list, list, list]:
    """
    Prepare lists for Sankey diagram
    :param series: A Pandas.Series with a MultiIndex of two levels. The first 
                   (second) level is considered the source (target).
    :return: A tuple of three lists: list for sources, targets, and values.
    """
    # Create lists if not provided
    if link_source is None:
        link_source = []
    if link_target is None:
        link_target = []
    if link_value is None:
        link_value = []
    # Fill lists
    for (source, target), value in series.items():
        if value == 0 or not np.isfinite(value):
            continue

        link_source.append(source)
        link_target.append(target)
        link_value.append(value)
    
    return link_source, link_target, link_value