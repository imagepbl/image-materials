"""Combine all infrastructure stock streams into a single ``xr.DataArray`` (section 9d).

Roads (active + obsolete), bridges and tunnels per type (active + obsolete),
parking (active + obsolete) and rail elements (active + obsolete) are
concatenated along ``Type`` and wrapped in a ``prism.Q_`` with units km².
Also returns the type lists and a fresh ``KnowledgeGraph`` reflecting the
full type universe, plus the obsolete→active map used for mass balance.
"""
import xarray as xr
import pandas as pd
import prism

from imagematerials.concepts import KnowledgeGraph, Node


def combine_all_stocks(roads_result, bt_active_types, bt_active_das,
                       bt_obs_types, bt_obs_das,
                       parking_types, parking_das,
                       parking_obs_types, parking_obs_das,
                       rail_element_types, rail_element_das,
                       rail_obs_types, rail_obs_das):
    """Combine all stock streams into the final preprocessing dictionary fields.

    ``roads_result`` is the dict returned by ``roads.build_road_type_stocks``.

    Returns a dict ready to be merged into ``preprocessing_results`` with
    keys: stocks, active_types, obsolete_types, obsolete_to_active_map,
    knowledge_graph, set_unit_flexible.
    """
    all_types = roads_result["all_types"]
    obsolete_types = roads_result["obsolete_types"]
    combined_types = roads_result["combined_types"]
    combined_da_list = roads_result["combined_da_list"]

    final_all_types = (combined_types + bt_active_types + bt_obs_types
                       + parking_types + parking_obs_types
                       + rail_element_types + rail_obs_types)
    final_all_das = (combined_da_list + bt_active_das + bt_obs_das
                     + parking_das + parking_obs_das
                     + rail_element_das + rail_obs_das)

    da_all = xr.concat(final_all_das, pd.Index(final_all_types, name="Type"))
    da_all = prism.Q_(da_all, "km**2")

    # reorder da_roads to have the order: TIME, REGION, TYPE
    da_all = da_all.transpose("Time", "Region", "Type")

    # Update type tracking for mass balance
    # Active types now include roads + bridges/tunnels + parking + rail
    all_active_types = all_types + bt_active_types + parking_types + rail_element_types
    all_obsolete_types = obsolete_types + bt_obs_types + parking_obs_types + rail_obs_types

    # Map obsolete types to their active counterparts for mass balance
    # Road obsolete -> road active (already paired by index)
    # Bridge/tunnel obsolete -> bridge/tunnel active (paired by index)
    # Parking obsolete -> parking active (paired by index)
    # Rail obsolete -> rail active (need explicit mapping)
    obs_to_active_map = {}
    for a, o in zip(all_types, obsolete_types):
        obs_to_active_map[o] = a
    for a, o in zip(bt_active_types, bt_obs_types):
        obs_to_active_map[o] = a
    for a, o in zip(parking_types, parking_obs_types):
        obs_to_active_map[o] = a
    obs_to_active_map["obsolete_total_rail"] = "total_rail"
    obs_to_active_map["obsolete_rail_bridge"] = "rail_bridge"
    obs_to_active_map["obsolete_rail_tunnel"] = "rail_tunnel"

    # Update KnowledgeGraph
    infra_graph = KnowledgeGraph(Node("Infrastructure"))
    for t in final_all_types:
        infra_graph.add(Node(t, inherits_from="Infrastructure"))

    return {
        "stocks": da_all,
        "active_types": all_active_types,
        "obsolete_types": all_obsolete_types,
        "obsolete_to_active_map": obs_to_active_map,
        "knowledge_graph": infra_graph,
        "set_unit_flexible": prism.U_(da_all),
        "final_all_types": final_all_types,
        "all_active_types": all_active_types,
        "all_obsolete_types": all_obsolete_types,
    }
