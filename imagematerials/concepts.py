from dataclasses import dataclass, field
from typing import Optional

import xarray as xr
import json

@dataclass
class Node():
    name: str
    synonyms: list[str] = field(default_factory=list)
    inherits_from: Optional[str] = None

    def __post_init__(self):
        if self.inherits_from is not None:
            try:
                self.match_coords(self.inherits_from)
                raise ValueError(f"Cannot create node that inherits from itself: {self}.")
            except KeyError:
                pass

    def match_coords(self, input_coords):
        if isinstance(input_coords, str):
            input_coords = [input_coords]
        name_list = [self.name] + self.synonyms
        if not set(name_list).intersection(input_coords):
            raise KeyError()
        for name in name_list:
            if name in input_coords:
                return name

    def to_dict(self) -> str:
        return {"name": self.name, "synonyms": self.synonyms,
                           "inherits_from": self.inherits_from}
    @classmethod
    def from_dict(cls, node_dict):
        return cls(**node_dict)

class KnowledgeGraph():
    def __init__(self, *items):
        self._items: list[Node] = []
        for item in items:
            self.add(item)

    def add(self, node: Node):
        """Add new node to knowledge graph and validate it.

        Parameters
        ----------
        node
            New node to be added.

        Raises
        ------
        ValueError
            If the node already exists or the parent does not exist.

        """
        if node.name in self._items:
            raise ValueError("Node already exists.")
        if node.inherits_from is not None and node.inherits_from not in self:
            print(node.inherits_from is not None, node.inherits_from not in self)
            raise ValueError(f"Parent {node.inherits_from} of {node.name} does not exist.")
        self._items.append(node)

    def __contains__(self, node_name) -> bool:
        try:
            self[node_name]
        except KeyError:
            return False
        return True

    def __getitem__(self, node_name) -> Node:
        for item in self._items:
            if item.name == node_name:
                return item
            if node_name in item.synonyms:
                return item
        raise KeyError(f"Key {node_name} does not exist")

    def find_relations(self, input_coords, output_coords):
        relations = {}
        for cur_out_coord in output_coords:
            relations[cur_out_coord] = self.find_one_relation(input_coords, cur_out_coord)
        return relations

    def find_one_relation(self, input_coords: list[str], output_name: str) -> list[str]:
        if output_name in input_coords:
            return [output_name]

        output_node = self[output_name]
        try:
            return [output_node.match_coords(input_coords)]
        except KeyError:
            pass

        try:
            ancestors = self._find_ancestors(input_coords, output_node)
        except KeyError:
            ancestors = []
        try:
            descendants = self._find_descendants(input_coords, output_node)
        except KeyError:
            descendants = []

        if len(ancestors) > 0 and len(descendants) > 0:
            raise ValueError(f"Cannot find relations, because {output_node} has both ancestors "
                                "and descendants")
        if len(ancestors) + len(descendants) == 0:
            raise ValueError(f"Cannot find relations of {output_node} in input coordinates.")

        return ancestors + descendants

    def find_relations_inverse(self, input_coords, output_coords):
        relations = self.find_relations(input_coords, output_coords)
        inverse_relations = {}
        for cur_output, cur_input_list in relations.items():
            inverse_relations.update({cur_input: cur_output for cur_input in cur_input_list})
        return inverse_relations

    def _find_ancestors(self, input_coords, output_node):
        try:
            return [output_node.match_coords(input_coords)]
        except KeyError:
            if output_node.inherits_from is None:
                raise KeyError("Ancestor does not exist.")
            return self._find_ancestors(input_coords, self[output_node.inherits_from])


    def _find_descendants(self, input_coords, output_node):
        try:
            return [output_node.match_coords(input_coords)]
        except KeyError:
            pass

        all_descendants = []
        for item in self._items:
            if item.inherits_from == output_node.name:
                all_descendants.extend(self._find_descendants(input_coords, item))
        return all_descendants

    def rebroadcast_xarray(self, input_array, output_coords, dim="Type"):
        input_coords = input_array.coords[dim].values
        if list(input_coords) == list(output_coords):
            return input_array
        new_coords = {coord.name: coord for coord in input_array.coords.values()}
        new_coords[dim] = output_coords

        keep_coords = []
        new_array = xr.DataArray(0.0, dims=input_array.dims, coords=new_coords)
        for cur_coord in list(output_coords):
            if cur_coord in input_coords:
                keep_coords.append(cur_coord)
                continue
            parent = self.find_relations(input_coords, [cur_coord])[cur_coord][0]
            new_array.loc[{dim: cur_coord}] = input_array.loc[{dim: parent}]
        new_array.loc[{dim: keep_coords}] = input_array.loc[{dim: keep_coords}]
        return new_array

    def to_netcdf(self, out_fp, **kwargs):
        json_items = [x.to_dict() for x in self._items]
        data = xr.DataArray()
        data.attrs["knowledge_graph"] = json.dumps(json_items)
        data.to_netcdf(out_fp, **kwargs)

    @classmethod
    def from_dataarray(cls, data):
        # data = xr.open_dataarray(in_fp, **kwargs)
        json_str = data.attrs["knowledge_graph"]
        knowledge_list = json.loads(json_str)
        items = [Node.from_dict(item) for item in knowledge_list]
        return cls(*items)

    # def to_json_str(self):


def create_vehicle_graph():
    vehicle_subtypes = ["BEV", "FCV", "HEV", "ICE", "PHEV", "Trolley"]
    vehicle_supertypes = ["Cars", "Heavy Freight Trucks", "Light Commercial Vehicles",
                        "Medium Freight Trucks", "Regular Buses", "Midi Buses"]

    vehicle_knowledge_graph = KnowledgeGraph(Node("Vehicles"))
    vehicle_knowledge_graph.add(Node("Bikes", inherits_from="Vehicles"))
    vehicle_knowledge_graph.add(Node("Airplanes", inherits_from="Vehicles"))
    for subtype in ["Freight Planes", "Passenger Planes"]:
        vehicle_knowledge_graph.add(Node(subtype, inherits_from="Airplanes"))

    # Ships
    vehicle_knowledge_graph.add(Node("Ships", inherits_from="Vehicles"))
    for subtype in ["Small Ships", "Medium Ships", "Large Ships", "Very Large Ships",
                    "Inland Ships"]:
        vehicle_knowledge_graph.add(Node(subtype, inherits_from="Ships"))
    #vehicle_knowledge_graph.add(Node("Trains", inherits_from="Vehicles"))

    # Trains
    # TODO: Fix Trains -> Regular Trains
    for subtype in ["Freight Trains", "Trains", "High Speed Trains"]:
        vehicle_knowledge_graph.add(Node(subtype, inherits_from="Vehicles"))

    # Cars, Trucks, Buses
    vehicle_knowledge_graph.add(Node("Buses", inherits_from="Vehicles"))
    vehicle_knowledge_graph.add(Node("Trucks", inherits_from="Vehicles"))
    for super_type in vehicle_supertypes:
        if super_type in ["Heavy Freight Trucks", "Light Commercial Vehicles",
                          "Medium Freight Trucks"]:
            vehicle_knowledge_graph.add(Node(name=super_type, inherits_from="Trucks"))
        elif super_type in ["Regular Buses", "Midi Buses"]:
            vehicle_knowledge_graph.add(Node(name=super_type, inherits_from="Buses"))
        else:
            vehicle_knowledge_graph.add(Node(name=super_type))
        for sub_type in vehicle_subtypes:
            vehicle_knowledge_graph.add(Node(f"{super_type} - {sub_type}", inherits_from=super_type))
    return vehicle_knowledge_graph


def create_building_graph():
    building_nodes = []
    for super_type in ["Detached", "Semi-detached", "Appartment", "High-rise"]:
        building_nodes.append(Node(super_type))

        for sub_type in ["Urban", "Rural"]:
            building_nodes.append(Node(f"{super_type} - {sub_type}", inherits_from=super_type))


    building_knowledge_graph = KnowledgeGraph(*building_nodes)
    # knowledge_graph = KnowledgeGraph(*building_nodes)


    # building_knowledge_graph = KnowledgeGraph(
    #     Node("Detached"),
    #     Node("Detached - Urban", "Detached"),
    #     Node("Detached - Rural", "Detached"),
    #     Node("Detached"),
    #     Node("Detached - Urban", "Detached"),
    #     Node("Detached - Rural", "Detached"),
        
    # )
    return building_knowledge_graph


def create_region_graph():
    target_regions= ["Canada","Central Europe","China","India","Japan",
                     "USA","Western Europe","Latin America","Middle East and Northern Africa",
                    "Other Asia","Other OECD","Reforming Economies","Subsaharan Africa"]
    region_knowledge_graph = KnowledgeGraph()
        # Add target regions as main nodes
    for region in target_regions:
        region_knowledge_graph.add(Node(region))
    
    # Assign image regions stepwise to their respective target regions

    for region in ["RCAM", "BRA", "RSAM"]:
        region_knowledge_graph.add(Node(region, inherits_from="Latin America"))
    for region in ["NAF", "ME"]:
        region_knowledge_graph.add(Node(region, inherits_from="Middle East and Northern Africa"))
    for region in ["WAF", "EAF", "SAF", "RSAF"]:
        region_knowledge_graph.add(Node(region, inherits_from="Subsaharan Africa"))
    for region in [ "UKR", "STAN", "RUS"]:
        region_knowledge_graph.add(Node(region, inherits_from="Reforming Economies"))
    for region in ["KOR", "SEAS", "INDO", "RSAS"]:
        region_knowledge_graph.add(Node(region, inherits_from="Other Asia"))
    for region in ["TUR", "OCE", "MEX"]:
        region_knowledge_graph.add(Node(region, inherits_from="Other OECD"))
    region_knowledge_graph["Canada"].synonyms = ["CAN","1"]
    region_knowledge_graph["USA"].synonyms = ["USA","US","2"]
    region_knowledge_graph["Western Europe"].synonyms = ["WEU","W.Europe","11"]
    region_knowledge_graph["Central Europe"].synonyms = ["CEU","C.Europe","12"]
    region_knowledge_graph["India"].synonyms = ["INDIA","18"]
    region_knowledge_graph["China"].synonyms = ["CHN","20"]
    region_knowledge_graph["Japan"].synonyms = ["JAP","23"]
    region_knowledge_graph["BRA"].synonyms = ["Brazil","5"]
    region_knowledge_graph["EAF"].synonyms = ["E.Africa","9"]
    region_knowledge_graph["INDO"].synonyms = ["Indonesia","22"]
    region_knowledge_graph["KOR"].synonyms = ["Korea","19"]
    region_knowledge_graph["ME"].synonyms = ["M.East","17"]
    region_knowledge_graph["MEX"].synonyms = ["Mexico","3"]
    region_knowledge_graph["NAF"].synonyms = ["N.Africa","7"]
    region_knowledge_graph["OCE"].synonyms = ["Oceania","24"]
    region_knowledge_graph["RCAM"].synonyms = ["Rest C.Am.","4"]
    region_knowledge_graph["RSAM"].synonyms = ["Rest S.Am.","6"]
    region_knowledge_graph["RSAF"].synonyms = ["Rest S.Africa","26"]
    region_knowledge_graph["RSAS"].synonyms = ["Rest S.Asia","25"]
    region_knowledge_graph["RUS"].synonyms = ["Russia","16"]
    region_knowledge_graph["SEAS"].synonyms = ["SE.Asia","21"]
    region_knowledge_graph["SAF"].synonyms = ["South Africa","10"]
    region_knowledge_graph["STAN"].synonyms = ["Stan","15"]
    region_knowledge_graph["TUR"].synonyms = ["Turkey","13"]
    region_knowledge_graph["UKR"].synonyms = ["Ukraine","14"]
    region_knowledge_graph["WAF"].synonyms = ["W.Africa","8"]

    return region_knowledge_graph

knowledge_graph = KnowledgeGraph(*create_building_graph()._items, *create_vehicle_graph()._items)

def create_electricity_graph():

    # Generation -----------
    generation_types = ["Solar PV", "Solar PV residential", "CSP", "Wind onshore", "Wind offshore", 
                        "Wave", "Hydro", "Other Renewables", "Geothermal","Hydrogen power", "Nuclear", "Conv. Coal",
                        "Conv. Oil", "Conv. Natural Gas","Waste", "IGCC", "OGCC", "NG CC", "Biomass CC",
                        "Coal + CCS", "Oil/Coal + CCS", "Natural Gas + CCS", "Biomass + CCS",
                        "CHP Coal", "CHP Oil", "CHP Natural Gas", "CHP Biomass","CHP Geothermal", "CHP Hydrogen",
                        "CHP Coal + CCS", "CHP Oil + CCS", "CHP Natural Gas + CCS", "CHP Biomass + CCS"]
    # generation_types_renew = ["Solar PV", "Solar PV residential", "CSP", "Wind onshore", "Wind offshore", 
    #                     "Wave", "Other Renewables", "Geothermal", "Biomass CC","Biomass + CCS",
    #                     "CHP Biomass","CHP Geothermal", "CHP Biomass + CCS"] # "Hydro", "CHP Hydrogen",
    # generation_subtypes = ["c-Si", "a-Si", "CIGS", "CdTe", "Perovskite", 
    #                        "Fresnel Reflector", "Central Receiver", "Parabolic Trough", "Parabolic Dish",
    #                        "Geared - High Speed", "Geared - Medium speed", "Direct Drive"]
    # generation_supertypes = ["Renewables", "Fossil", "Other"]
    # generation_supertypes = ["Renewables", "Fossil", "Fossil + CCS"]

    electricity_knowledge_graph = KnowledgeGraph(Node("Electricity"))
    electricity_knowledge_graph.add(Node("Generation", inherits_from="Electricity"))
    for type in generation_types:
        electricity_knowledge_graph.add(Node(type, inherits_from="Generation"))



    # Transmission -----------
    transmission_types = ["HV", "MV", "LV"]
    transmission_subtypes = ["Lines", "Substations", "Transformers"]
    line_subtypes = ["Overhead", "Underground"]
  
    electricity_knowledge_graph.add(Node("Transmission", inherits_from="Electricity"))
    for type in transmission_types:
        electricity_knowledge_graph.add(Node(type, inherits_from="Transmission"))
        for sub_type in transmission_subtypes:
            electricity_knowledge_graph.add(Node(f"{type} - {sub_type}", inherits_from=type))
            if sub_type == "Lines":
                for line_subtype in line_subtypes:
                    electricity_knowledge_graph.add(Node(f"{type} - {sub_type} - {line_subtype}", inherits_from=f"{type} - {sub_type}"))


    # Storage -----------

    # electricity_knowledge_graph.add(Node("Storage", inherits_from="Electricity"))

    return electricity_knowledge_graph


knowledge_graph = KnowledgeGraph(*create_building_graph()._items, *create_vehicle_graph()._items, *create_electricity_graph()._items)

