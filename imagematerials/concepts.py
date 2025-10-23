import json
from dataclasses import dataclass, field
from typing import Optional

import xarray as xr
import prism


@dataclass
class Node():
    name: str
    synonyms: list[str] = field(default_factory=list)
    inherits_from: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.inherits_from is None:
            self.inherits_from = []
        elif isinstance(self.inherits_from, str):
            self.inherits_from = [self.inherits_from]

        if self.name in self.synonyms:
            raise ValueError(f"Cannot create node that has its own name as a synonym: {self}.")

        for parent in self.inherits_from:
            try:
                self.match_coords(parent)
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
        for name in [node.name] + node.synonyms:
            if name in self:
                raise ValueError(f"Node {name} already exists.")
        for parent in node.inherits_from:
            if parent not in self:
                raise ValueError(f"Parent {parent} of {node.name} does not exist.")

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

        ancestors = self._find_ancestors(input_coords, output_node)

        try:
            descendants = self._find_descendants(input_coords, output_node)
        except KeyError:
            descendants = []

        if len(ancestors) > 0 and len(descendants) > 0:
            raise ValueError(f"Cannot find relations, because {output_node} has both ancestors ({ancestors}) "
                             f"and descendants ({descendants})")
        if len(ancestors) + len(descendants) == 0:
            raise ValueError(f"Cannot find relations of {output_node} in input coordinates.")

        return list(set(ancestors + descendants))

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
            ancestors = []
            for parent in output_node.inherits_from:
                try:
                    ancestors.extend(self._find_ancestors(input_coords, self[parent]))
                except KeyError:
                    pass
        return ancestors


    def _find_descendants(self, input_coords, output_node):
        try:
            return [output_node.match_coords(input_coords)]
        except KeyError:
            pass

        all_descendants = []
        for item in self._items:
            if output_node.name in item.inherits_from:
                all_descendants.extend(self._find_descendants(input_coords, item))
        return all_descendants

    def rebroadcast_xarray(self, input_array, output_coords, dim="Type", shares=None, dim_shares=None):
        """Disaggregates supertypes into subtypes. If shares for the subtypes are provided,
        the values of the input_array data is adjusted accordingly (value of supertype * shares = values of subtypes).
        If no shares are provided, the value of the supertype is taken for all subtypes.

        Parameters
        ----------
        input_array
            The xr.DataArray to be disaggregated.
        output_coords
            The output coordinates (subtypes).
        dim, optional
            Dimension of the input_array to disaggregate over, by default "Type"
        shares, optional
            Shares to be used for rebroadcasting, by default None. Must have the dimension "Type".
        dim_shares, optional
            Dimension of the shares DataArray. By default None. If shares is defined and dim_shares not, dim is used.
            In case the shares have a different dimension name than the input_array, it can be specified here.

        Returns
        -------
            Disaggregated xr.DataArray.

        """

        if shares is not None and dim_shares is None:
            dim_shares = dim

        input_coords = input_array.coords[dim].values
        if list(input_coords) == list(output_coords):
            return input_array
        new_coords = {coord.name: coord for coord in input_array.coords.values()}
        new_coords[dim] = output_coords

        keep_coords = []
        new_array = xr.DataArray(0.0, dims=input_array.dims, coords=new_coords)
        if input_array.pint.units:
            new_array = prism.Q_(new_array, input_array.pint.units)
        for cur_coord in list(output_coords):
            if cur_coord in input_coords:
                keep_coords.append(cur_coord)
                continue
            relations = self.find_relations(input_coords, [cur_coord])[cur_coord]
            if len(relations) > 1:
                raise ValueError("Cannot rebroadcast DataArray, because multiple input relations "
                                 f"exist ({relations}) for output coordinate {cur_coord}. For "
                                 "aggregation use the aggregate_sum method instead.")
            parent = relations[0]
            if shares is not None and cur_coord in shares.coords[dim_shares]:
                # disaggregate according to shares
                new_array.loc[{dim: cur_coord}] = (input_array.loc[{dim: parent}]
                                                   * shares.loc[{dim_shares: cur_coord}])
            else:
                # disaggregate, by just taking the value of the parent
                new_array.loc[{dim: cur_coord}] = input_array.loc[{dim: parent}]
        new_array.loc[{dim: keep_coords}] = input_array.loc[{dim: keep_coords}]
        # check if input array had a unit and if so, reapply this unit to new array

        return new_array

    def aggregate_sum(self, input_array, output_coords, dim="Type"):
        """Aggregate the data over subtypes and sums them.

        Parameters
        ----------
        input_array
            The xr.DataArray to be aggregated/summed.
        output_coords
            The output coordinates over which to aggregate/sum.
        dim, optional
            Dimension to aggregate/sum over, by default "Type"

        Returns
        -------
            Aggregated xr.DataArray.

        """
        output_to_input = self.find_relations(input_array.coords[dim].values, output_coords)
        output_arrays = []
        for output, input_list in output_to_input.items():
            output_arrays.append(input_array.sel(**{dim: input_list}).sum(dim))
        output_xr = xr.concat(output_arrays, dim=dim)
        output_xr.coords[dim] = output_coords
        return output_xr

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
    building_nodes = [
    Node("Buildings"),

]
    for super_type in ["Detached", "Semi-detached", "Appartment", "High-rise"]:
        building_nodes.append(Node(super_type, inherits_from="Buildings"))

        for sub_type in ["Urban", "Rural"]:
            building_nodes.append(Node(f"{super_type} - {sub_type}", inherits_from=super_type))

    for c in ["Retail+", "Hotels+", "Office", "Govt+"]:
        building_nodes.append(Node(c, inherits_from="Buildings"))

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
    #TODO move to seperate file
    
    region_knowledge_graph = KnowledgeGraph()

    # Add target regions as main nodes (non-numeric)
    target_regions = [
        "Latin America", "Middle East and Northern Africa",
        "Other Asia", "Other OECD", "Reforming Economies", "Subsaharan Africa"
    ]
    for region in target_regions:
        region_knowledge_graph.add(Node(region))

    # Add numeric region nodes first with full synonyms
    numeric_region_map = {
        "1": ["Canada", "CAN"],
        "2": ["USA", "US"],
        "3": ["Mexico", "MEX"],
        "4": ["Rest of Central America", "RCAM", "Rest C.Am."],
        "5": ["Brazil", "BRA"],
        "6": ["Rest of South America", "RSAM", "Rest S.Am."],
        "7": ["Northern Africa", "NAF", "N.Africa"],
        "8": ["Western Africa", "WAF", "W.Africa"],
        "9": ["Eastern Africa", "EAF", "E.Africa"],
        "10": ["South Africa", "SAF"],
        "11": ["Western Europe", "WEU", "W.Europe"],
        "12": ["Central Europe", "CEU", "C.Europe"],
        "13": ["Turkey", "TUR"],
        "14": ["Ukraine +", "UKR", "Ukraine"],
        "15": ["Asian-Stan", "STAN", "Stan"],
        "16": ["Russia +", "RUS", "Russia"],
        "17": ["Middle East", "ME", "M.East"],
        "18": ["India +", "INDIA", "India"],
        "19": ["Korea", "KOR"],
        "20": ["China +", "CHN", "China"],
        "21": ["Southeastern Asia", "SEAS", "SE.Asia"],
        "22": ["Indonesia +", "INDO", "Indonesia"],
        "23": ["Japan", "JAP"],
        "24": ["Oceania", "OCE"],
        "25": ["Rest of South Asia", "RSAS", "Rest S.Asia"],
        "26": ["Rest of Southern Africa", "RSAF", "Rest S.Africa"]
    }

    subregion_division = {
        "Latin America": {"RCAM", "BRA", "RSAM"},
        "Middle East and Northern Africa": {"NAF", "ME"},
        "Subsaharan Africa": {"WAF", "EAF", "SAF", "RSAF"},
        "Reforming Economies": {"UKR", "STAN", "RUS"},
        "Other Asia": {"KOR", "SEAS", "INDO", "RSAS"},
        "Other OECD": {"TUR", "OCE", "MEX"},
    }

    for number, synonyms in numeric_region_map.items():
        inherits_from = None
        for super_region, regions in subregion_division.items():
            if len(regions.intersection(synonyms)) > 0:
                inherits_from = super_region
                break
        region_knowledge_graph.add(Node(number, synonyms=synonyms, inherits_from=inherits_from))

    return region_knowledge_graph

knowledge_graph = KnowledgeGraph(*create_building_graph()._items, *create_vehicle_graph()._items)

def create_electricity_graph():

    # Generation ======================================================================================
    
    numeric_generation_map = {
        "1": ["Solar PV", "SPV"],               # Solar PV power (central)
        "2": ["Solar PV residential", "SPVR"],  # Solar PV power (decentral/residential)
        "3": ["CSP"],                           # Concentrated Solar Power 
        "4": ["Wind onshore", "WON"],           # Onshore wind power
        "5": ["Wind offshore", "WOFF"],         # Offshore wind power
        "6": ["Wave", "WAVE"],                  # Wave power
        "7": ["Hydro", "HYD"],                  # Hydro power
        "8": ["Other Renewables", "OREN"],      # Other renewables (tidal and geothermal power)
        "9": ["Geothermal", "GEO"],             # Geothermal power
        "10": ["Hydrogen power", "H2P"],        # Hydrogen to power
        "11": ["Nuclear", "NUC"],               # Nuclear
        "12": ["<EMPTY>", "FREE12"],            # Free spot
        "13": ["Conv. Coal", "ClST"],           # Coal steam turbine
        "14": ["Conv. Oil", "OlST"],            # Oil steam turbine
        "15": ["Conv. Natural Gas", "NGOT"],    # NG open cycle turbine
        "16": ["Waste", "BioST"],               # Biomass steam turbine
        "17": ["IGCC"],                         # Integrated gasification combined cycle
        "18": ["OGCC", "OlCC"],                 # Oil combined cycle
        "19": ["NG CC", "NGCC"],                # NG combined cycle
        "20": ["Biomass CC", "BioCC"],          # Biomass combined cycle
        "21": ["Coal + CCS", "ClCS"],           # Coal carbon capture and storage
        "22": ["Oil/Coal + CCS", "OlCS"],       # Oil carbon capture and storage
        "23": ["Natural Gas + CCS", "NGCS"],    # NG carbon capture and storage
        "24": ["Biomass + CCS", "BioCS"],       # Biomass carbon capture and storage
        "25": ["CHP Coal", "ClCHP"],            # Coal combined heat and power
        "26": ["CHP Oil", "OlCHP"],             # Oil combined heat and power
        "27": ["CHP Natural Gas", "NGCHP"],     # NG combined heat and power
        "28": ["CHP Biomass", "BioCHP"],        # Biomass combined heat and power
        "29": ["CHP Coal + CCS", "ClCHPCS"],    # Coal combined heat and power carbon capture and storage
        "30": ["CHP Oil + CCS", "OlCHPCS"],     # Oil combined heat and power carbon capture and storage
        "31": ["CHP Natural Gas + CCS", "NGCHPCS"], # NG combined heat and power carbon capture and storage
        "32": ["CHP Biomass + CCS", "BioCHPCS"],    # Biomass combined heat and power carbon capture and storage
        "33": ["CHP Geothermal", "GeoCHP"],         # Geothermal combined heat and power
        "34": ["CHP Hydrogen", "H2CHP"]             # Hydrogen combined heat and power
    }


    electricity_knowledge_graph = KnowledgeGraph(Node("Electricity"))
    electricity_knowledge_graph.add(Node("Generation", inherits_from="Electricity"))
    for number, synonyms in numeric_generation_map.items():
        electricity_knowledge_graph.add(Node(number, synonyms=synonyms, inherits_from="Generation"))

    # # Sub-Technologies --------------------------

    # for subtype in ["c-Si", "a-Si", "CIGS", "CdTe", "Perovskite"]:
    #     electricity_knowledge_graph.add(Node(subtype, inherits_from="Solar PV"))
    #     electricity_knowledge_graph.add(Node(subtype, inherits_from="Solar PV residential")) # is this possible?

    # for subtype in ["Fresnel Reflector", "Central Receiver", "Parabolic Trough", "Parabolic Dish"]:
    #     electricity_knowledge_graph.add(Node(subtype, inherits_from="CSP"))

    # for subtype in ["Geared - High Speed", "Geared - Medium speed", "Direct Drive"]:
    #     electricity_knowledge_graph.add(Node(subtype, inherits_from="Wind onshore"))
    #     electricity_knowledge_graph.add(Node(subtype, inherits_from="Wind offshore"))

    # # Category Relations -------------------------

    # # Renewables
    # electricity_knowledge_graph.add(Node("Renewables", inherits_from="Generation"))
    # for supertype in ["Solar PV", "Solar PV residential", "CSP", "Wind onshore", "Wind offshore",
    #                   "Wave", "Other Renewables", "Geothermal", "Biomass CC","Biomass + CCS",
    #                   "CHP Biomass","CHP Geothermal", "CHP Biomass + CCS"]:
    #     electricity_knowledge_graph.add(Node(supertype, inherits_from="Renewables"))

    # # Climate Neutral
    # electricity_knowledge_graph.add(Node("Climate Neutral", inherits_from="Generation"))
    # for supertype in ["Solar PV", "Solar PV residential", "CSP", "Wind onshore", "Wind offshore",
    #                   "Wave", "Other Renewables", "Geothermal", "Biomass CC","Biomass + CCS",
    #                   "CHP Biomass","CHP Geothermal", "CHP Biomass + CCS","Coal + CCS", "Oil/Coal + CCS", 
    #                   "Natural Gas + CCS", "Nuclear"]:
    #     electricity_knowledge_graph.add(Node(supertype, inherits_from="Climate Neutral"))

    # # Fossil
    # electricity_knowledge_graph.add(Node("Fossil", inherits_from="Generation"))
    # for supertype in ["Conv. Coal", "Conv. Oil", "Conv. Natural Gas"]:
    #     electricity_knowledge_graph.add(Node(supertype, inherits_from="Fossil"))


    # Transmission Grid ============================================================================
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


    # Storage ======================================================================================
    storage_supertypes = ["PHS", "V2G-Batteries", "Other Storage"] # Pumped Hydro Storage, Vehicle-to-Grid Batteries
    # Storage calculations follow a 3 tiered structure: Demand is filled first with PHS, then with (anyway available) V2G-Batteries, and 
    # the residual demand with Other Storage
    storage_subtypes = ["Flywheel", "Compressed Air", "Hydrogen FC", "NiMH", "Deep-cycle Lead-Acid", "LMO",
                        "NMC", "NCA", "LFP", "LTO", "Zinc-Bromide", "Vanadium Redox", "Sodium-Sulfur", "ZEBRA",
                        "Lithium Sulfur", "Lithium Ceramic", "Lithium-air"]

    electricity_knowledge_graph.add(Node("Storage", inherits_from="Electricity"))

    for supertype in storage_supertypes:
        electricity_knowledge_graph.add(Node(supertype, inherits_from="Storage"))
    # TODO: V2G Batteries are also related to Vehicles + Add V2G-Batteries subtypes?
    for subtype in storage_subtypes:
        electricity_knowledge_graph.add(Node(subtype, inherits_from="Other Storage"))
    

    return electricity_knowledge_graph


knowledge_graph = KnowledgeGraph(*create_building_graph()._items, *create_vehicle_graph()._items, *create_electricity_graph()._items)

