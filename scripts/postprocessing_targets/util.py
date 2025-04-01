def create_region_graph():
    target_regions= ["Canada","Central Europe","China","India","Japan",
                     "USA","Western Europe","Latin America","Middle East and Northern Africa",
                    "Other Asia","Other OECD","Reforming Economies","Subsaharan Africa"]
    region_knowledge_graph = KnowledgeGraph()
        # Add target regions as main nodes
    for region in target_regions:
        region_knowledge_graph.add(Node(region))
    
    # Assign image regions stepwise to their respective target regions

    for region in ["MEX", "RCAM", "BRA", "RSAM"]:
        region_knowledge_graph.add(Node(region, inherits_from="Latin America"))
    for region in ["NAF", "ME"]:
        region_knowledge_graph.add(Node(region, inherits_from="Middle East and Northern Africa"))
    for region in ["WAF", "EAF", "SAF", "RSAF"]:
        region_knowledge_graph.add(Node(region, inherits_from="Subsaharan Africa"))
    for region in ["TUR", "UKR", "STAN", "RUS"]:
        region_knowledge_graph.add(Node(region, inherits_from="Reforming Economies"))
    for region in ["KOR", "SEAS", "INDO", "RSAS"]:
        region_knowledge_graph.add(Node(region, inherits_from="Other Asia"))
    region_knowledge_graph["Canada"].synonyms = ["CAN"]
    region_knowledge_graph["Canada"].synonyms = ["CAN"]
    region_knowledge_graph["USA"].synonyms = ["USA"]
    region_knowledge_graph["Western Europe"].synonyms = ["WEU"]
    region_knowledge_graph["Central Europe"].synonyms = ["CEU"]
    region_knowledge_graph["India"].synonyms = ["INDIA"]
    region_knowledge_graph["China"].synonyms = ["CHN"]
    region_knowledge_graph["Japan"].synonyms = ["JAP"]
    region_knowledge_graph["Other OECD"].synonyms = ["OCE"]
    
    return region_knowledge_graph

def verify_image_regions(region_knowledge_graph):
    image_regions = [
        'CAN', 'USA', 'MEX', 'RCAM', 'BRA', 'RSAM', 'NAF', 'WAF', 'EAF',
        'SAF', 'WEU', 'CEU', 'TUR', 'UKR', 'STAN', 'RUS', 'ME', 'INDIA',
        'KOR', 'CHN', 'SEAS', 'INDO', 'JAP', 'OCE', 'RSAS', 'RSAF'
    ]
    
    missing_regions = [region for region in image_regions if region not in region_knowledge_graph]
    
    if missing_regions:
        print("Missing regions in knowledge graph:", missing_regions)
    else:
        print("All image regions are included in the knowledge graph!")
