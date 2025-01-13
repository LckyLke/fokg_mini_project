from rdflib import Graph, URIRef

def enrich_kg_with_hierarchy(kg_file, hierarchy_file, output_file):
    # Namespaces
    rdf_type = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    rdfs_subclass = URIRef("http://www.w3.org/2000/01/rdf-schema#subClassOf")
    
    # Load the graphs
    kg = Graph()
    hierarchy = Graph()
    kg.parse(kg_file, format="nt")
    hierarchy.parse(hierarchy_file, format="nt")

    # Build a subclass-to-superclass mapping
    subclass_to_superclass = {}
    for subclass, _, superclass in hierarchy.triples((None, rdfs_subclass, None)):
        if subclass not in subclass_to_superclass:
            subclass_to_superclass[subclass] = set()
        subclass_to_superclass[subclass].add(superclass)

    # Expand types in the KG
    new_statements = set()
    for entity, _, typ in kg.triples((None, rdf_type, None)):
        current_types = {typ}
        while current_types:
            next_type = current_types.pop()
            if next_type in subclass_to_superclass:
                for superclass in subclass_to_superclass[next_type]:
                    new_statement = (entity, rdf_type, superclass)
                    if new_statement not in kg:
                        new_statements.add(new_statement)
                        print(f"Added type relation: {new_statement}")
                        current_types.add(superclass)

    # Add new statements to the KG
    for stmt in new_statements:
        kg.add(stmt)

    # Write the enriched KG to a file
    kg.serialize(destination=output_file, format="nt")


