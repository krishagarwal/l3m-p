import logging
import sys

def load_graph(db_name: str, graph_name: str):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    BUILD_INDEX = False

    from .age import AgeGraphStore
    graph_store = AgeGraphStore(
        dbname=db_name,
        user="postgres",
        password="password",
        host="localhost",
        port=5432,
        graph_name=graph_name,
        node_label="entity"
    )

    cur = graph_store.cursor()
    entities_query = "SELECT DISTINCT ON (instance_of.entity_id) \
                        instance_of.entity_id AS id, ea_str.attribute_value AS name, 'instance' as type \
                        FROM instance_of INNER JOIN entity_attributes_str AS ea_str \
                        ON instance_of.entity_id = ea_str.entity_id AND ea_str.attribute_name = 'name' \
                        AND instance_of.concept_name != 'pose' and instance_of.concept_name != 'region' and instance_of.concept_name != 'map' \
                        UNION SELECT concepts.entity_id AS id, concepts.concept_name AS name, 'concept' as type FROM concepts \
                        ORDER BY id ASC"

    cur.execute(entities_query)
    for row in cur.fetchall():
        id = row[0]
        name = row[1]
        type = row[2]
        cur.execute(f"SELECT * FROM cypher('{graph_name}', $$CREATE (a:entity {{id: '{id}', name: '{name}', type: '{type}'}}) RETURN a $$) as (a agtype);")
        
    def add_attribute_to_graph(attribute, val_type, query, cur, graph_store):
        if val_type == 'id':
            upsert_triplet = graph_store.upsert_triplet_entity
        else:
            upsert_triplet = getattr(graph_store, "upsert_triplet_" + val_type)
        cur.execute(query)
        for row in cur.fetchall():
            upsert_triplet(row[0], attribute, row[1])
    
    value_types = ["id", "bool", "float", "int", "str"]
    for value_type in value_types:
        attributes_query = f"SELECT * FROM attributes WHERE attributes.type = '{value_type}' and attributes.attribute_name != 'name'"
        cur.execute(attributes_query)
        for row in cur.fetchall():
            attribute = row[0]
            attribute_query = f"SELECT ea.entity_id AS start_id, \
                                ea.attribute_value AS end_value \
                                FROM entity_attributes_{value_type} as ea \
                                WHERE ea.attribute_name = '{attribute}'"
            add_attribute_to_graph(attribute, value_type, attribute_query, cur, graph_store)

    # instance_of_query = "SELECT instance_of.entity_id AS start_id, \
    #                         concepts.entity_id AS end_id \
    #                         FROM instance_of \
    #                         INNER JOIN concepts ON instance_of.concept_name = concepts.concept_name \
    #                         WHERE instance_of.concept_name != 'pose' and instance_of.concept_name != 'region' and instance_of.concept_name != 'map' \
    #                         ORDER BY start_id ASC "
    # add_attribute_to_graph("instance_of", "id", instance_of_query, cur, graph_store)
    cur.close()

if __name__ == "__main__":
    load_graph("knowledge_base", "knowledge_graph")