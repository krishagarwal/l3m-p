from typing import Any, Dict, List, Optional
from llama_index.core.graph_stores.types import GraphStore

try:
    import psycopg2
except ImportError:
    raise ImportError("Please install psycopg2")

class AgeGraphStore(GraphStore): # type: ignore
    def __init__(
        self,
        dbname: str,
        user: str,
        password: str,
        host: str,
        port: int,
        graph_name: str,
        node_label: str,
        **kwargs: Any,
    ) -> None:
        try:
            self._conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
            self._conn.autocommit = True
            cur = self._conn.cursor()
            cur.execute(f"LOAD 'age'")
            cur.execute(f"SET search_path = ag_catalog, '$user', public;")
        except psycopg2.OperationalError as err:
            raise ValueError(err)
        self._dbname = dbname
        self._graph_name = graph_name
        self._node_label = node_label
        cur.execute("SELECT entities.entity_id, attribute_value \"name\" FROM entities JOIN entity_attributes_str ON entities.entity_id = entity_attributes_str.entity_id WHERE attribute_name = 'name';")
        results = cur.fetchall()
        self.id_to_name = {row[0] : row[1] for row in results}
        self.name_to_id = {row[1] : row[0] for row in results}

    def cursor(self):
        return self._conn.cursor()

    def get(self, subj: str) -> List[List[str]]:
        """Get triplets."""
        query = (
                    f"SELECT * FROM ag_catalog.cypher('{self._graph_name}', $$ "
                    f"MATCH (:{self._node_label} {{name:'{subj}'}})-[r]->(n2:{self._node_label})"
                    f"RETURN type(r), n2.name"
                    f"$$) as (rel agtype, obj agtype);"
        )
        cur = self.cursor()
        cur.execute(query)
        results = cur.fetchall()
        return [[eval(rel), eval(obj)] for (rel, obj) in results]

    def get_rel_map(
            self, subjs: Optional[List[str]] = None, depth: int = 2, limit: int=30
    ) -> Dict[str, List[List[str]]]:
        """Get flat rel map."""

        rel_map: Dict[Any, List[Any]] = {}
        if subjs is None or len(subjs) == 0:
            # unlike simple graph_store, we don't do get_all here
            return rel_map
        
        subjs = [subj.lower().replace('"', '') for subj in subjs]

        # max 100 can be processed at a time by db
        for i in range(depth):
            path = f"-[]-(:{self._node_label})" * i
            per_iter = 50 // (2 ** i)
            for j in range(0, len(subjs), per_iter):
                subjs_str = '["' + '", "'.join(subjs[j:j+per_iter]) + '"]'
                query = (f"SELECT * FROM ag_catalog.cypher('{self._graph_name}', $$ "
                        f"MATCH p=(n1:{self._node_label}){path}-[]-() "
                        f"WHERE n1.name IN {subjs_str} "
                        f"WITH n1.name AS subj, p, relationships(p) AS rels "
                        f"UNWIND rels AS rel "
                        f"WITH startNode(rel).name AS subj, p, collect([startNode(rel).name, type(rel), endNode(rel).name]) AS predicates "
                        f"RETURN subj, predicates LIMIT {limit}"
                        f"$$) as (subj agtype, rel agtype);"
                        )
                cur = self.cursor()
                try:
                    cur.execute(query)
                except Exception as err:
                    print("Query:", query)
                    print("Subjects:", subjs_str)
                    raise err
                results = cur.fetchall()
                for row in results:
                    for rel in eval(row[1]):
                        rel_str = "" + rel[0] + ", -[" + rel[1] + "], " + "-> " + rel[2] + ""
                        key = eval(row[0])
                        if key not in rel_map:
                            rel_map[key] = [rel_str]
                        elif rel_str not in rel_map[eval(row[0])]:
                            rel_map[eval(row[0])].append(rel_str)

        return rel_map

    def upsert_triplet(self, subj: str, rel: str, obj: str) -> None:
        """Add triplet."""
        cur = self.cursor()
        cur.execute(
            f"SELECT * FROM cypher('{self._graph_name}', "
            f"$$MERGE (u:{self._node_label} {{name: '{subj}'}})"
            f"MERGE (v:{self._node_label} {{name: '{obj}'}}) "
            f"MERGE (u)-[e:{rel}]->(v) $$) as (e agtype);")

    def upsert_triplet_entity(self, subj: str, rel: str, obj: str) -> None:
        """Add triplet with entity value."""
        cur = self.cursor()
        cur.execute(
            f"SELECT * FROM cypher('{self._graph_name}', "
            f"$$MERGE (a:{self._node_label} {{id: '{subj}' }}) "
            f"RETURN a $$) as (a agtype);")
        cur.execute(
            f"SELECT * FROM cypher('{self._graph_name}', "
            f"$$MERGE (a:{self._node_label} {{id: '{obj}' }}) "
            f"RETURN a $$) as (a agtype);")
        cur.execute(
            f"SELECT * FROM cypher('{self._graph_name}', $$MATCH (u:{self._node_label} {{id: '{subj}'}}), "
            f"(v:{self._node_label} {{id: '{obj}'}}) CREATE (u)-[e:{rel}]->(v) RETURN e$$) as (e agtype);")

    def upsert_triplet_bool(self, subj: str, rel: str, obj_bool: bool) -> None:
        """Add triplet with bool value."""
        obj = str(obj_bool).lower()
        cur = self.cursor()
        cur.execute(
            f"SELECT * FROM cypher('{self._graph_name}', "
            f"$$MERGE (a:bool {{name: '{obj}' }}) "
            f"RETURN a $$) as (a agtype);")
        cur.execute(
            f"SELECT * FROM cypher('{self._graph_name}', $$MATCH (u:{self._node_label} {{id: '{subj}'}}), "
            f"(v:bool {{name: '{obj}'}}) CREATE (u)-[e:{rel}]->(v) RETURN e$$) as (e agtype);")

    def upsert_triplet_float(self, subj: str, rel: str, obj: float) -> None:
        """Add triplet with float value."""
        cur = self.cursor()
        cur.execute(
            f"SELECT * FROM cypher('{self._graph_name}', "
            f"$$MERGE (a:float {{name: '{obj}' }}) "
            f"RETURN a $$) as (a agtype);")
        cur.execute(
            f"SELECT * FROM cypher('{self._graph_name}', $$MATCH (u:{self._node_label} {{id: '{subj}'}}), "
            f"(v:float {{name: '{obj}'}}) CREATE (u)-[e:{rel}]->(v) RETURN e$$) as (e agtype);")
    
    def upsert_triplet_int(self, subj: str, rel: str, obj: int) -> None:
        """Add triplet with int value."""
        cur = self.cursor()
        cur.execute(
            f"SELECT * FROM cypher('{self._graph_name}', "
            f"$$MERGE (a:int {{name: '{obj}' }}) "
            f"RETURN a $$) as (a agtype);")
        cur.execute(
            f"SELECT * FROM cypher('{self._graph_name}', $$MATCH (u:{self._node_label} {{id: '{subj}'}}), "
            f"(v:int {{name: '{obj}'}}) CREATE (u)-[e:{rel}]->(v) RETURN e$$) as (e agtype);")

    def upsert_triplet_str(self, subj: str, rel: str, obj: str) -> None:
        """Add triplet with string value."""
        cur = self.cursor()
        cur.execute(
            f"SELECT * FROM cypher('{self._graph_name}', "
            f"$$MERGE (a:str {{name: '{obj}' }}) "
            f"RETURN a $$) as (a agtype);")
        cur.execute(
            f"SELECT * FROM cypher('{self._graph_name}', $$MATCH (u:{self._node_label} {{id: '{subj}'}}), "
            f"(v:str {{name: '{obj}'}}) CREATE (u)-[e:{rel}]->(v) RETURN e$$) as (e agtype);")

    def delete(self, subj: str, rel: str, obj: str) -> None:
        """Delete triplet."""
        cur = self.cursor()

        def check_edges(entity: str) -> bool:
            cur.execute(
                f"SELECT * FROM cypher('{self._graph_name}', "
                f"$$MATCH (u {{name: '{entity}'}})-[]-(v) "
                f"RETURN v $$) as (v agtype);")
            results = cur.fetchall()
            return bool(len(results))

        def delete_entity(entity: str) -> None:
            cur.execute(
                f"SELECT * FROM cypher('{self._graph_name}', "
                f"$$MATCH (u {{name: '{entity}'}}) DELETE u$$) as (u agtype);")

        def delete_rel(subj: str, obj: str, rel: str) -> None:
            cur.execute(
                f"SELECT * FROM cypher('{self._graph_name}', "
                f"$$MATCH (u {{name: '{subj}'}})-[e:{rel}]->(v {{name: '{obj}'}}) DELETE e$$) as (e agtype);")

        delete_rel(subj, obj, rel)
        # if not check_edges(subj):
        #     delete_entity(subj)
        # if not check_edges(obj):
        #     delete_entity(obj)

    def delete_rel_with_subj(self, subj: str, rel: str) -> None:
        """Delete triplet with subj and rel."""
        cur = self.cursor()
        cur.execute(
            f"SELECT * FROM cypher('{self._graph_name}', "
            f"$$MATCH (u:{self._node_label} {{name: '{subj}'}})-[e:{rel}]->() DELETE e$$) as (e agtype);")

    def delete_rel_with_obj(self, rel: str, obj: str) -> None:
        """Delete triplet with obj and rel."""
        cur = self.cursor()
        cur.execute(
            f"SELECT * FROM cypher('{self._graph_name}', "
            f"$$MATCH (u)-[e:{rel}]->(v:{self._node_label} {{name: '{obj}'}}) DELETE e$$) as (e agtype);")

    def query(self, query: str, param_map: Optional[Dict[str, Any]] = {}, return_count: int = 1) -> Any:
        cur = self.cursor()
        if param_map: # only format if param map isn't empty
            query = query.format(param_map)
        return_list = ", ".join(f'"{i}" agtype' for i in range(return_count))
        try:
            cur.execute(
                f"SELECT * FROM cypher('{self._graph_name}', "
                f"$${query}$$) as ({return_list});")
            results = cur.fetchall()
        except:
            return None
        return results
    
    def rel_exists(self, subj: str, rel: str, obj: str) -> bool:
        result = self.query(f"MATCH (V {{name: '{subj}'}})-[:{rel}]-(V2 {{name: '{obj}'}}) RETURN COUNT(V) > 0")
        return result is not None and len(result) == 1 and result[0] is not None and len(result[0]) == 1 and result[0][0] == 'true'