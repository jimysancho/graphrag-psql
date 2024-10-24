from graphrag.indexing.upsert import upsert_data_and_create_graph
from graphrag.query.vector_search import _similarity_search
from graphrag.query.types import Node, Edge
from graphrag.llm.llm import extract_keywords_from_query
from graphrag.database.base import get_db
from graphrag.database.models import Relationship, Entity

from typing import Any, Dict, Tuple, List
from collections import Counter
from sqlalchemy.orm import aliased

import networkx as nx


async def local_query_graph(query: str, top_k: int, order_range: int) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:

    graph: nx.Graph = await upsert_data_and_create_graph(entities=[], 
                                                         relationships=[], 
                                                         chunks=[])
    
    keywords = await extract_keywords_from_query(query=query, return_all=True)
    entities_with_scores = await _similarity_search(text=keywords, table='entity', top_k=top_k)

    entities = [entity for entity, _ in entities_with_scores]
    
    entity_nodes = [graph.nodes.get(entity.entity_name) for entity in entities]

    node_models = [Node(element=entity_node, degree=graph.degree(entity_node['entity_name'])) for entity_node in entity_nodes]
    nodes = [
        {**node.model_dump()['element'], 'degree': node.model_dump()['degree']}
        for node in node_models
    ]

    edges = [
        list(graph.edges(node['entity_name'])) for node in nodes
    ]
        
    neighbors = []
    for node in nodes:
        neighbors.extend(
            list(nx.neighbors(graph, node['entity_name']))
        )
    
    neighbors_nodes = [graph.nodes.get(neighbor) for neighbor in neighbors]
    neighbor_chunk_ids_mapping = {
        node['entity_name']: set(node['chunk_id'].split(", ")) for node in neighbors_nodes if node
    }
    
    connected_nodes: Dict[str, Dict[str, Any]] = {}

    for index, (keyword_node, keyword_edges) in enumerate(zip(nodes, edges)):
        node: str = keyword_node['entity_name']
        del keyword_node['entity_name']
        if node in connected_nodes: continue
        connected_nodes[node] = {}
        chunk_ids = keyword_node['chunk_id'].split(", ")
        for chunk_id in chunk_ids:
            if chunk_id not in connected_nodes[node]: 
                connected_nodes[node][chunk_id] = {"relation_counts": 0, "keywords": set(), "order": index, "graph_node": keyword_node}
            for edge in keyword_edges:
                neighbor = edge[1]
                neighbor_data = neighbor_chunk_ids_mapping[neighbor]
                if chunk_id in neighbor_data:
                    connected_nodes[node][chunk_id]['relation_counts'] += 1
                    connected_nodes[node][chunk_id]['keywords'].add(neighbor)

    final_result = {}
    for node, chunk_data in connected_nodes.items():
        for chunk_id, data in chunk_data.items():
            if chunk_id not in final_result:
                final_result[chunk_id] = {"node": node, **data}
            else:
                if abs(final_result[chunk_id]['order'] - order_range) <= data['order'] <= abs(final_result[chunk_id]['order'] + order_range):
                    if data['relation_counts'] > final_result[chunk_id]['relation_counts']:
                        final_result[chunk_id] = {"node": node, **data}

    connected_nodes = final_result
            
    return connected_nodes, keywords


async def global_query_graph(query: str, top_k: int, order_range: int) -> Tuple[Counter, List[str]]:
    
    db = next(get_db())

    graph: nx.Graph = await upsert_data_and_create_graph(entities=[], 
                                                         relationships=[], 
                                                         chunks=[])
    
    keywords = await extract_keywords_from_query(query=query, return_all=True)
    relationships_with_scores = await _similarity_search(text=keywords, table='relationship', top_k=top_k)
    relationships = [r for r, _ in relationships_with_scores]
    
    source_entity, target_entity = aliased(Entity), aliased(Entity)
    relationship_with_entities = db.query(Relationship, source_entity.entity_name, target_entity.entity_name)\
        .join(source_entity, Relationship.source_id == source_entity.entity_id)\
            .join(target_entity, Relationship.target_id == target_entity.entity_id)\
                .filter(Relationship.relationship_id.in_([r.relationship_id for r in relationships]))\
                    .all()
    
    relationship_edges = [
        graph.edges.get((source, target)) for (_, source, target) in relationship_with_entities
    ]

    db.close()
    edge_models = [
        Edge(edge=edge, degree=graph.degree(source_entity) + graph.degree(target_entity)) for (edge, (_, source_entity, target_entity)) in zip(relationship_edges, relationship_with_entities)
    ]
    
    edges = [
        {**edge.model_dump()['edge'], 'degree': edge.model_dump()['degree']}
        for edge in edge_models
    ]
    
    entity_names_of_edges = set()
    for (_, s, t) in relationship_with_entities:
        entity_names_of_edges.add(s)
        entity_names_of_edges.add(t)
     
    chunk_ids = []
    for edge in edges: chunk_ids.extend(edge['chunk_id'].split(", "))
    
    return Counter(chunk_ids), keywords
