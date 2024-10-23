from graphrag.indexing.upsert import upsert_data_and_create_graph
from graphrag.query.vector_search import _similarity_search
from graphrag.query.types import Node
from graphrag.llm.llm import extract_keywords_from_query

from typing import Any, Dict

import networkx as nx


async def query_graph(query: str) -> Any:

    graph: nx.Graph = await upsert_data_and_create_graph(entities=[], 
                                                         relationships=[], 
                                                         chunks=[])
    
    keywords = await extract_keywords_from_query(query=query, return_all=True)
    entities_with_scores = await _similarity_search(text=keywords, table='entity', top_k=60) # FIXME -> do not hardcode top_k

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
    
    chunk_ids = [node['chunk_id'].split(", ") for node in nodes]
        
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
    order_range = 5 # FIXME -> do not hardcode this
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
