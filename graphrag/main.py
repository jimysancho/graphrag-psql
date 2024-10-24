from graphrag.indexing import (create_chunks, 
                               upsert_data_and_create_graph, 
                               extract_entities)
from graphrag.query.generate import _local_query, _global_query, _hybrid_query, _naive_rag

from graphrag.config import GlobalConfig
from typing import List, Dict, Tuple, Any, Set

import networkx as nx


async def insert(text: str, config: GlobalConfig) -> nx.Graph:
    chunks = await create_chunks(text=text, 
                                 min_token_size=config.min_chunk_size, 
                                 max_token_size=config.max_chunk_size)
    print(f"{len(chunks)} chunks created")
    entities, relationships, kept_vs_merged, chunk_models = await extract_entities(chunks=chunks, entity_types=config.entity_types, gleaning=config.max_gleaning, batch=config.batch)
    print(f"{len(entities)} entities extracted and {len(relationships)} relationships extracted. ")
    graph = await upsert_data_and_create_graph(entities=entities, relationships=relationships, chunks=chunk_models)
    return graph

async def local_query(query: str, config: GlobalConfig) -> Tuple[str | None, List[str], Dict[str, Dict[str, Any]], List[str]]:
    response, chunk_texts, nodes, keywords = await _local_query(query=query, top_k=config.keywords_top_k, max_nodes=config.graph_top_k, order_range=config.order_range)
    return response, chunk_texts, nodes, keywords

async def global_query(query: str, config: GlobalConfig) -> Tuple[str | None, List[str], Dict[str, Any], List[str]]:
    response, chunk_texts, chunks, keywords = await _global_query(query=query, top_k=config.keywords_top_k, max_nodes=config.graph_top_k, alpha=config.alpha)
    return response, chunk_texts, chunks, keywords

async def hybrid_query(query: str, config: GlobalConfig) -> Tuple[str | None, List[str], Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]], Set[str]]:
    return await _hybrid_query(query=query, top_k=config.keywords_top_k, max_nodes=config.graph_top_k, alpha=config.alpha, order_range=config.order_range)

async def naive_query(query: str, config: GlobalConfig) -> Tuple[str, List[str]]:
    return await _naive_rag(query=query, top_k=config.graph_top_k)
