from graphrag.indexing import (create_chunks, 
                               upsert_data_and_create_graph, 
                               extract_entities)
from graphrag.query.generate import aquery

from graphrag.config import GlobalConfig
from typing import List, Dict, Tuple, Any

import networkx as nx


async def insert(text: str, config: GlobalConfig) -> nx.Graph:
    chunks = await create_chunks(text=text, 
                                 min_token_size=config.min_chunk_size, 
                                 max_token_size=config.max_chunk_size)
    entities, relationships, kept_vs_merged, chunk_models = await extract_entities(chunks=chunks, entity_types=config.entity_types, gleaning=config.max_gleaning, batch=config.batch)
    graph = await upsert_data_and_create_graph(entities=entities, relationships=relationships, chunks=chunk_models)
    return graph

async def query(query: str, config: GlobalConfig) -> Tuple[str | None, List[str], Dict[str, Dict[str, Any]], List[str]]:
    response, chunk_texts, nodes, keywords = await aquery(query=query, top_k=config.keywords_top_k, max_nodes=config.graph_top_k, order_range=config.order_range)
    return response, chunk_texts, nodes, keywords
