from graphrag.llm.llm import generate_response
from graphrag.query.graph_search import local_query_graph

from graphrag.database.base import get_db
from graphrag.database.models import Chunk

from typing import List, Tuple, Any, Dict


async def aquery(
    query: str, top_k: int, max_nodes: int=3, order_range: int=5
) -> Tuple[str | None, List[str], Dict[str, Dict[str, Any]], List[str]]:
    
    db = next(get_db())
    nodes, keywords = await local_query_graph(query=query, top_k=top_k, order_range=order_range)
    chunk_texts: List[str] = []
    for chunk_id in nodes:
        if len(chunk_texts) >= max_nodes:
            break
        chunk_texts.append(
            db.get(Chunk, chunk_id).text
        )
        
    context = "\n".join(chunk_texts)
    response = await generate_response(context=context, query=query)
    db.close()
    return response, chunk_texts, nodes, keywords
