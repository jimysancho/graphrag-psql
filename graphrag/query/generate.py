from graphrag.llm.llm import generate_response
from graphrag.query.graph_search import query_graph

from graphrag.database.base import get_db
from graphrag.database.models import Chunk

from typing import List, Tuple, Any


async def aquery(query: str, max_nodes: int=3) -> Tuple[str | None, List[str], List[Any], List[str]]:
    
    db = next(get_db())
    nodes, keywords = await query_graph(query=query)
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
