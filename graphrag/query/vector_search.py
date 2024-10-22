from graphrag.llm.llm import create_embeddings
from graphrag.database.base import get_db, SQLALCHEMY_DATABASE_URL
from graphrag.database.models import Entity, Relationship, Chunk

from typing import List, Tuple
from pgvector.psycopg2 import register_vector

import psycopg2


async def _similarity_search(text: str | List[str], table: str, top_k: int) -> List[Tuple[Entity | Relationship | Chunk, float]]:
    
    assert table in ("chunk", "entity", "relationship"), f"{table} is not a valid table"
    string_to_table = {
        "chunk": Chunk, "entity": Entity, "relationship": Relationship
    }
    id = f"{table}_id"
    if isinstance(text, str): text = [text]
    embeddings = await create_embeddings(texts=text)
    
    conn = psycopg2.connect(SQLALCHEMY_DATABASE_URL)
    register_vector(conn)
    nodes_with_distances = []

    for embedding in embeddings:
        cur = conn.cursor()
        cur.execute(f"""
            SELECT {id}, 1 - cosine_distance({table}_embedding, %s::vector) AS similarity 
            FROM {table} 
            ORDER BY similarity DESC 
            LIMIT {top_k};
        """, (embedding,))
        
        nodes_with_distances.extend(cur.fetchall())
        
    nodes_with_distances_sorted = sorted(nodes_with_distances, key=lambda x: -x[1])
    db = next(get_db())
    
    db_elements = []
    database_table = string_to_table[table]

    for id, distance in nodes_with_distances_sorted:
        element = db.get(database_table, id)
        db_elements.append(
            (element, distance)
        )
        
    db.close()
    return db_elements
