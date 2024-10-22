from graphrag.indexing.types import (
    EntityModel, 
    RelationshipModel, 
    ChunkModel, 
)

from graphrag.llm.llm import create_embeddings
from graphrag.database.base import get_db
from graphrag.database.models import Entity, Relationship, Chunk
from graphrag.indexing.utils import calculate_hash

from typing import List

import uuid
import asyncio
import networkx as nx

import uuid
import os


async def upsert_data_and_create_graph(
    entities: List[EntityModel], relationships: List[RelationshipModel], chunks: List[ChunkModel]
) -> nx.Graph:
    
    GRAPH_PATH = "./graph.graphml"
    
    if os.path.exists(GRAPH_PATH):
        graph = nx.read_graphml(GRAPH_PATH)
        return graph

    chunk_embeddings_task = create_embeddings(texts=[chunk.text for chunk in chunks])
    entities_embeddings_task = create_embeddings(texts=[entity.entity_text for entity in entities])
    relationship_embeddings_task = create_embeddings(texts=[
        relationship.relationship_text
        for relationship in relationships
    ])
    
    chunk_embeddings, entities_embeddings, relationship_embeddings = await asyncio.gather(*[
        chunk_embeddings_task, entities_embeddings_task, relationship_embeddings_task
    ])
    
    db = next(get_db())
    not_valid_relationships = set()
    
    for index, chunk in enumerate(chunks):
        
        hash = "chunk-" + calculate_hash(text=chunk.text)
        if db.query(Chunk).filter(Chunk.hash == hash).all(): 
            continue
    
        db.add(
            Chunk(
                chunk_id=uuid.UUID(chunk.id),
                text=chunk.text,
                chunk_embedding=chunk_embeddings[index],
                hash=hash
            )
        )

    db.commit()

    for index, entity in enumerate(entities):
        hash = "ent-" + calculate_hash(text=entity.entity_name)
        if db.query(Entity).filter(Entity.hash == hash).all(): 
            continue

        entity_db = Entity(
            hash=hash,
            entity_name=entity.entity_name,
            entity_type=entity.entity_type,
            description=entity.entity_description,
            chunk_id=uuid.UUID(entity.get_chunk_id),
            entity_embedding=entities_embeddings[index]
        )

        db.add(entity_db)
        db.flush()

    db.commit()
    
    for index, relationship in enumerate(relationships):
        
        hash = "rel-" + calculate_hash(text=relationship.relationship_description)
        if db.query(Relationship).filter(Relationship.hash == hash).all(): 
            continue

        source_id = db.query(Entity).filter(Entity.entity_name == relationship.source_entity).first()
        target_id = db.query(Entity).filter(Entity.entity_name == relationship.target_entity).first()
        
        if source_id is None:
            print(f"Source: {relationship.source_entity} was not found in the database")
            not_valid_relationships.add((relationship.source_entity, relationship.target_entity))
            continue
        if target_id is None:
            print(f"Target: {relationship.target_entity} was not found in the database")
            not_valid_relationships.add((relationship.source_entity, relationship.target_entity))
            continue
        source_id = source_id.entity_id
        target_id = target_id.entity_id
        
        relationship_db = Relationship(
            hash=hash,
            description=relationship.relationship_description,
            relationship_embedding=relationship_embeddings[index],
            source_id=source_id,
            target_id=target_id,
            chunk_id=uuid.UUID(relationship.get_chunk_id),
            keywords=relationship.relationship_keywords,
            weight=relationship.relationship_strength
        )
        
        db.add(relationship_db)
    db.commit()
    db.close()

    print("Database created and updated")
    graph = nx.Graph()
    
    for entity in entities:
        entity.chunk_id = ", ".join(list(entity.chunk_id)) if isinstance(entity.chunk_id, set) else entity.chunk_id
        graph.add_node(
            entity.entity_name, **entity.model_dump()
        )
    
    for relationship in relationships:
        if (relationship.source_entity, relationship.target_entity) in not_valid_relationships: continue
        relationship.relationship_keywords = ", ".join(list(relationship.relationship_keywords))
        relationship.chunk_id = ", ".join(list(relationship.chunk_id)) if isinstance(relationship.chunk_id, set) else relationship.chunk_id
        graph.add_edge(
            relationship.source_entity, relationship.target_entity, **relationship.model_dump()
        )
    
    print(f"Graph created: {len(graph.nodes())} nodes and {len(graph.edges())} edges")
    nx.write_graphml(graph, GRAPH_PATH)
    return graph
