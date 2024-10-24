from graphrag.indexing.types import (
    EntityModel, 
    RelationshipModel, 
    ChunkModel, 
    HighLevelKeywords
)

from graphrag.database.base import get_db
from graphrag.database.models import Chunk, Entity, Relationship

from graphrag.indexing.utils import calculate_hash
from graphrag.llm.llm import extract_entities_completion

from typing import (List, 
                    Tuple, 
                    Dict, 
                    Any, 
                    Set)
from fuzzywuzzy import fuzz
from openai import RateLimitError

import uuid
import asyncio
import uuid


def _extract_chunk_info_from_db_if_exists(chunk: str) -> Tuple[List[EntityModel], List[RelationshipModel], ChunkModel] | None:

    db = next(get_db())
    hash = "chunk-" + calculate_hash(text=chunk)
    exists = db.query(Chunk).filter(Chunk.hash == hash).first()
    if exists:
        entities_models, relationships_models = [], [] 
        entities = db.query(Entity).filter(Entity.chunk_id == exists.chunk_id).all()
        for entity in entities:
            entity_model = EntityModel(
                entity_name=entity.entity_name, 
                entity_type=entity.entity_type, 
                entity_description=entity.description,
                chunk_id=str(exists.chunk_id)
            )
            entities_models.append(entity_model)
        relationships = db.query(Relationship).filter(Relationship.chunk_id == exists.chunk_id).all()
        for rel in relationships:
            rel_model = RelationshipModel(
                source_entity=rel.source_entity.entity_name, 
                target_entity=rel.target_entity.entity_name,
                relationship_description=rel.description,
                relationship_strength=rel.weight,
                chunk_id=str(exists.chunk_id),
                relationship_keywords=rel.keywords
            )
            relationships_models.append(rel_model)
        db.close()
        return entities_models, relationships_models, ChunkModel(text=exists.text, id=str(exists.chunk_id))
    db.close()
    return None
    

def _merge_entities(entities: List[EntityModel], threshold: int=75) -> Tuple[List[EntityModel], Dict[str, Set[str]]]:
    
    def find_most_similar(entity: EntityModel, candidates: List[EntityModel], threshold: int) -> List[EntityModel]:
        
        most_sim_entity: List[EntityModel] = []
        for candidate_entity in candidates:
            if (entity.entity_type != candidate_entity.entity_type): continue
            try:
                score = fuzz.ratio(entity.entity_name, candidate_entity.entity_name)
            except IndexError:
                continue
            if score > threshold:
                most_sim_entity.append(candidate_entity)

        return most_sim_entity
    
    kept_vs_merged = {}

    merged_entities = set()
    modified_entities: Dict[Tuple[str, str, str, str], List[EntityModel]] = {}
    for index, entity in enumerate(entities):
        most_sim_entities = find_most_similar(entity=entity, candidates=entities[index+1:], threshold=threshold)
        entity_key = (entity.entity_name, entity.entity_type, entity.entity_description)

        if entity_key in merged_entities:
            print(f"{entity_key} already exists as an entity")
            continue
    
        if (entity.entity_name, entity.entity_type, entity.entity_description, entity.get_chunk_id) not in modified_entities:
            modified_entities[(entity.entity_name, entity.entity_type, entity.entity_description, entity.get_chunk_id)] = []
        if most_sim_entities:
            for most_sim_entity in most_sim_entities:
                most_sim_key = (most_sim_entity.entity_name, most_sim_entity.entity_type, most_sim_entity.entity_description)
                print(f"{most_sim_key[:2]} has been identified as similar to another entity")
                merged_entities.add(most_sim_key)
                modified_entities[(entity.entity_name, entity.entity_type, entity.entity_description, entity.get_chunk_id)].append(most_sim_entity)
                if most_sim_entity.entity_name not in kept_vs_merged:
                    kept_vs_merged[most_sim_entity.entity_name] = {entity.entity_name}
                else:
                    kept_vs_merged[most_sim_entity.entity_name].add(entity.entity_name)

    updated_entities = []
    for entity_info, sim_entities in modified_entities.items():

        if entity_info[:-1] in merged_entities and not len(sim_entities): continue
        
        updated_entities.append(
            EntityModel(
                entity_name=entity_info[0], 
                entity_type=entity_info[1], 
                entity_description=entity_info[2] + "\n".join([sim_entity.entity_description for sim_entity in sim_entities]), 
                chunk_id=set([entity_info[3]] + [sim_entity.get_chunk_id for sim_entity in sim_entities])
            )
        )
        
    return updated_entities, kept_vs_merged


def _merge_relationships(
    relationships: List[RelationshipModel], kept_vs_merged_entities: Dict[str, List[str]]
) -> List[RelationshipModel]:

    for relationship in relationships:
        source, target = relationship.source_entity, relationship.target_entity
        try:
            if source in kept_vs_merged_entities:
                relationship.source_entity = list(kept_vs_merged_entities[source])[0]
            if target in kept_vs_merged_entities:
                relationship.target_entity = list(kept_vs_merged_entities[target])[0]
        except (KeyError, IndexError):
            print(f"Something went wrong for edge: {(source, target)}")
            continue
        

    merged_relationships = {}
    for relationship in relationships:
        edge = (relationship.source_entity, relationship.target_entity)
        if edge not in merged_relationships:
            merged_relationships[edge] = relationship
            continue
        print(f"Edge: ({source}, {target}) already exists")
        existing_edge = merged_relationships[edge]
        existing_edge.relationship_description += "\n" + relationship.relationship_description
        existing_edge.relationship_strength += relationship.relationship_strength
        existing_edge.relationship_keywords += relationship.relationship_keywords
        existing_edge.relationship_keywords = list(set(existing_edge.relationship_keywords))
        existing_edge.update_chunk_ids(relationship.chunk_id)
        
    return list(merged_relationships.values())
                

async def _extract_graph_information_from_chunk(chunk: str, entity_types: Dict[str, str], gleaning: int=1) -> Tuple[List[EntityModel], List[RelationshipModel], ChunkModel] | None:
    
    already_exists = _extract_chunk_info_from_db_if_exists(chunk=chunk)
    if already_exists is not None:
        return already_exists
    chunk_info: Dict[str, Any] = {}
    for _ in range(gleaning):
        gleaning_chunk_info = await extract_entities_completion(chunk=chunk, 
                                                                history=None, 
                                                                entity_types=entity_types)
        if gleaning_chunk_info is None: continue
    
        more_chunk_info = await extract_entities_completion(
            chunk=chunk, history=str(chunk_info), entity_types=entity_types
        )
        if more_chunk_info is not None:
            chunk_info.update(more_chunk_info)
    
    chunk_model = ChunkModel(text=chunk, id=str(uuid.uuid4()))
    try:
        entities, relationships, high_level_keywords = [chunk_info[key] for key in ("entities", "relationships", "content_keywords")]
    except KeyError as e:
        print(f"KeyError -> {e}")
        print(chunk_info)
        raise e
    if isinstance(high_level_keywords, list): high_level_keywords = {"content_keywords": high_level_keywords}
    entities_models, relationships_models, high_level_keywords_models = [
        [model(**val, chunk_id={chunk_model.id}) for val in values] if isinstance(values, list) else [model(**values, chunk_id={chunk_model.id})]
        for model, values in zip((EntityModel, RelationshipModel, HighLevelKeywords), 
                                 (entities, relationships, high_level_keywords))
    ]
    return entities_models, relationships_models, chunk_model


async def extract_entities(chunks: List[str], entity_types: Dict[str, str], gleaning: int=1, batch: int=15) -> Tuple[List[EntityModel], List[RelationshipModel], Dict[str, Set[str]], List[ChunkModel]]:

    if len(chunks) > batch:
        results = []
        for k in range(0, len(chunks), batch):
            batch_chunks = chunks[k: k + batch]
            try:
                results.extend(
                    await asyncio.gather(*[
                        _extract_graph_information_from_chunk(chunk=chunk, gleaning=gleaning, entity_types=entity_types) for chunk in batch_chunks
                    ])
                )
            except RateLimitError:
                print("Rate limit error. Sleeping for a few seconds...")
                await asyncio.sleep(2)
                sub_batch = batch // 2
                for j in range(0, len(batch_chunks), sub_batch):
                    results.extend(
                        await asyncio.gather(*[
                            _extract_graph_information_from_chunk(chunk=chunk, gleaning=gleaning, entity_types=entity_types) for chunk in batch_chunks[j: j + sub_batch]
                        ])
                    )
                    await asyncio.sleep(1)
            await asyncio.sleep(1)
    else:
        results = await asyncio.gather(*[
            _extract_graph_information_from_chunk(chunk=chunk, gleaning=gleaning, entity_types=entity_types) for chunk in chunks
        ])
    
    if results is None:
        return None
    
    entities, relationships, chunks_models = [], [], []
        
    for result in results:
        if result is None:
            continue
        entities_n, relationships_n, chunk = result
        entities.extend(entities_n)
        relationships.extend(relationships_n)
        chunks_models.append(chunk)
        
    entities, kept_vs_merged = _merge_entities(entities=entities)
    relationships = _merge_relationships(relationships=relationships, kept_vs_merged_entities=kept_vs_merged)
    
    return entities, relationships, kept_vs_merged, chunks_models
