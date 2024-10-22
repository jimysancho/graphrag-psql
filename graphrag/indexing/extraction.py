from graphrag.indexing.types import (
    EntityModel, 
    RelationshipModel, 
    ChunkModel, 
    HighLevelKeywords
)

from graphrag.llm.llm import extract_entities_completion

from typing import List, Tuple, Dict, Any
from fuzzywuzzy import fuzz

import uuid
import asyncio
import uuid


def _merge_entities(entities: List[EntityModel], threshold: int=75) -> Tuple[List[EntityModel], Dict[str, List[str]]]:
    
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
            continue
    
        if (entity.entity_name, entity.entity_type, entity.entity_description, entity.get_chunk_id) not in modified_entities:
            modified_entities[(entity.entity_name, entity.entity_type, entity.entity_description, entity.get_chunk_id)] = []
        if most_sim_entities:
            for most_sim_entity in most_sim_entities:
                most_sim_key = (most_sim_entity.entity_name, most_sim_entity.entity_type, most_sim_entity.entity_description)
                merged_entities.add(most_sim_key)
                modified_entities[(entity.entity_name, entity.entity_type, entity.entity_description, entity.get_chunk_id)].append(most_sim_entity)
                if most_sim_entity.entity_name not in kept_vs_merged:
                    kept_vs_merged[most_sim_entity.entity_name] = [entity.entity_name]
                else:
                    kept_vs_merged[most_sim_entity.entity_name].append(entity.entity_name)

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
                relationship.source_entity = kept_vs_merged_entities[source][0]
            if target in kept_vs_merged_entities:
                relationship.target_entity = kept_vs_merged_entities[target][0]
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
                

async def _extract_graph_information_from_chunk(chunk: str, gleaning: int=1) -> Tuple[List[EntityModel], List[RelationshipModel], ChunkModel] | None:
    
    chunk_info: Dict[str, Any] = {}
    for _ in range(gleaning):
        gleaning_chunk_info = await extract_entities_completion(chunk=chunk, history=None)
        if gleaning_chunk_info is None: continue
    
        more_chunk_info = await extract_entities_completion(
            chunk=chunk, history=str(chunk_info)
        )
        if more_chunk_info is not None:
            chunk_info.update(more_chunk_info)
    
    chunk_model = ChunkModel(text=chunk, id=str(uuid.uuid4()))

    entities, relationships, high_level_keywords = [chunk_info[key] for key in ("entities", "relationships", "content_keywords")]
    entities_models, relationships_models, high_level_keywords_models = [
        [model(**val, chunk_id={chunk_model.id}) for val in values] if isinstance(values, list) else [model(**values, chunk_id={chunk_model.id})]
        for model, values in zip((EntityModel, RelationshipModel, HighLevelKeywords), 
                                 (entities, relationships, high_level_keywords))
    ]
    return entities_models, relationships_models, chunk_model


async def extract_entities(chunks: List[str], gleaning: int=1) -> Any:

    results = await asyncio.gather(*[
        _extract_graph_information_from_chunk(chunk=chunk, gleaning=gleaning) for chunk in chunks
    ])
    
    if results is None:
        return None
    
    entities, relationships, chunks = [], [], []
        
    for result in results:
        if result is None:
            continue
        entities_n, relationships_n, chunk = result
        entities.extend(entities_n)
        relationships.extend(relationships_n)
        chunks.append(chunk)
        
    entities, kept_vs_merged = _merge_entities(entities=entities)
    relationships = _merge_relationships(relationships=relationships, kept_vs_merged_entities=kept_vs_merged)
    
    return entities, relationships, kept_vs_merged, chunks
