from pydantic import BaseModel, model_validator
from typing import Set, Any

from unidecode import unidecode


class EntityModel(BaseModel):
    entity_name: str
    entity_type: str
    entity_description: str
    chunk_id: Set[str]

    @property
    def get_chunk_id(self) -> Any:
        if len(self.chunk_id) <= 1:
            return list(self.chunk_id)[0]
        return self.chunk_id
    
    @model_validator(mode='after')
    def remove_tildes(self):
        self.entity_name = unidecode(self.entity_name).lower()
        self.entity_type = unidecode(self.entity_type).lower()
        return self
    
    def update_chunk_ids(self, chunk_id: str | Set[str]):
        self.chunk_id.add(chunk_id) if isinstance(chunk_id, str) else self.chunk_id.union(chunk_id)
    
    
class RelationshipModel(BaseModel):
    source_entity: str
    target_entity: str
    relationship_description: str
    relationship_keywords: str
    relationship_strength: float
    chunk_id: Set[str]
    
    @model_validator(mode='after')
    def string_to_list(self):
        self.relationship_keywords = self.relationship_keywords.split(", ")
        return self
    
    @model_validator(mode='after')
    def remove_tildes(self):
        self.source_entity = unidecode(self.source_entity).lower()
        self.target_entity = unidecode(self.target_entity).lower()
        return self
    
    def update_chunk_ids(self, chunk_id: str | Set[str]):
        self.chunk_id.add(chunk_id) if isinstance(chunk_id, str) else self.chunk_id.union(chunk_id)
    

class HighLevelKeywords(BaseModel):
    content_keywords: str
    chunk_id: Set[str]
    
    @model_validator(mode='after')
    def string_to_list(self):
        self.content_keywords = self.content_keywords.split(", ")
        return self


class ChunkModel(BaseModel):
    text: str
    id: str
