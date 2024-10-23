from pydantic import BaseModel, model_validator
from typing import Set, Any, List

from unidecode import unidecode


class EntityModel(BaseModel):
    entity_name: str
    entity_type: str
    entity_description: str
    chunk_id: Set[str] | str
    entity_text: str | None = None

    @property
    def get_chunk_id(self) -> Any:
        if isinstance(self.chunk_id, str): return self.chunk_id
        return list(self.chunk_id)[0]
    
    @model_validator(mode='after')
    def remove_tildes(self):
        self.entity_name = unidecode(self.entity_name).lower().strip()
        self.entity_type = unidecode(self.entity_type).lower().strip()
        self.entity_text = f"{self.entity_name} it's of type: {self.entity_type}: {self.entity_description}"
        return self
    
    def update_chunk_ids(self, chunk_id: str | Set[str]):
        self.chunk_id.add(chunk_id) if isinstance(chunk_id, str) else self.chunk_id.union(chunk_id)
    
    
class RelationshipModel(BaseModel):
    source_entity: str
    target_entity: str
    relationship_description: str
    relationship_keywords: str | List[str]
    relationship_strength: float
    chunk_id: Set[str] | str
    
    relationship_text: str | None = None
        
    @model_validator(mode='after')
    def remove_tildes(self):
        self.source_entity = unidecode(self.source_entity).lower().strip()
        self.target_entity = unidecode(self.target_entity).lower().strip()
        self.relationship_keywords = self.relationship_keywords.split(", ") if isinstance(self.relationship_keywords, str) else self.relationship_keywords
        self.relationship_text = f"{self.source_entity} is related to {self.target_entity} because of: {self.relationship_description}"
        return self

    def update_chunk_ids(self, chunk_id: str | Set[str]):
        self.chunk_id.add(chunk_id) if isinstance(chunk_id, str) else self.chunk_id.union(chunk_id)
        
    @property
    def get_chunk_id(self) -> str:
        if isinstance(self.chunk_id, str): return self.chunk_id
        return list(self.chunk_id)[0]
    

class HighLevelKeywords(BaseModel):
    content_keywords: str | List[str]
    chunk_id: Set[str] | str
    
    @model_validator(mode='after')
    def string_to_list(self):
        self.content_keywords = self.content_keywords.split(", ") if isinstance(self.content_keywords, str) else self.content_keywords
        return self


class ChunkModel(BaseModel):
    text: str
    id: str
