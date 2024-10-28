from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import Dict, Optional


class GlobalConfig(BaseModel):

    max_gleaning: int = Field(default=1, description="Number of times entities and relationships will be extracted from the same chunk")
    batch: int = Field(default=15, description="Extract entities from chunk in 'batch' batches")
    entity_types: Optional[Dict[str, str]] = Field(default=None, description="keys: entity types themselves, values: description of entity types")
    min_chunk_size: int = Field(default=120, description="Min chunk size to create the chunks via semantic chunking")
    max_chunk_size: int = Field(default=150, description="Max chunk size to create the chunks via semantic chunking")
    keywords_top_k: int = Field(default=60, description="Number of entities to retrieve via similarity search over keyword vector database")
    graph_top_k: int = Field(default=5, description="Number of chunks to use as final context")
    order_range: int = Field(default=5, description="When getting the most connected components, max number of order difference in similarity search to substitute one communiy over other")    
    alpha: float = Field(default=0.7, description="Importance to similarity vs weight of relationships. Value between 0 and 1")
    
    @field_validator('alpha')
    def validate_alpha(cls, value):
        if not (0 <= value <= 1):
            raise ValidationError(f"{value} is not a valid value for alpha. It must be between 0 and 1")
