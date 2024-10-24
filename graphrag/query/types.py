from pydantic import BaseModel, model_validator
from typing import List, Any
from unidecode import unidecode


class Keywords(BaseModel):
    high_level_keywords: List[str]
    low_level_keywords: List[str]
    
    @model_validator(mode='after')
    def remove_tildes(self):
        self.high_level_keywords = [unidecode(keyword).lower() for keyword in self.high_level_keywords]
        self.low_level_keywords = [unidecode(keyword).lower() for keyword in self.low_level_keywords]
        return self


class Node(BaseModel):
    element: Any
    degree: int
    
class Edge(BaseModel):
    edge: Any
    degree: int