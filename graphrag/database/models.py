from graphrag.database.base import Base

from sqlalchemy import (
    Column, 
    String,
    Text,
    ForeignKey,
    Float
)

from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from pgvector.sqlalchemy import Vector

import uuid
    

class Chunk(Base):

    __tablename__ = "chunk"
    chunk_id = Column(UUID(as_uuid=True), primary_key=True, index=True, default=uuid.uuid4)    
    text = Column(Text, nullable=False)
    chunk_embedding = Column(Vector(1536))
    hash = Column(String, nullable=False, index=True)
    
    chunk_entities = relationship("Entity", back_populates="entity_chunk", cascade='all, delete-orphan')
    chunk_relationships = relationship("Relationship", back_populates="relationship_chunk", cascade='all, delete-orphan')

    
class Entity(Base):
    
    __tablename__ = "entity"
    entity_id = Column(UUID(as_uuid=True), primary_key=True, index=True, default=uuid.uuid4)
    
    hash = Column(String, nullable=False, index=True)
    entity_name = Column(String, nullable=False)
    entity_type = Column(String, nullable=True, default="unknown")
    description = Column(String, nullable=False)
    entity_embedding = Column(Vector(1536))
    
    chunk_id = Column(UUID(as_uuid=True), ForeignKey("chunk.chunk_id", ondelete='CASCADE'), nullable=False)
    
    entity_chunk = relationship("Chunk", foreign_keys=[chunk_id], back_populates="chunk_entities")
    sources = relationship("Relationship", back_populates="source_entity", foreign_keys="[Relationship.source_id]", uselist=True)
    targets = relationship("Relationship", back_populates="target_entity", foreign_keys="[Relationship.target_id]", uselist=True)


class Relationship(Base):
    
    __tablename__ = "relationship"
    relationship_id = Column(UUID(as_uuid=True), primary_key=True, index=True, default=uuid.uuid4)
    hash = Column(String, nullable=False, index=True)
    description = Column(String, nullable=False)
    relationship_embedding = Column(Vector(1536))
    keywords = Column(String, nullable=True)
    weight = Column(Float, nullable=True)
    
    source_id = Column(UUID(as_uuid=True), ForeignKey("entity.entity_id", ondelete="CASCADE"), nullable=False)
    target_id = Column(UUID(as_uuid=True), ForeignKey("entity.entity_id", ondelete="CASCADE"), nullable=False)
    chunk_id = Column(UUID(as_uuid=True), ForeignKey("chunk.chunk_id", ondelete='CASCADE'), nullable=False)
    
    relationship_chunk = relationship("Chunk", foreign_keys=[chunk_id], back_populates="chunk_relationships")
    source_entity = relationship("Entity", foreign_keys=[source_id], remote_side="[Entity.entity_id]", back_populates="sources")
    target_entity = relationship("Entity", foreign_keys=[target_id], remote_side="[Entity.entity_id]", back_populates="targets")
