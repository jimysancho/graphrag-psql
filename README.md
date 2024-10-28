# LightRAG connected to Postgresql

This repo is completly based on LightRAG. It serves as the bridge between the LightRAG logic (with some differences and improvements), and **postgresql** as a **vector and text storage**. The graph tool is still **networkx**. 


## 1. Database

We only need to define 3 models: 
- Chunk: Stores the text information. 
- Entity: Stores the entities extracted from the text. 
- Relationships: Stores the relationships between the different entities extracted from the text. 

### 1.1 Dockerfile

In lightrag, the entities and relationships are transformed into vectors to perform similarity search over to retrieve the most relevant chunks to the given query. To do so in postgres, we need `pgvector`. The Dockerfile image: 

```Dockerfile
FROM postgres:16

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    postgresql-server-dev-16 \
    wget \
    unzip \
    git \
    clang \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN cd /tmp \
    && git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git \
    && cd pgvector \
    && make \
    && make install

COPY ./init.sql /docker-entrypoint-initdb.d/
```

Where the *init.sql*: 
```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
```

### 1.2 Schema

The database schema is defined as: 

```python
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
```

### 1.3 Vector columns

To deal with entities and relationships more confortable throughout the process, some pydantic models are created: 

```python
class EntityModel(BaseModel):
    entity_name: str
    entity_type: str
    entity_description: str
    chunk_id: Set[str] | str
    entity_text: str | None = None


class RelationshipModel(BaseModel):
    source_entity: str
    target_entity: str
    relationship_description: str
    relationship_keywords: str | List[str]
    relationship_strength: float
    chunk_id: Set[str] | str
    
    relationship_text: str | None = None
```

To compute the vectors that will later be used to perform similarity search given the query, we'll use the following attributes: 

- For the entities

```python
@model_validator(mode='after')
def remove_tildes(self):
    self.entity_name = unidecode(self.entity_name).lower().strip()
    self.entity_type = unidecode(self.entity_type).lower().strip()
    self.entity_text = f"{self.entity_name} it's of type: {self.entity_type}: {self.entity_description}"
    return self
```

*entity_text* attribute is the one that will be transformed to a vector. 

- For the relationships

```python
@model_validator(mode='after')
def remove_tildes(self):
    self.source_entity = unidecode(self.source_entity).lower().strip()
    self.target_entity = unidecode(self.target_entity).lower().strip()
    self.relationship_keywords = self.relationship_keywords.split(", ") if isinstance(self.relationship_keywords, str) else self.relationship_keywords
    self.relationship_text = f"{self.source_entity} is related to {self.target_entity} because of: {self.relationship_description}"
    return self
```

*relationship_text* attribute is the one that will be transformed to a vector. 


## 2. Extraction and Upsert

The extraction of entities and relationships is a two-step process (just as lightrag): 

1. Entities and relationships themselves (we'll use the default prompt of graphrag and lighrag, but with the examples in json)
2. Merging of entities and relationships. 

The step 1) is the same as with lightrag. The step 2) is a little bit different, since we'll use `fuzzywuzzy` package to find similar entities and relationships and merge those.
First we'll merge the entities, and then, keeping track of those merged entities, we'll merge the relationships. 

### 2.1 Entities

```python
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
```

We iterate over the entities to find similar entities among the rest of them, and keep track of those that are found in a python dictionary: *kept_vs_merged* 

```python
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
```

and then, we update the original entities like in lightrag:
- Joining the descriptions
- Keeping track of the chunks they appear in

```python
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
```

### 2.2 Relationships

By using the `kept_vs_merged` entities found on the previous step, we update the relationships by replace the old entities with the new entities. Like in lightrag: 
- *relationship_strength* is updated
- *relationship_keywords* is updated
- *relationship_description* is joined between the common relationships

```python
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
```

### 2.3 Upsert and Graph Creation

Using the merged entities and the merged relationships, we upsert this data into the database and we create the graph as well. 


## 3. Generation of response

There are several steps involed before generating a response: 
1. Keyword extraction from query.
2. Similarity Search
3. Graph transversal for both local and global query
4. Generate a response

### 3.1 Keyword extraction
Given a query, we'll extract from it using llms: 

- **High level keywords** which will be used in the `global query` approach. 
- **Low level keywords** which will be used in the `local query` approach. 

### 3.2 Query


#### 3.2.1 Similarity Search

Both `local query` and `global query` use `similarity search` as the first step to find the relevant parts of the graph to begin the retrieval process. 

```python
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
```

#### 3.2.2 Local Query

This type of query is mainly focused on retrieving specific entities along with their associated attributes or relationships. Queries at this level are detail-oriented and aim to extract precise information about particular nodes or edges within the graph. 

**Process**: 
1. Using the **low level keywords** and similarity search over the **entities**, we extract the most relevant entities to these keywords. 
2. Get nodes found on step 1 from the graph and the first neighbors of these nodes. 
3. Get the most connected entities to use the chunks they belong to as context to feed the llm. 

```python
connected_nodes: Dict[str, Dict[str, Any]] = {}

for index, (keyword_node, keyword_edges) in enumerate(zip(nodes, edges)):
    node: str = keyword_node['entity_name']
    del keyword_node['entity_name']
    if node in connected_nodes: continue
    connected_nodes[node] = {}
    chunk_ids = keyword_node['chunk_id'].split(", ")
    for chunk_id in chunk_ids:
        if chunk_id not in connected_nodes[node]: 
            connected_nodes[node][chunk_id] = {"relation_counts": 0, "keywords": set(), "order": index, "graph_node": keyword_node}
        for edge in keyword_edges:
            neighbor = edge[1]
            neighbor_data = neighbor_chunk_ids_mapping[neighbor]
            if chunk_id in neighbor_data:
                connected_nodes[node][chunk_id]['relation_counts'] += 1
                connected_nodes[node][chunk_id]['keywords'].add(neighbor)
```


#### 3.2.3 Global Query

This level addresses broader topics and overarching themes. Queries at this level aggregate information acorss multiple related entities and relationships, providing insights into higher-level concepts and summaries rather than specific details. 

**Process**:
1. Using the **high level keywords** and similarity search over the **relationships**, we extract the most relevant relationships to these keywords.
2. Compute an heuristic metric to define the **importance** of an edge (i.e relationship) given the query. 

```python
for index, chunk_edge_ids in enumerate(chunk_ids):
    for chunk_id in chunk_edge_ids:
        edges_of_chunk = chunk_to_edges[chunk_id]
        edges_of_chunk_graph = [graph.edges.get(k) for k in edges_of_chunk]
        if chunk_id not in chunk_ids_to_metric:
            chunk_ids_to_metric[chunk_id] = {
                'order': index, 
                'weight': sum([edge['relationship_strength'] for edge in edges_of_chunk_graph]), 
                'n': len(edges_of_chunk_graph), 
                'edges': edges_of_chunk_graph
            }
        else:
            chunk_ids_to_metric[chunk_id]['order'] += index
            chunk_ids_to_metric[chunk_id]['weight'] += sum([edge['relationship_strength'] for edge in edges_of_chunk_graph])
            chunk_ids_to_metric[chunk_id]['n'] += len(edges_of_chunk_graph)
            chunk_ids_to_metric[chunk_id]['edges'].extend(edges_of_chunk_graph)

for chunk_id, metrics in chunk_ids_to_metric.items():
    metrics['order'] /= metrics['n']
    metrics['weight'] /= metrics['n']
    
_, max_weight = max(chunk_ids_to_metric.items(), key=lambda x: x[1]['weight'])
max_weight = max_weight['weight']
for chunk_id, metrics in chunk_ids_to_metric.items():
    metrics['importance'] = (1 - metrics['order'] / index) * alpha + (metrics['weight'] / max_weight) * (1 - alpha)
```

Where **alpha** determines which metric has more relevance to the heuristic: 
- weight
- the order of the similarity search


#### 3.2.4 Hybrid Query

We combine local query and global query to generate a response. 

## 4 Usage

### 4.1 Main functions

These are the availables types of queries of these project:

```python
async def insert(text: str, config: GlobalConfig) -> nx.Graph:
    chunks = await create_chunks(text=text, 
                                 min_token_size=config.min_chunk_size, 
                                 max_token_size=config.max_chunk_size)
    print(f"{len(chunks)} chunks created")
    entities, relationships, kept_vs_merged, chunk_models = await extract_entities(chunks=chunks, entity_types=config.entity_types, gleaning=config.max_gleaning, batch=config.batch)
    print(f"{len(entities)} entities extracted and {len(relationships)} relationships extracted. ")
    graph = await upsert_data_and_create_graph(entities=entities, relationships=relationships, chunks=chunk_models)
    return graph

async def local_query(query: str, config: GlobalConfig) -> Tuple[str | None, List[str], Dict[str, Dict[str, Any]], List[str]]:
    response, chunk_texts, nodes, keywords = await _local_query(query=query, top_k=config.keywords_top_k, max_nodes=config.graph_top_k, order_range=config.order_range)
    return response, chunk_texts, nodes, keywords

async def global_query(query: str, config: GlobalConfig) -> Tuple[str | None, List[str], Dict[str, Any], List[str]]:
    response, chunk_texts, chunks, keywords = await _global_query(query=query, top_k=config.keywords_top_k, max_nodes=config.graph_top_k, alpha=config.alpha)
    return response, chunk_texts, chunks, keywords

async def hybrid_query(query: str, config: GlobalConfig) -> Tuple[str | None, List[str], Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]], Set[str]]:
    return await _hybrid_query(query=query, top_k=config.keywords_top_k, max_nodes=config.graph_top_k, alpha=config.alpha, order_range=config.order_range)

async def naive_query(query: str, config: GlobalConfig) -> Tuple[str, List[str]]:
    return await _naive_rag(query=query, top_k=config.graph_top_k)

```

where `GlobalConfig`:

```python
class GlobalConfig(BaseModel):

    max_gleaning: int = Field(default=1, description="Number of times entities and relationships will be extracted from the same chunk")
    batch: int = Field(default=15, description="Extract entities from chunk in 'batch' batches")
    entity_types: Dict[str, str] = Field(..., description="keys: entity types themselves, values: description of entity types")
    min_chunk_size: int = Field(default=120, description="Min chunk size to create the chunks via semantic chunking")
    max_chunk_size: int = Field(default=150, description="Max chunk size to create the chunks via semantic chunking")
    keywords_top_k: int = Field(default=60, description="Number of entities to retrieve via similarity search over keyword vector database")
    graph_top_k: int = Field(default=5, description="Number of chunks to use as final context")
    order_range: int = Field(default=5, description="When getting the most connected components, max number of order difference in similarity search to substitute one communiy over other")    
    alpha: float = Field(default=0.7, description="Importance to similarity vs weight of relationships. Value between 0 and 1")
    
```

### Example

```python
from graphrag import (
    insert, 
    GlobalConfig, 
    local_query, 
    global_query, 
    hybrid_query, 
    naive_query
)

entity_types = {
    "location": "", 
    "price tier": "",
    "claim": "",
    "condition": "",
    "excess": "", 
    "gadget": "",
    "coverage quantity": "",
    
}

config = GlobalConfig(
    entity_types=entity_types, 
    min_chunk_size=400, 
    max_chunk_size=800
)

plain_text = "YOUR TEXT to create the graph from" 
graph = await insert(text=plain_text, config=config)
query = "YOUR QUERY"

# local query
response, chunk_texts, nodes, keywords = await local_query(query=query, config=config)

# global query
response, chunk_texts, edges, keywords = await global_query(query=query, config=config)

# hybrid query
response, chunk_texts, (global_chunks, local_nodes), keywords = await hybrid_query(query=query, config=config)

# naive query
response, chunk_texts = await naive_query(query=query, config=config)
```