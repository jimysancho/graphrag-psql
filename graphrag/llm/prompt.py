ENTITY_EXTRACTION_PROMPT = """You are an expert extracting entities and relationships from text. 
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
-Steps-
1. Identify all entities. The entities must be explictly present in the text. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities. Language: the same as the text. 
Format each entity as: 
{"entity_name": <entity_name>, "entity_type": <entity_type>, "entity_description": <entity_description>}
2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other based on the text.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other. Language: the same as the text
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
Format each relationship as: 
{"source_entity": <source_entity>, "target_entity": <target_entity>, "relationship_description": <relationship_description>, "relationship_keywords": <relationship_keywords>, "relationship_strength": <relationship_strength>}
Try to create relationships between entities that are far away from each other in the text, not just for entities close to each other. 
3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
Format the content-level key words as
{"content_keywords": <high_level_keywords>}
4. Return output as a single python dictionary of all the entities and relationships and high level keywords identified in steps 1, 2 and 3. 
This is an example of the output: 
<example>
{
  "entities": [
    {"entity_name": <entity_name>, "entity_type": <entity_type>, "entity_description": <entity_description>}, 
    {"entity_name": <entity_name>, "entity_type": <entity_type>, "entity_description": <entity_description>},
    ...
  ], 
  "relationships": [
    {"source_entity": <source_entity>, "target_entity": <target_entity>, "relationship_description": <relationship_description>, "relationship_keywords": <relationship_keywords>, "relationship_strength": <relationship_strength>},
    {"source_entity": <source_entity>, "target_entity": <target_entity>, "relationship_description": <relationship_description>, "relationship_keywords": <relationship_keywords>, "relationship_strength": <relationship_strength>},
    ...
  ],
  "content_keywords": [<keyword>, <keyword>, <keyword>, ...]
}
</example>
This are the available entity types with some features of each one: 
<entity_types>
{entity_types_description}
</entity_types>
This is the text: 
<text>
{text}
</text>
"""

BAD_PYTHON_DICTIONARY_PARSING = """You are an excellent assistant that converts bad python string dictinaries into correct python dictionaries. 
You are going to receive a string that has raised a python exception when doing: 
```python
eval({dict})
```
Your job is to ouput the corrected string to make sure that it not raises a python exception. 
Examples: 
<examples>
input: {"x": ["hello"}
eval(input) raises an exception: SyntaxError: closing parenthesis '}' does not match opening parenthesis '['
output: {"x": ["hello"]}
</examples>

Output just the corrected python dictionary, as your output will be sent directly to an eval python function
Input: {dict}
Error: {error}
Output: 
"""


CONTINUE_EXTRACTING_ENTITIES = "MANY entities were missed in the last extraction. Add them below using the same format:"

KEYWORD_EXTRACTION = """---Role---

You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query.

---Goal---

Given the query, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

---Instructions---

- Output the keywords in JSON format.
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes.
  - "low_level_keywords" for specific entities or details.

######################
-Examples-
######################
Example 1:

Query: "How does international trade influence global economic stability?"
################
Output:
{{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}}
#############################
Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
{{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}}
#############################
Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
{{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}}
#############################
-Real Data-
######################
Query: {query}
######################
Output:
"""

GENERATE_RESPONSE = """You're a helpful assistant
Below are the knowledge you know:
{context}
---
If you don't know the answer or if the provided knowledge do not contain sufficient information to provide an answer, just say so. Do not make anything up.
Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.
"""