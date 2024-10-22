ENTITY_EXTRACTION_PROMPT = """-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as: 
{"entity_name": <entity_name>, "entity_type": <entity_type>, "entity_description": <entity_description>}

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
Format each relationship as: 
{"source_entity": <source_entity>, "target_entity": <target_entity>, "relationship_description": <relationship_description>, "relationship_keywords": <relationship_keywords>, "relationship_strength": <relationship_strength>}

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
Format the content-level key words as
{"content_keywords": <high_level_keywords>}

4. Return output in Spanish as a single python dictionary of all the entities and relationships and high level keywords identified in steps 1, 2 and 3.

######################
-Examples-
######################
Example 1:

Entity_types: [person, technology, mission, organization, location]
Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.”

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
################
Output:
{
  "entities": [
    {
      "entity_name": "Alex",
      "entity_type": "person",
      "entity_description": "Alex is a character who experiences frustration and is observant of the dynamics among other characters."
    },
    {
      "entity_name": "Taylor",
      "entity_type": "person",
      "entity_description": "Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective."
    },
    {
      "entity_name": "Jordan",
      "entity_type": "person",
      "entity_description": "Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device."
    },
    {
      "entity_name": "Cruz",
      "entity_type": "person",
      "entity_description": "Cruz is associated with a vision of control and order, influencing the dynamics among other characters."
    },
    {
      "entity_name": "The Device",
      "entity_type": "technology",
      "entity_description": "The Device is central to the story, with potential game-changing implications, and is revered by Taylor."
    }
  ],
  "relationships": [
    {
      "source_entity": "Alex",
      "target_entity": "Taylor",
      "relationship_description": "Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device.",
      "relationship_keywords": "power dynamics, perspective shift",
      "relationship_strength": 7
    },
    {
      "source_entity": "Alex",
      "target_entity": "Jordan",
      "relationship_description": "Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision.",
      "relationship_keywords": "shared goals, rebellion",
      "relationship_strength": 6
    },
    {
      "source_entity": "Taylor",
      "target_entity": "Jordan",
      "relationship_description": "Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce.",
      "relationship_keywords": "conflict resolution, mutual respect",
      "relationship_strength": 8
    },
    {
      "source_entity": "Jordan",
      "target_entity": "Cruz",
      "relationship_description": "Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order.",
      "relationship_keywords": "ideological conflict, rebellion",
      "relationship_strength": 5
    },
    {
      "source_entity": "Taylor",
      "target_entity": "The Device",
      "relationship_description": "Taylor shows reverence towards the device, indicating its importance and potential impact.",
      "relationship_keywords": "reverence, technological significance",
      "relationship_strength": 9
    }
  ],
  "content_keywords": {
    "content_keywords": "power dynamics, ideological conflict, discovery, rebellion"
  }
}
#############################
Example 2:

Entity_types: [person, technology, mission, organization, location]
Text:
They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve.

Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril.

Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence— the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly
#############
Output:
{
  "entities": [
    {
      "entity_name": "Washington",
      "entity_type": "location",
      "entity_description": "Washington is a location where communications are being received, indicating its importance in the decision-making process."
    },
    {
      "entity_name": "Operation: Dulce",
      "entity_type": "mission",
      "entity_description": "Operation: Dulce is described as a mission that has evolved to interact and prepare, indicating a significant shift in objectives and activities."
    },
    {
      "entity_name": "The team",
      "entity_type": "organization",
      "entity_description": "The team is portrayed as a group of individuals who have transitioned from passive observers to active participants in a mission, showing a dynamic change in their role."
    }
  ],
  "relationships": [
    {
      "source_entity": "The team",
      "target_entity": "Washington",
      "relationship_description": "The team receives communications from Washington, which influences their decision-making process.",
      "relationship_keywords": "decision-making, external influence",
      "relationship_strength": 7
    },
    {
      "source_entity": "The team",
      "target_entity": "Operation: Dulce",
      "relationship_description": "The team is directly involved in Operation: Dulce, executing its evolved objectives and activities.",
      "relationship_keywords": "mission evolution, active participation",
      "relationship_strength": 9
    }
  ],
  "content_keywords": {
    "content_keywords": "mission evolution, decision-making, active participation, cosmic significance"
  }
}
#############################
Example 3:

Entity_types: [person, role, technology, organization, event, location, concept]
Text:
their voice slicing through the buzz of activity. "Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data.

"It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning."

Alex surveyed his team—each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back."

Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history.

The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation
#############
Output:
{
  "entities": [
    {
      "entity_name": "Sam Rivera",
      "entity_type": "person",
      "entity_description": "Sam Rivera is a member of a team working on communicating with an unknown intelligence, showing a mix of awe and anxiety."
    },
    {
      "entity_name": "Alex",
      "entity_type": "person",
      "entity_description": "Alex is the leader of a team attempting first contact with an unknown intelligence, acknowledging the significance of their task."
    },
    {
      "entity_name": "Control",
      "entity_type": "concept",
      "entity_description": "Control refers to the ability to manage or govern, which is challenged by an intelligence that writes its own rules."
    },
    {
      "entity_name": "Intelligence",
      "entity_type": "concept",
      "entity_description": "Intelligence here refers to an unknown entity capable of writing its own rules and learning to communicate."
    },
    {
      "entity_name": "First Contact",
      "entity_type": "event",
      "entity_description": "First Contact is the potential initial communication between humanity and an unknown intelligence."
    },
    {
      "entity_name": "Humanity's Response",
      "entity_type": "event",
      "entity_description": "Humanity's Response is the collective action taken by Alex's team in response to a message from an unknown intelligence."
    }
  ],
  "relationships": [
    {
      "source_entity": "Sam Rivera",
      "target_entity": "Intelligence",
      "relationship_description": "Sam Rivera is directly involved in the process of learning to communicate with the unknown intelligence.",
      "relationship_keywords": "communication, learning process",
      "relationship_strength": 9
    },
    {
      "source_entity": "Alex",
      "target_entity": "First Contact",
      "relationship_description": "Alex leads the team that might be making the First Contact with the unknown intelligence.",
      "relationship_keywords": "leadership, exploration",
      "relationship_strength": 10
    },
    {
      "source_entity": "Alex",
      "target_entity": "Humanity's Response",
      "relationship_description": "Alex and his team are the key figures in Humanity's Response to the unknown intelligence.",
      "relationship_keywords": "collective action, cosmic significance",
      "relationship_strength": 8
    },
    {
      "source_entity": "Control",
      "target_entity": "Intelligence",
      "relationship_description": "The concept of Control is challenged by the Intelligence that writes its own rules.",
      "relationship_keywords": "power dynamics, autonomy",
      "relationship_strength": 7
    }
  ],
  "content_keywords": {
    "content_keywords": "first contact, control, communication, cosmic significance"
  }
}
#############################
-Real Data-
######################
Entity_types: {entity_types}
Text: {text}
######################
Output:
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