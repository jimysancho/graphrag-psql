from openai import AsyncClient
from graphrag.llm.prompt import (BAD_PYTHON_DICTIONARY_PARSING, 
                                 ENTITY_EXTRACTION_PROMPT,
                                 CONTINUE_EXTRACTING_ENTITIES, 
                                 KEYWORD_EXTRACTION, 
                                 GENERATE_RESPONSE)

from typing import Dict, Any, List
import os


RETRIES = 3

async def extract_entities_completion(chunk: str, entity_types: Dict[str, str], history: str | None=None) -> Dict[str, Any] | None:

    client = AsyncClient(api_key=os.environ['OPENAI_API_KEY'])
    entity_types_string = ", ".join(list(entity_types.keys()))
    entity_types_description_string = "\n".join([f"{entity_type}: {description}" for entity_type, description in entity_types.items()])
    for _ in range(RETRIES):
        if history is None:
            response = await client.chat.completions.create(
                messages=[
                    {"role": "system", "content": ENTITY_EXTRACTION_PROMPT\
                        .replace("{entity_types}", entity_types_string)\
                            .replace("{text}", chunk)\
                                .replace("{entity_types_description}", entity_types_description_string)}
                    ], 
                model='gpt-4o-mini'
            )
        else:
            history = ENTITY_EXTRACTION_PROMPT.replace("{entity_types}", entity_types_string).replace("{text}", chunk) + history
            response = await client.chat.completions.create(
                messages=[{"role": "system", "content": history + CONTINUE_EXTRACTING_ENTITIES}], 
                model='gpt-4o-mini'
            )
        
        answer = response.choices[0].message.content
        try:
            return eval(answer) if answer is not None else None
        except SyntaxError as e:
            corrected_response = await client.chat.completions.create(
                messages=[
                    {"role": "system", "content": BAD_PYTHON_DICTIONARY_PARSING.replace("{dict}", str(answer)).replace("{error}", str(e))}
                ], 
                model='gpt-4o-mini'
            )
            corrected_answer = corrected_response.choices[0].message.content
            try:
                return eval(corrected_answer) if corrected_answer is not None else None
            except SyntaxError as e:
                continue
    return {}


async def create_embeddings(texts: List[str]) -> List[List[float]]:
    client = AsyncClient(api_key=os.environ["OPENAI_API_KEY"])
    results = await client.embeddings.create(input=texts, model="text-embedding-3-small")
    return [result.embedding for result in results.data]


async def extract_keywords_from_query(query: str, return_all: bool=True) -> Dict[str, List[str]] | List[str]:

    client = AsyncClient(api_key=os.environ['OPENAI_API_KEY'])
    
    for _ in range(RETRIES):
        response = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": KEYWORD_EXTRACTION.replace("{query}", query)}
                ], 
            model='gpt-4o-mini', 
            temperature=0
        )
        
        answer = response.choices[0].message.content
        try:
            if answer is not None:
                keywords = eval(answer)
                if return_all:
                    all_keywords = []
                    for level_keywords in keywords.values(): all_keywords.extend(level_keywords)
                    return all_keywords

        except SyntaxError as e:
            corrected_response = await client.chat.completions.create(
                messages=[
                    {"role": "system", "content": BAD_PYTHON_DICTIONARY_PARSING.replace("{dict}", str(answer)).replace("{error}", str(e))}
                ], 
                model='gpt-4o-mini', 
                temperature=0
            )
            corrected_answer = corrected_response.choices[0].message.content
            try:
                if corrected_answer is None:
                    continue
                keywords = eval(corrected_answer)
                if return_all:
                    all_keywords = []
                    for level_keywords in keywords.values(): all_keywords.extend(level_keywords)
                    return all_keywords
            except SyntaxError as e:
                continue
    return {}

async def generate_response(query: str, context: str) -> str | None:
    client = AsyncClient(api_key=os.environ['OPENAI_API_KEY'])
    response = await client.chat.completions.create(
        messages=[{"role": "system", "content": GENERATE_RESPONSE.replace("{context}", context)},
                  {"role": "user", "content": query}], 
        model='gpt-4o-mini', 
        temperature=0
    )
    
    answer = response.choices[0].message.content
    return answer
