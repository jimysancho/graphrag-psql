from openai import AsyncClient
from graphrag.llm.prompt import (BAD_PYTHON_DICTIONARY_PARSING, 
                                 ENTITY_EXTRACTION_PROMPT,
                                 CONTINUE_EXTRACTING_ENTITIES)

from typing import Dict, Any
import os


entity_types = ", ".join([
    "Sintoma", "Medicacion", "Enfermedad", "Diagnostico"
])

RETRIES = 3

async def extract_entities_completion(chunk: str, history: str | None=None) -> Dict[str, Any] | None:

    client = AsyncClient(api_key=os.environ['OPENAI_API_KEY'])
    
    for _ in range(RETRIES):
        if history is None:
            response = await client.chat.completions.create(
                messages=[
                    {"role": "system", "content": ENTITY_EXTRACTION_PROMPT.replace("{entity_types}", entity_types).replace("{text}", chunk)}
                    ], 
                model='gpt-4o-mini'
            )
        else:
            history = ENTITY_EXTRACTION_PROMPT.replace("{entity_types}", entity_types).replace("{text}", chunk) + history
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
