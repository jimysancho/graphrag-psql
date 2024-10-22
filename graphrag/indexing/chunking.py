from typing import List
from openai import AsyncClient

import numpy as np
import tiktoken
import os
import re


def count_tokens(text: str) -> int:
    encoder = tiktoken.encoding_for_model("gpt-4o-mini")
    return len(encoder.encode(text))

def split_text(text: str, batch_size: int = 5) -> List[str]:

    sentences = text.split("\n")
    if len(sentences) <= 1:
        sentences = re.split(r"(?<=[.?!])\s+", text)

    batches = []
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i : i + batch_size]
        batch_text = "\n".join(batch_sentences).strip()

        if batch_text:
            batch_text = batch_text.replace("\n\n", "\n")
            batches.append(batch_text)

    new_batches = []
    i = 0
    while i < len(batches):
        current_batch = batches[i]

        while current_batch.endswith("?") and i + 1 < len(batches):
            current_batch += "\n" + batches[i + 1].split("\n", 1)[0]
            i += 1

        if current_batch.endswith("\n"):
            current_batch = current_batch.rstrip("\n")

        new_batches.append(current_batch)
        i += 1

    return [batch for batch in new_batches if batch.strip()]


async def _compute_chunks(
    sentences: List[str],
    min_token_size: int,
    max_token_size: int,
    sim_th: float,
    max_positional_distance: int = 2,
    created_chunks: List[str] | None = None,
    text_embeddings: List[float] | None = None,
) -> List[str]:

    async def create_embeddings(sentences: List[str]) -> List[List[float]]:
        client = AsyncClient(api_key=os.environ["OPENAI_API_KEY"])
        results = await client.embeddings.create(input=sentences, model="text-embedding-3-small")
        return [result.embedding for result in results.data]

    if created_chunks is None:
        created_chunks = []

    if not len(sentences):
        return created_chunks

    if len(sentences) <= 1:
        sentences = split_text(sentences[0])

    tokens = [count_tokens(sentence) for sentence in sentences]
    sentences_array = np.array(sentences)
    text_embeddings = np.array(await create_embeddings(sentences)) if text_embeddings is None else text_embeddings
    tokens_array = np.array(tokens)
    cumulative_tokens = np.cumsum(tokens_array)
    try:
        possible_joins_start_index = np.where(cumulative_tokens > min_token_size)[0][0] + 1
    except IndexError:
        return created_chunks

    try:
        first_chunk = sentences_array[:possible_joins_start_index]
    except IndexError as e:
        sentences_array = np.array([sentences])
        first_chunk = sentences_array

    first_chunk_text = " ".join(first_chunk)
    next_chunk = sentences_array[possible_joins_start_index:]

    if not len(next_chunk):
        if count_tokens(first_chunk_text) > max_token_size and sim_th < 1.0:
            recursive_chunks = await _compute_chunks(
                sentences="\n".join(first_chunk),
                min_token_size=min_token_size,
                max_token_size=max_token_size,
                sim_th=sim_th * 1.05,
                max_positional_distance=max_positional_distance,
                text_embeddings=text_embeddings,
            )

            created_chunks.extend(recursive_chunks)
        else:
            created_chunks.append(first_chunk_text)
        return created_chunks

    try:
        embeddings_current_chunk = text_embeddings[:possible_joins_start_index]
        embeddings_next_chunk = text_embeddings[possible_joins_start_index:]
    except IndexError:
        return created_chunks

    sim_matrix = embeddings_current_chunk.dot(embeddings_next_chunk.T)
    valid_sentences = []
    min_sentences = max(1, len(first_chunk) // 2)

    for i in range(sim_matrix.shape[-1]):
        n_valid = (sim_matrix[:, i] > sim_th).sum()
        if n_valid >= min_sentences:
            valid_sentences.append(i)

    valid_sentences_array = np.array(valid_sentences)
    actual_valid_sentences = None

    try:
        if valid_sentences[0] <= max_positional_distance:
            diff = np.concatenate((np.array([0]), np.diff(valid_sentences_array))) <= max_positional_distance
        else:
            diff = [False]
    except IndexError:
        diff = [False]

    try:
        current_diff = diff[0]
    except IndexError:
        current_diff = False

    k = 0

    if len(diff) > 1:
        while current_diff:
            actual_valid_sentences = valid_sentences_array[k]
            try:
                current_diff = diff[k + 1]
            except IndexError:
                current_diff = False
            k += 1
    else:
        actual_valid_sentences = valid_sentences_array[k] if current_diff else None

    if actual_valid_sentences is not None:
        first_chunk_text += " " + " ".join(next_chunk[: actual_valid_sentences + 1])
    else:
        first_chunk_text = first_chunk_text
        actual_valid_sentences = -1

    if count_tokens(first_chunk_text) > max_token_size:
        recursive_chunks = await _compute_chunks(
            sentences=("\n".join(first_chunk) + "\n".join(next_chunk[: actual_valid_sentences + 1])).split("\n"),
            min_token_size=min_token_size,
            max_token_size=max_token_size,
            sim_th=sim_th * 1.05,
            max_positional_distance=max_positional_distance,
            text_embeddings=text_embeddings[: len(first_chunk) + actual_valid_sentences + 1],
        )

        created_chunks.extend(recursive_chunks)
    else:
        created_chunks.append(first_chunk_text)

    return await _compute_chunks(
        sentences=next_chunk[actual_valid_sentences + 1 :],
        min_token_size=min_token_size,
        max_token_size=max_token_size,
        max_positional_distance=max_positional_distance,
        created_chunks=created_chunks,
        sim_th=sim_th,
        text_embeddings=text_embeddings[len(first_chunk) + actual_valid_sentences + 1 :],
    )


async def create_chunks(
    text: str,
    min_token_size: int = 80,
    max_token_size: int = 180,
    sim_threshold: float = 0.75,
    max_positional_distance: int = 2,
    overlap_percentage: int | float = 15,
):

    text = " ".join(text.split("\n"))

    if overlap_percentage < 0:
        overlap_percentage = 0
    elif overlap_percentage > 100:
        overlap_percentage = 100

    chunks = await _compute_chunks(
        sentences=text.split("\n"),
        min_token_size=min_token_size,
        max_token_size=max_token_size,
        max_positional_distance=max_positional_distance,
        sim_th=sim_threshold,
    )

    if not chunks:  # We deal with the case where no chunks have been found. We return the whole text as a single chunk.
        return [text]

    overlapped_chunks = []
    overlap = None
    for chunk in chunks:
        if overlap is None:
            overlapped_chunks.append(chunk)
        else:
            new_chunk = " ".join(overlap) + chunk
            overlapped_chunks.append(new_chunk)

        words = chunk.split(" ")
        n_words = (
            round(len(words) * overlap_percentage / 100)
            if overlap_percentage > 1
            else round(len(words) * overlap_percentage)
        )
        overlap = words[-n_words:]

    return overlapped_chunks