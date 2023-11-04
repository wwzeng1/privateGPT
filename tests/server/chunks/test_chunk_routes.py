from pathlib import Path

from fastapi.testclient import TestClient

from private_gpt.server.chunks.chunks_router import ChunksBody, ChunksResponse
from tests.fixtures.ingest_helper import IngestHelper


def test_chunks_retrieval(test_client: TestClient, ingest_helper: IngestHelper) -> None:
    # Setup
    text = "b483dd15-78c4-4d67-b546-21a0d690bf43"
    context_filter = None
    limit = 10
    prev_next_chunks = 2
    service = root_injector.get(ChunksService)

    # Exercise
    chunks = service.retrieve_relevant(text, context_filter, limit, prev_next_chunks)

    # Verify
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
def test_create_index_and_retriever(test_client: TestClient, ingest_helper: IngestHelper) -> None:
    # Setup
    context_filter = None
    limit = 10
    service = root_injector.get(ChunksService)

    # Exercise
    index, retriever = service.create_index_and_retriever(context_filter, limit)

    # Verify
    assert isinstance(index, VectorStoreIndex)
    assert callable(retriever.retrieve)

def test_retrieve_and_sort_nodes(test_client: TestClient, ingest_helper: IngestHelper) -> None:
    # Setup
    text = "b483dd15-78c4-4d67-b546-21a0d690bf43"
    context_filter = None
    limit = 10
    service = root_injector.get(ChunksService)
    _, retriever = service.create_index_and_retriever(context_filter, limit)

    # Exercise
    nodes = service.retrieve_and_sort_nodes(retriever, text)

    # Verify
    assert isinstance(nodes, list)
    assert all(isinstance(node, NodeWithScore) for node in nodes)
    assert all(nodes[i].score >= nodes[i+1].score for i in range(len(nodes)-1))

def test_construct_chunks(test_client: TestClient, ingest_helper: IngestHelper) -> None:
    # Setup
    text = "b483dd15-78c4-4d67-b546-21a0d690bf43"
    context_filter = None
    limit = 10
    prev_next_chunks = 2
    service = root_injector.get(ChunksService)
    _, retriever = service.create_index_and_retriever(context_filter, limit)
    nodes = service.retrieve_and_sort_nodes(retriever, text)

    # Exercise
    chunks = service.construct_chunks(nodes, prev_next_chunks)

    # Verify
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
