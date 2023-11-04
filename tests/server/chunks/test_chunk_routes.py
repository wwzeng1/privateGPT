from pathlib import Path

from fastapi.testclient import TestClient

from private_gpt.server.chunks.chunks_router import ChunksBody, ChunksResponse
from tests.fixtures.ingest_helper import IngestHelper


def test_chunks_retrieval(test_client: TestClient, ingest_helper: IngestHelper) -> None:
    # Make sure there is at least some chunk to query in the database
    path = Path(__file__).parents[0] / "chunk_test.txt"
    ingest_helper.ingest_file(path)

    body = ChunksBody(text="b483dd15-78c4-4d67-b546-21a0d690bf43")
    response = test_client.post("/v1/chunks", json=body.model_dump())
    assert response.status_code == 200
    chunk_response = ChunksResponse.model_validate(response.json())
    assert len(chunk_response.data) > 0
def test_create_vector_store_index(test_client: TestClient, ingest_helper: IngestHelper) -> None:
    # Setup
    path = Path(__file__).parents[0] / "chunk_test.txt"
    ingest_helper.ingest_file(path)
    service = root_injector.get(ChunksService)

    # Call function
    index = service.create_vector_store_index()

    # Assert index is created correctly
    assert isinstance(index, VectorStoreIndex)

def test_retrieve_nodes(test_client: TestClient, ingest_helper: IngestHelper) -> None:
    # Setup
    path = Path(__file__).parents[0] / "chunk_test.txt"
    ingest_helper.ingest_file(path)
    service = root_injector.get(ChunksService)
    vector_index_retriever = service.create_vector_store_index()

    # Call function
    nodes = service.retrieve_nodes(vector_index_retriever, "test text")

    # Assert nodes are retrieved correctly
    assert isinstance(nodes, list)
    assert all(isinstance(node, NodeWithScore) for node in nodes)

def test_sort_nodes(test_client: TestClient, ingest_helper: IngestHelper) -> None:
    # Setup
    path = Path(__file__).parents[0] / "chunk_test.txt"
    ingest_helper.ingest_file(path)
    service = root_injector.get(ChunksService)
    vector_index_retriever = service.create_vector_store_index()
    nodes = service.retrieve_nodes(vector_index_retriever, "test text")

    # Call function
    sorted_nodes = service.sort_nodes(nodes)

    # Assert nodes are sorted correctly
    assert sorted_nodes == sorted(nodes, key=lambda n: n.score or 0.0, reverse=True)

def test_create_chunks_from_nodes(test_client: TestClient, ingest_helper: IngestHelper) -> None:
    # Setup
    path = Path(__file__).parents[0] / "chunk_test.txt"
    ingest_helper.ingest_file(path)
    service = root_injector.get(ChunksService)
    vector_index_retriever = service.create_vector_store_index()
    nodes = service.retrieve_nodes(vector_index_retriever, "test text")
    sorted_nodes = service.sort_nodes(nodes)

    # Call function
    chunks = service.create_chunks_from_nodes(sorted_nodes, 2)

    # Assert chunks are created correctly
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
