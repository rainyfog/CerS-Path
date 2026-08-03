"""Optional Milvus Lite retrieval backend for the CerS-M demo."""

from __future__ import annotations

from typing import Any


class MilvusHybridRetriever:
    """Retrieve text from a Milvus collection with BGE-M3 embeddings.

    The collection is expected to contain the fields ``dense_vector``,
    ``sparse_vector``, ``text`` and, optionally, ``summary``.
    """

    def __init__(
        self,
        db_path: str,
        collection_name: str,
        embedding_model: str = "BAAI/bge-m3",
        device: str = "cpu",
    ) -> None:
        try:
            from pymilvus import Collection, connections, utility
            from pymilvus.model.hybrid import BGEM3EmbeddingFunction
        except ImportError as exc:
            raise RuntimeError(
                "RAG dependencies are unavailable. Install requirements.txt "
                "from this directory."
            ) from exc

        self._ann_search_request, self._weighted_ranker = _milvus_search_types()
        self.embedder = BGEM3EmbeddingFunction(
            model_name=embedding_model,
            device=device,
            use_fp16=device.startswith("cuda"),
        )

        connections.connect(uri=db_path)
        if not utility.has_collection(collection_name):
            raise ValueError(
                f"Milvus collection {collection_name!r} was not found in {db_path!r}."
            )

        self.collection = Collection(collection_name)
        self.collection.load()

    def run(
        self,
        query_text: str,
        top_k: int = 3,
        sparse_weight: float = 0.3,
        dense_weight: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Return unique retrieval hits ordered by the hybrid score."""
        if not query_text.strip():
            return []

        embeddings = self.embedder.encode_queries([query_text.strip()])
        dense_vector = embeddings["dense"][0]
        sparse_vector = embeddings["sparse"][0]

        dense_request = self._ann_search_request(
            [dense_vector],
            "dense_vector",
            {"metric_type": "IP", "params": {}},
            limit=max(top_k * 2, top_k),
        )
        sparse_request = self._ann_search_request(
            [sparse_vector],
            "sparse_vector",
            {"metric_type": "IP", "params": {}},
            limit=max(top_k * 2, top_k),
        )
        ranker = self._weighted_ranker(sparse_weight, dense_weight)

        results = self.collection.hybrid_search(
            [sparse_request, dense_request],
            rerank=ranker,
            limit=max(top_k * 2, top_k),
            output_fields=["text", "summary"],
        )[0]

        hits: list[dict[str, Any]] = []
        seen: set[str] = set()
        for result in results:
            text = _get_field(result, "text")
            if not text or text in seen:
                continue
            seen.add(text)
            hits.append(
                {
                    "rank": len(hits) + 1,
                    "text": text,
                    "summary": _get_field(result, "summary"),
                    "score": float(result.score),
                }
            )
            if len(hits) == top_k:
                break
        return hits


def _milvus_search_types():
    from pymilvus import AnnSearchRequest, WeightedRanker

    return AnnSearchRequest, WeightedRanker


def _get_field(hit: Any, field: str) -> Any:
    """Support the entity access styles used by multiple PyMilvus releases."""
    try:
        return hit.get(field)
    except (AttributeError, TypeError):
        entity = getattr(hit, "entity", None)
        return entity.get(field) if entity is not None else None

