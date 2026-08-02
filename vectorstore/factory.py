"""Chroma 向量数据库封装。"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from models.document import RetrievedChunk


class BaseVectorStore(ABC):
    def __init__(self, collection_name: str, dimension: int):
        self.collection_name = collection_name
        self.dimension = dimension

    @abstractmethod
    def add_texts(self, texts, embeddings, metadatas=None, ids=None):
        pass

    @abstractmethod
    def similarity_search(self, query_embedding, top_k=5, filter_dict=None, score_threshold=None):
        pass

    @abstractmethod
    def delete(self, ids=None, filter_dict=None):
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def count(self):
        pass

    @staticmethod
    def _generate_ids(number: int) -> List[str]:
        return [str(uuid.uuid4()) for _ in range(number)]


class ChromaVectorStore(BaseVectorStore):
    def __init__(
        self,
        collection_name: str,
        dimension: int,
        persist_directory: Optional[str] = None,
        distance_metric: str = "cosine",
    ):
        super().__init__(collection_name, dimension)
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:
            raise ImportError("请安装 chromadb: pip install chromadb") from exc

        self.persist_directory = persist_directory or "./chroma_db"
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance_metric, "embedding_dimension": dimension},
        )
        stored_dimension = self.collection.metadata.get("embedding_dimension")
        if stored_dimension and int(stored_dimension) != dimension:
            raise ValueError(
                f"Collection '{collection_name}' 维度为 {stored_dimension}，当前配置为 {dimension}；"
                "请更换 Collection 名称并重新摄取文档"
            )

    def add_texts(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        if not texts:
            return []
        if len(texts) != len(embeddings):
            raise ValueError("文本数量与向量数量不一致")
        ids = ids or self._generate_ids(len(texts))
        metadatas = metadatas or [{} for _ in texts]
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        return ids

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        if self.collection.count() == 0:
            return []
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            where=filter_dict,
            include=["documents", "metadatas", "distances"],
        )
        retrieved = []
        for index, doc_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][index]
            score = 1.0 - distance
            if score_threshold is not None and score < score_threshold:
                continue
            metadata = results["metadatas"][0][index] or {}
            retrieved.append(RetrievedChunk(
                content=results["documents"][0][index],
                metadata={**metadata, "chunk_id": doc_id},
                score=score,
            ))
        return sorted(retrieved, key=lambda item: item.score, reverse=True)

    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        if ids is not None:
            self.collection.delete(ids=ids)
        elif filter_dict is not None:
            self.collection.delete(where=filter_dict)
        else:
            raise ValueError("必须提供 ids 或 filter_dict")

    def clear(self) -> None:
        metadata = dict(self.collection.metadata or {})
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata=metadata,
        )

    def count(self) -> int:
        return self.collection.count()

    def get_all(self, limit: int = 100000) -> List[RetrievedChunk]:
        total = self.collection.count()
        if total == 0:
            return []
        results = self.collection.get(limit=min(limit, total), include=["documents", "metadatas"])
        ids = results.get("ids") or []
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []
        return [
            RetrievedChunk(
                content=document,
                metadata={
                    **(metadatas[index] if index < len(metadatas) else {}),
                    "chunk_id": ids[index] if index < len(ids) else "",
                },
                score=1.0,
            )
            for index, document in enumerate(documents)
        ]


class VectorStoreFactory:
    @staticmethod
    def create(provider: str, collection_name: str, dimension: int, **kwargs) -> BaseVectorStore:
        if provider.lower() != "chroma":
            raise ValueError("本项目仅支持 Chroma 向量数据库")
        return ChromaVectorStore(collection_name, dimension, **kwargs)
