import uuid
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.collection_name = "documents"

        try:
            self.collection = self.client.get_collection(self.collection_name)
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "XML, JSON, and text documents with metadata"}
            )

    def add_document(self, document_data: Dict[str, Any]) -> str:
        doc_id = str(uuid.uuid4())

        text_chunks = document_data.get("text_chunks", [])
        metadata = document_data.get("metadata", {})

        if not text_chunks:
            raise ValueError("No text chunks found in document")

        chunk_ids = []
        chunk_embeddings = []
        chunk_metadatas = []
        chunk_documents = []

        for i, chunk in enumerate(text_chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            chunk_ids.append(chunk_id)

            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_index": i,
                "document_id": doc_id,
                "chunk_length": len(chunk)
            })

            chunk_metadatas.append(chunk_metadata)
            chunk_documents.append(chunk)
        chunk_embeddings = self.embedding_model.encode(chunk_documents).tolist()

        self.collection.add(
            embeddings=chunk_embeddings,
            documents=chunk_documents,
            metadatas=chunk_metadatas,
            ids=chunk_ids
        )

        return doc_id

    def search(self, query: str, n_results: int = 5, where: Optional[Dict] = None) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_model.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        formatted_results = []
        for i in range(len(results["documents"][0])):
            formatted_results.append({
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "id": results["ids"][0][i]
            })

        return formatted_results

    def delete_document(self, document_id: str):
        existing_chunks = self.collection.get(
            where={"document_id": document_id}
        )

        if existing_chunks["ids"]:
            self.collection.delete(ids=existing_chunks["ids"])

    def get_document_count(self) -> int:
        return self.collection.count()

    def list_documents(self) -> List[Dict[str, Any]]:
        all_data = self.collection.get(include=["metadatas"])

        documents = {}
        for metadata in all_data["metadatas"]:
            doc_id = metadata.get("document_id")
            if doc_id and doc_id not in documents:
                doc_type = metadata.get("document_type", "xml")

                if doc_type == "json":
                    documents[doc_id] = {
                        "document_id": doc_id,
                        "filename": metadata.get("filename", "unknown"),
                        "document_type": doc_type,
                        "data_type": metadata.get("data_type", ""),
                        "total_keys": metadata.get("total_keys", 0),
                        "structure_summary": metadata.get("structure_summary", "")
                    }
                elif doc_type == "text":
                    documents[doc_id] = {
                        "document_id": doc_id,
                        "filename": metadata.get("filename", "unknown"),
                        "document_type": doc_type,
                        "total_words": metadata.get("total_words", 0),
                        "total_lines": metadata.get("total_lines", 0),
                        "title": metadata.get("title", ""),
                        "language_hints": metadata.get("language_hints", [])
                    }
                else:
                    documents[doc_id] = {
                        "document_id": doc_id,
                        "filename": metadata.get("filename", "unknown"),
                        "document_type": doc_type,
                        "root_tag": metadata.get("root_tag", ""),
                        "element_count": metadata.get("element_count", 0)
                    }

        return list(documents.values())

    def get_document_content(self, document_id: str) -> Optional[Dict[str, Any]]:
        chunks = self.collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"]
        )

        if not chunks["documents"]:
            return None

        # Sort chunks by chunk_index to reconstruct original order
        chunk_data = list(zip(chunks["documents"], chunks["metadatas"]))
        chunk_data.sort(key=lambda x: x[1].get("chunk_index", 0))

        sorted_chunks = [chunk[0] for chunk in chunk_data]
        metadata = chunk_data[0][1] if chunk_data else {}

        # Remove chunk-specific metadata
        clean_metadata = {k: v for k, v in metadata.items()
                         if k not in ["chunk_index", "chunk_length"]}

        return {
            "document_id": document_id,
            "metadata": clean_metadata,
            "chunks": sorted_chunks,
            "full_content": "\n\n".join(sorted_chunks),
            "total_chunks": len(sorted_chunks)
        }