import uuid
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from document_storage import DocumentStorage


class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db", storage_directory: str = "./document_storage"):
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.default_collection_name = "documents"

        # Initialize document storage
        self.document_storage = DocumentStorage(storage_directory)

        # Cache for collections to avoid repeated lookups
        self._collections = {}

        # Ensure default collection exists
        self._get_or_create_collection(self.default_collection_name)

    def _get_or_create_collection(self, collection_name: str):
        """Get or create a collection and cache it"""
        if collection_name in self._collections:
            return self._collections[collection_name]

        try:
            collection = self.client.get_collection(collection_name)
        except:
            collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "XML, JSON, and text documents with metadata"}
            )

        self._collections[collection_name] = collection
        return collection

    def list_collections(self) -> List[str]:
        """List all available collections from document storage"""
        return self.document_storage.list_collections()

    def create_collection(self, collection_name: str) -> bool:
        """Create a new collection in both vector DB and document storage"""
        try:
            # Create in document storage first
            storage_success = self.document_storage.create_collection(collection_name)
            if not storage_success:
                return False

            # Create in vector database
            if collection_name in [col.name for col in self.client.list_collections()]:
                return True  # Vector collection already exists, storage creation succeeded

            collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "XML, JSON, and text documents with metadata"}
            )
            self._collections[collection_name] = collection
            return True
        except Exception as e:
            print(f"Error creating collection {collection_name}: {e}")
            return False

    def add_document(self, document_data: Dict[str, Any], collection_name: str = None) -> str:
        if collection_name is None:
            collection_name = self.default_collection_name

        collection = self._get_or_create_collection(collection_name)

        text_chunks = document_data.get("text_chunks", [])
        metadata = document_data.get("metadata", {})
        full_content = document_data.get("full_content", "")
        filename = metadata.get("filename", "unknown")

        if not text_chunks:
            raise ValueError("No text chunks found in document")

        # Store the full document in filesystem
        doc_id = self.document_storage.store_document(
            content=full_content,
            filename=filename,
            metadata=metadata,
            collection_name=collection_name
        )

        # Store only chunk embeddings and minimal metadata in vector DB
        chunk_ids = []
        chunk_embeddings = []
        chunk_metadatas = []
        chunk_documents = []

        for i, chunk in enumerate(text_chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            chunk_ids.append(chunk_id)

            # Minimal metadata for vector search - no full content
            chunk_metadata = {
                "chunk_index": i,
                "document_id": doc_id,
                "chunk_length": len(chunk),
                "filename": filename,
                "document_type": metadata.get("document_type", "unknown"),
                "collection_name": collection_name
            }

            chunk_metadatas.append(chunk_metadata)
            # Store chunk text for search, but full content is in filesystem
            chunk_documents.append(chunk)

        chunk_embeddings = self.embedding_model.encode(chunk_documents).tolist()

        collection.add(
            embeddings=chunk_embeddings,
            documents=chunk_documents,
            metadatas=chunk_metadatas,
            ids=chunk_ids
        )

        return doc_id

    def search(self, query: str, n_results: int = 5, where: Optional[Dict] = None, collection_name: str = None) -> List[Dict[str, Any]]:
        if collection_name is None:
            collection_name = self.default_collection_name

        collection = self._get_or_create_collection(collection_name)
        query_embedding = self.embedding_model.encode([query]).tolist()

        results = collection.query(
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

    def delete_document(self, document_id: str, collection_name: str = None):
        if collection_name is None:
            collection_name = self.default_collection_name

        # Delete from filesystem storage
        self.document_storage.delete_document(document_id, collection_name)

        # Delete from vector database
        collection = self._get_or_create_collection(collection_name)
        existing_chunks = collection.get(
            where={"document_id": document_id}
        )

        if existing_chunks["ids"]:
            collection.delete(ids=existing_chunks["ids"])

    def get_document_count(self, collection_name: str = None) -> int:
        if collection_name is None:
            collection_name = self.default_collection_name

        collection = self._get_or_create_collection(collection_name)
        return collection.count()

    def list_documents(self, collection_name: str = None, page: int = 1, per_page: int = 10,
                      search: str = None, document_type: str = None) -> Dict[str, Any]:
        """Use filesystem storage for efficient document listing with pagination"""
        return self.document_storage.list_documents(
            collection_name=collection_name,
            search=search,
            document_type=document_type,
            page=page,
            per_page=per_page
        )

    def get_document_content(self, document_id: str, collection_name: str = None) -> Optional[Dict[str, Any]]:
        """Get document content from filesystem storage"""
        # Get metadata from filesystem
        doc_metadata = self.document_storage.get_document_metadata(document_id, collection_name)
        if not doc_metadata:
            return None

        # Get full content from filesystem
        full_content = self.document_storage.get_document_content(document_id, collection_name)
        if full_content is None:
            return None

        # Get chunks from vector database for compatibility
        if collection_name is None:
            collection_name = self.default_collection_name

        collection = self._get_or_create_collection(collection_name)
        chunks = collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"]
        )

        # Sort chunks by chunk_index
        chunk_data = []
        if chunks["documents"]:
            chunk_data = list(zip(chunks["documents"], chunks["metadatas"]))
            chunk_data.sort(key=lambda x: x[1].get("chunk_index", 0))

        sorted_chunks = [chunk[0] for chunk in chunk_data] if chunk_data else []

        return {
            "document_id": document_id,
            "metadata": doc_metadata,
            "chunks": sorted_chunks,
            "full_content": full_content,
            "total_chunks": len(sorted_chunks)
        }