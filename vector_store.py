import uuid
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from document_storage import DocumentStorage
import json
import os


class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db", storage_directory: str = "./document_storage"):
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # Initialize available embedding models
        self.available_models = {
            "all-MiniLM-L6-v2": {
                "model": None,
                "name": "MiniLM (Default)",
                "description": "All-MiniLM-L6-v2 - Fast and efficient"
            },
            "google/embeddinggemma-300m": {
                "model": None,
                "name": "EmbeddingGemma",
                "description": "Google EmbeddingGemma-300m - High quality embeddings"
            }
        }

        # Cache for loaded embedding models
        self._embedding_models = {}

        # Default embedding model
        self.default_embedding_model = "all-MiniLM-L6-v2"
        self.default_collection_name = "documents"

        # Initialize document storage
        self.document_storage = DocumentStorage(storage_directory)

        # Cache for collections to avoid repeated lookups
        self._collections = {}

        # Collection metadata file to store embedding model info
        self.collection_metadata_file = os.path.join(persist_directory, "collection_metadata.json")
        self._collection_metadata = self._load_collection_metadata()

        # Ensure default collection exists
        self._get_or_create_collection(self.default_collection_name)

    def _load_collection_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load collection metadata from file"""
        if os.path.exists(self.collection_metadata_file):
            try:
                with open(self.collection_metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading collection metadata: {e}")
        return {}

    def _save_collection_metadata(self):
        """Save collection metadata to file"""
        try:
            with open(self.collection_metadata_file, 'w') as f:
                json.dump(self._collection_metadata, f, indent=2)
        except Exception as e:
            print(f"Error saving collection metadata: {e}")

    def _get_embedding_model(self, model_name: str) -> SentenceTransformer:
        """Get or load an embedding model"""
        if model_name not in self._embedding_models:
            print(f"Loading embedding model: {model_name}")
            try:
                self._embedding_models[model_name] = SentenceTransformer(model_name)
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")
                # Fallback to default model
                if model_name != self.default_embedding_model:
                    print(f"Falling back to default model: {self.default_embedding_model}")
                    return self._get_embedding_model(self.default_embedding_model)
                raise
        return self._embedding_models[model_name]

    def _get_collection_embedding_model(self, collection_name: str) -> str:
        """Get the embedding model used by a collection"""
        return self._collection_metadata.get(collection_name, {}).get("embedding_model", self.default_embedding_model)

    def _format_query_for_model(self, query: str, model_name: str) -> str:
        """Format query with model-specific prompt template"""
        if model_name == "google/embeddinggemma-300m":
            return f"task: search result | query: {query}"
        return query

    def _format_document_for_model(self, document: str, model_name: str, title: str = None) -> str:
        """Format document with model-specific prompt template"""
        if model_name == "google/embeddinggemma-300m":
            title_part = title if title else "none"
            return f"title: {title_part} | text: {document}"
        return document

    def _get_or_create_collection(self, collection_name: str, embedding_model: str = None):
        """Get or create a collection and cache it"""
        if collection_name in self._collections:
            return self._collections[collection_name]

        # Set default embedding model for collection if not specified
        if embedding_model is None:
            embedding_model = self._get_collection_embedding_model(collection_name)

        # Store/update collection metadata
        if collection_name not in self._collection_metadata:
            self._collection_metadata[collection_name] = {
                "embedding_model": embedding_model,
                "created_at": str(uuid.uuid4())  # Timestamp placeholder
            }
            self._save_collection_metadata()

        try:
            collection = self.client.get_collection(collection_name)
        except:
            collection = self.client.create_collection(
                name=collection_name,
                metadata={
                    "description": "XML, JSON, and text documents with metadata",
                    "embedding_model": embedding_model
                }
            )

        self._collections[collection_name] = collection
        return collection

    def list_collections(self) -> List[str]:
        """List all available collections from document storage"""
        return self.document_storage.list_collections()

    def create_collection(self, collection_name: str, embedding_model: str = None, chunk_size: int = 1000, chunk_overlap: int = 200) -> bool:
        """Create a new collection in both vector DB and document storage"""
        if embedding_model is None:
            embedding_model = self.default_embedding_model

        try:
            # Create in document storage first
            storage_success = self.document_storage.create_collection(collection_name)
            if not storage_success:
                return False

            # Create in vector database
            if collection_name in [col.name for col in self.client.list_collections()]:
                # Update metadata for existing collection
                self._collection_metadata[collection_name] = {
                    "embedding_model": embedding_model,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "created_at": str(uuid.uuid4())
                }
                self._save_collection_metadata()
                return True

            collection = self.client.create_collection(
                name=collection_name,
                metadata={
                    "description": "XML, JSON, and text documents with metadata",
                    "embedding_model": embedding_model,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap
                }
            )
            self._collections[collection_name] = collection

            # Store collection metadata
            self._collection_metadata[collection_name] = {
                "embedding_model": embedding_model,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "created_at": str(uuid.uuid4())
            }
            self._save_collection_metadata()

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

        # Get the embedding model for this collection
        embedding_model_name = self._get_collection_embedding_model(collection_name)
        embedding_model = self._get_embedding_model(embedding_model_name)

        # Format documents for the specific embedding model
        formatted_chunks = []
        for chunk in chunk_documents:
            formatted_chunk = self._format_document_for_model(
                chunk,
                embedding_model_name,
                title=metadata.get("title") or metadata.get("filename")
            )
            formatted_chunks.append(formatted_chunk)

        chunk_embeddings = embedding_model.encode(formatted_chunks).tolist()

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

        # Get the embedding model for this collection
        embedding_model_name = self._get_collection_embedding_model(collection_name)
        embedding_model = self._get_embedding_model(embedding_model_name)

        # Format query for the specific embedding model
        formatted_query = self._format_query_for_model(query, embedding_model_name)
        query_embedding = embedding_model.encode([formatted_query]).tolist()

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

    def get_available_embedding_models(self) -> Dict[str, Dict[str, str]]:
        """Get list of available embedding models"""
        return {
            model_id: {
                "name": info["name"],
                "description": info["description"]
            }
            for model_id, info in self.available_models.items()
        }

    def get_collection_info(self, collection_name: str = None) -> Dict[str, Any]:
        """Get collection information including embedding model"""
        if collection_name is None:
            collection_name = self.default_collection_name

        embedding_model = self._get_collection_embedding_model(collection_name)
        model_info = self.available_models.get(embedding_model, {})

        chunk_size = self._collection_metadata.get(collection_name, {}).get("chunk_size", 1000)
        chunk_overlap = self._collection_metadata.get(collection_name, {}).get("chunk_overlap", 200)

        return {
            "collection_name": collection_name,
            "embedding_model": embedding_model,
            "embedding_model_name": model_info.get("name", embedding_model),
            "embedding_model_description": model_info.get("description", "Unknown model"),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        }

    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection and all its data"""
        if collection_name == self.default_collection_name:
            raise ValueError("Cannot delete the default collection")

        try:
            # Delete from ChromaDB
            self.client.delete_collection(collection_name)

            # Delete all documents from filesystem storage for this collection
            self.document_storage.delete_collection(collection_name)

            # Remove from collection metadata and cache
            if collection_name in self._collection_metadata:
                del self._collection_metadata[collection_name]

            if collection_name in self._collections:
                del self._collections[collection_name]

            return True
        except Exception as e:
            print(f"Error deleting collection {collection_name}: {e}")
            return False

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