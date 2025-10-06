import os
import json
import uuid
from typing import Dict, Any, List, Optional
from pathlib import Path
import shutil
from datetime import datetime


class DocumentStorage:
    """
    Manages document storage on filesystem with collection support.

    Directory structure:
    storage_root/
    ├── collections/
    │   ├── documents/           # default collection
    │   │   ├── metadata.json    # collection metadata
    │   │   ├── documents/       # actual document files
    │   │   │   ├── doc_id_1.xml
    │   │   │   ├── doc_id_2.json
    │   │   │   └── doc_id_3.txt
    │   │   └── index.json       # document index with metadata
    │   └── custom_collection/
    │       ├── metadata.json
    │       ├── documents/
    │       └── index.json
    """

    def __init__(self, storage_root: str = "./document_storage"):
        self.storage_root = Path(storage_root)
        self.collections_dir = self.storage_root / "collections"
        self._ensure_storage_structure()

    def _ensure_storage_structure(self):
        """Ensure the storage directory structure exists"""
        self.storage_root.mkdir(exist_ok=True)
        self.collections_dir.mkdir(exist_ok=True)

        # Ensure default collection exists
        self._ensure_collection_exists("documents")

    def _ensure_collection_exists(self, collection_name: str):
        """Ensure a collection directory structure exists"""
        collection_dir = self.collections_dir / collection_name
        collection_dir.mkdir(exist_ok=True)

        documents_dir = collection_dir / "documents"
        documents_dir.mkdir(exist_ok=True)

        # Create collection metadata if it doesn't exist
        metadata_file = collection_dir / "metadata.json"
        if not metadata_file.exists():
            metadata = {
                "name": collection_name,
                "created_at": datetime.utcnow().isoformat(),
                "description": f"Document collection: {collection_name}",
                "document_count": 0
            }
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

        # Create document index if it doesn't exist
        index_file = collection_dir / "index.json"
        if not index_file.exists():
            with open(index_file, 'w') as f:
                json.dump({}, f, indent=2)

    def store_document(self, content: str, filename: str, metadata: Dict[str, Any],
                      collection_name: str = "documents") -> str:
        """
        Store a document and its metadata

        Args:
            content: Document content
            filename: Original filename
            metadata: Document metadata
            collection_name: Collection to store in

        Returns:
            Document ID
        """
        self._ensure_collection_exists(collection_name)

        doc_id = str(uuid.uuid4())
        collection_dir = self.collections_dir / collection_name
        documents_dir = collection_dir / "documents"

        # Determine file extension from original filename or document type
        if filename:
            ext = Path(filename).suffix
        else:
            doc_type = metadata.get('document_type', 'txt')
            ext = f'.{doc_type}'

        # Store the document file
        doc_filename = f"{doc_id}{ext}"
        doc_path = documents_dir / doc_filename

        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(content)

        # Update document index
        index_file = collection_dir / "index.json"
        with open(index_file, 'r') as f:
            index = json.load(f)

        # Store document metadata in index
        document_metadata = {
            "document_id": doc_id,
            "filename": filename,
            "original_filename": filename,
            "stored_filename": doc_filename,
            "file_path": str(doc_path.relative_to(self.storage_root)),
            "collection_name": collection_name,
            "stored_at": datetime.utcnow().isoformat(),
            "file_size": len(content.encode('utf-8')),
            **metadata  # Include all processing metadata
        }

        index[doc_id] = document_metadata

        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2)

        # Update collection metadata
        self._update_collection_count(collection_name)

        return doc_id

    def get_document_content(self, doc_id: str, collection_name: str = None) -> Optional[str]:
        """Get document content by ID"""
        doc_metadata = self.get_document_metadata(doc_id, collection_name)
        if not doc_metadata:
            return None

        file_path = self.storage_root / doc_metadata["file_path"]
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return None

    def get_document_metadata(self, doc_id: str, collection_name: str = None) -> Optional[Dict[str, Any]]:
        """Get document metadata by ID"""
        if collection_name:
            collections_to_search = [collection_name]
        else:
            collections_to_search = self.list_collections()

        for collection in collections_to_search:
            index_file = self.collections_dir / collection / "index.json"
            if index_file.exists():
                with open(index_file, 'r') as f:
                    index = json.load(f)
                    if doc_id in index:
                        return index[doc_id]

        return None

    def list_documents(self, collection_name: str = None, search: str = None,
                      document_type: str = None, page: int = 1, per_page: int = 10) -> Dict[str, Any]:
        """
        List documents with filtering and pagination

        Args:
            collection_name: Collection to search (None for all)
            search: Search term for filename/content
            document_type: Filter by document type
            page: Page number (1-based)
            per_page: Documents per page

        Returns:
            Dictionary with documents list and pagination info
        """
        all_documents = []

        collections_to_search = [collection_name] if collection_name else self.list_collections()

        for collection in collections_to_search:
            index_file = self.collections_dir / collection / "index.json"
            if index_file.exists():
                try:
                    with open(index_file, 'r') as f:
                        index = json.load(f)
                        all_documents.extend(index.values())
                except (json.JSONDecodeError, FileNotFoundError):
                    # Skip corrupted or missing index files
                    continue

        # Apply filters
        filtered_documents = all_documents

        if search:
            search_lower = search.lower()
            filtered_documents = [
                doc for doc in filtered_documents
                if (search_lower in doc.get('filename', '').lower() or
                    search_lower in doc.get('title', '').lower())
            ]

        if document_type:
            filtered_documents = [
                doc for doc in filtered_documents
                if doc.get('document_type', '').lower() == document_type.lower()
            ]

        # Sort by stored_at (newest first)
        filtered_documents.sort(key=lambda x: x.get('stored_at', ''), reverse=True)

        # Apply pagination
        total_count = len(filtered_documents)
        total_pages = (total_count + per_page - 1) // per_page if total_count > 0 else 1
        page = max(1, min(page, total_pages))

        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_documents = filtered_documents[start_idx:end_idx]

        return {
            "documents": paginated_documents,
            "total_count": total_count,
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1,
            "start_index": start_idx + 1 if paginated_documents else 0,
            "end_index": min(end_idx, total_count)
        }

    def delete_document(self, doc_id: str, collection_name: str = None) -> bool:
        """Delete a document and its metadata"""
        doc_metadata = self.get_document_metadata(doc_id, collection_name)
        if not doc_metadata:
            return False

        actual_collection = doc_metadata["collection_name"]

        # Delete the file
        file_path = self.storage_root / doc_metadata["file_path"]
        try:
            file_path.unlink()
        except FileNotFoundError:
            pass  # File already gone

        # Remove from index
        index_file = self.collections_dir / actual_collection / "index.json"
        with open(index_file, 'r') as f:
            index = json.load(f)

        if doc_id in index:
            del index[doc_id]

            with open(index_file, 'w') as f:
                json.dump(index, f, indent=2)

            # Update collection count
            self._update_collection_count(actual_collection)
            return True

        return False

    def list_collections(self) -> List[str]:
        """List all available collections"""
        if not self.collections_dir.exists():
            return ["documents"]

        return [d.name for d in self.collections_dir.iterdir() if d.is_dir()]

    def create_collection(self, collection_name: str) -> bool:
        """Create a new collection"""
        if collection_name in self.list_collections():
            return False

        self._ensure_collection_exists(collection_name)
        return True

    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection and all its data"""
        if collection_name == "documents":
            raise ValueError("Cannot delete the default collection")

        collection_dir = self.collections_dir / collection_name
        if not collection_dir.exists():
            return False

        try:
            # Delete the entire collection directory
            import shutil
            shutil.rmtree(collection_dir)
            return True
        except Exception as e:
            print(f"Error deleting collection directory {collection_dir}: {e}")
            return False

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics for a collection"""
        if collection_name not in self.list_collections():
            return {}

        metadata_file = self.collections_dir / collection_name / "metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        index_file = self.collections_dir / collection_name / "index.json"
        with open(index_file, 'r') as f:
            index = json.load(f)

        # Calculate actual document count and types
        doc_types = {}
        total_size = 0

        for doc in index.values():
            doc_type = doc.get('document_type', 'unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            total_size += doc.get('file_size', 0)

        return {
            **metadata,
            "actual_document_count": len(index),
            "document_types": doc_types,
            "total_size_bytes": total_size
        }

    def _update_collection_count(self, collection_name: str):
        """Update the document count in collection metadata"""
        metadata_file = self.collections_dir / collection_name / "metadata.json"
        index_file = self.collections_dir / collection_name / "index.json"

        with open(index_file, 'r') as f:
            index = json.load(f)

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        metadata["document_count"] = len(index)
        metadata["last_updated"] = datetime.utcnow().isoformat()

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)