from typing import List, Dict, Any, Optional
from xml_processor import XMLProcessor
from json_processor import JSONProcessor
from text_processor import TextProcessor
from vector_store import VectorStore


class DocumentRetriever:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.vector_store = VectorStore(persist_directory)
        self.xml_processor = XMLProcessor()
        self.json_processor = JSONProcessor()
        self.text_processor = TextProcessor()

    def add_xml_document(self, xml_content: str, filename: str = None, collection_name: str = None, chunk_size: int = 1000, chunk_overlap: int = 200) -> str:
        document_data = self.xml_processor.extract_text_and_metadata(xml_content, filename)
        return self.vector_store.add_document(document_data, collection_name)

    def add_json_document(self, json_content: str, filename: str = None, collection_name: str = None, chunk_size: int = 1000, chunk_overlap: int = 200) -> str:
        self.json_processor.update_chunk_config(chunk_size, chunk_overlap)
        document_data = self.json_processor.extract_text_and_metadata(json_content, filename)
        return self.vector_store.add_document(document_data, collection_name)

    def add_text_document(self, text_content: str, filename: str = None, collection_name: str = None, chunk_size: int = 1000, chunk_overlap: int = 200) -> str:
        self.text_processor.update_chunk_config(chunk_size, chunk_overlap)
        document_data = self.text_processor.extract_text_and_metadata(text_content, filename)
        return self.vector_store.add_document(document_data, collection_name)

    def add_xml_file(self, file_path: str, collection_name: str = None, chunk_size: int = 1000, chunk_overlap: int = 200) -> str:
        document_data = self.xml_processor.process_file(file_path)
        return self.vector_store.add_document(document_data, collection_name)

    def add_json_file(self, file_path: str, collection_name: str = None, chunk_size: int = 1000, chunk_overlap: int = 200) -> str:
        self.json_processor.update_chunk_config(chunk_size, chunk_overlap)
        document_data = self.json_processor.process_file(file_path)
        return self.vector_store.add_document(document_data, collection_name)

    def add_text_file(self, file_path: str, collection_name: str = None, chunk_size: int = 1000, chunk_overlap: int = 200) -> str:
        self.text_processor.update_chunk_config(chunk_size, chunk_overlap)
        document_data = self.text_processor.process_file(file_path)
        return self.vector_store.add_document(document_data, collection_name)

    def search_documents(self, query: str, n_results: int = 5, where: Optional[Dict] = None, collection_name: str = None) -> List[Dict[str, Any]]:
        return self.vector_store.search(query, n_results, where, collection_name)

    def delete_document(self, document_id: str, collection_name: str = None):
        self.vector_store.delete_document(document_id, collection_name)

    def list_documents(self, collection_name: str = None, page: int = 1, per_page: int = 10,
                      search: str = None, document_type: str = None) -> Dict[str, Any]:
        return self.vector_store.list_documents(
            collection_name=collection_name,
            page=page,
            per_page=per_page,
            search=search,
            document_type=document_type
        )

    def get_document_count(self, collection_name: str = None) -> int:
        return self.vector_store.get_document_count(collection_name)

    def get_document_content(self, document_id: str, collection_name: str = None):
        return self.vector_store.get_document_content(document_id, collection_name)

    def list_collections(self) -> List[str]:
        return self.vector_store.list_collections()

    def create_collection(self, collection_name: str, embedding_model: str = None, chunk_size: int = 1000, chunk_overlap: int = 200) -> bool:
        return self.vector_store.create_collection(collection_name, embedding_model, chunk_size, chunk_overlap)

    def delete_collection(self, collection_name: str) -> bool:
        return self.vector_store.delete_collection(collection_name)

    def get_available_embedding_models(self) -> Dict[str, Dict[str, str]]:
        return self.vector_store.get_available_embedding_models()

    def get_collection_info(self, collection_name: str = None) -> Dict[str, Any]:
        return self.vector_store.get_collection_info(collection_name)

    def _format_sources(self, search_results: List[Dict[str, Any]], collection_name: str) -> List[Dict[str, Any]]:
        formatted_sources = []
        for result in search_results:
            formatted_source = {
                "filename": result["metadata"].get("filename", "unknown"),
                "document_type": result["metadata"].get("document_type", "unknown"),
                "chunk_index": result["metadata"].get("chunk_index", 0),
                "distance": result["distance"],
                "relevance_score": round(1 - (result["distance"] / 2), 3),
                "content_preview": result["document"][:200] + "..." if len(result["document"]) > 200 else result["document"],
                "full_content": result["document"],
                "content": result["document"],
                "document_id": result["metadata"].get("document_id", ""),
                "collection_name": result["metadata"].get("collection_name", collection_name),
                "chunk_length": result["metadata"].get("chunk_length", len(result["document"])),
                "metadata": result["metadata"]
            }

            core_result_fields = {
                "filename", "document_type", "chunk_index", "distance", "relevance_score",
                "content_preview", "full_content", "content", "document_id", "collection_name", "chunk_length", "metadata"
            }

            for field, value in result["metadata"].items():
                if field not in core_result_fields:
                    formatted_source[field] = value

            formatted_sources.append(formatted_source)

        return formatted_sources