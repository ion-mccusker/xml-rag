import openai
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
from xml_processor import XMLProcessor
from json_processor import JSONProcessor
from text_processor import TextProcessor
from vector_store import VectorStore

load_dotenv()


class RAGPipeline:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.vector_store = VectorStore(persist_directory)
        self.xml_processor = XMLProcessor()
        self.json_processor = JSONProcessor()
        self.text_processor = TextProcessor()

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.openai_client = openai.OpenAI(api_key=api_key)

    def add_xml_document(self, xml_content: str, filename: str = None, collection_name: str = None) -> str:
        document_data = self.xml_processor.extract_text_and_metadata(xml_content, filename)
        return self.vector_store.add_document(document_data, collection_name)

    def add_json_document(self, json_content: str, filename: str = None, collection_name: str = None) -> str:
        document_data = self.json_processor.extract_text_and_metadata(json_content, filename)
        return self.vector_store.add_document(document_data, collection_name)

    def add_xml_file(self, file_path: str, collection_name: str = None) -> str:
        document_data = self.xml_processor.process_file(file_path)
        return self.vector_store.add_document(document_data, collection_name)

    def add_text_document(self, text_content: str, filename: str = None, collection_name: str = None) -> str:
        document_data = self.text_processor.extract_text_and_metadata(text_content, filename)
        return self.vector_store.add_document(document_data, collection_name)

    def add_json_file(self, file_path: str, collection_name: str = None) -> str:
        document_data = self.json_processor.process_file(file_path)
        return self.vector_store.add_document(document_data, collection_name)

    def add_text_file(self, file_path: str, collection_name: str = None) -> str:
        document_data = self.text_processor.process_file(file_path)
        return self.vector_store.add_document(document_data, collection_name)

    def search_documents(self, query: str, n_results: int = 5, where: Optional[Dict] = None, collection_name: str = None) -> List[Dict[str, Any]]:
        return self.vector_store.search(query, n_results, where, collection_name)

    def generate_answer(self, query: str, context_results: List[Dict[str, Any]], model: str = "gpt-3.5-turbo") -> str:
        context_text = "\n\n".join([
            f"Source: {result['metadata'].get('filename', 'unknown')} (chunk {result['metadata'].get('chunk_index', 0)})\n"
            f"Content: {result['document']}"
            for result in context_results
        ])

        system_prompt = """You are a helpful assistant that answers questions based on document content (XML, JSON, and text files).
Use only the provided context to answer questions. If the context doesn't contain enough information
to answer the question, say so clearly. Always cite which source document your answer comes from."""

        user_prompt = f"""Context from documents:
{context_text}

Question: {query}

Please provide a comprehensive answer based on the context above."""

        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def query(self, question: str, n_results: int = 5, where: Optional[Dict] = None, model: str = "gpt-3.5-turbo", collection_name: str = None) -> Dict[str, Any]:
        search_results = self.search_documents(question, n_results, where, collection_name)

        if not search_results:
            return {
                "answer": "No relevant documents found for your question.",
                "sources": [],
                "query": question
            }

        answer = self.generate_answer(question, search_results, model)

        return {
            "answer": answer,
            "sources": [
                {
                    "filename": result["metadata"].get("filename", "unknown"),
                    "document_type": result["metadata"].get("document_type", "unknown"),
                    "chunk_index": result["metadata"].get("chunk_index", 0),
                    "distance": result["distance"],
                    "relevance_score": round(1 - result["distance"], 3),
                    "content_preview": result["document"][:200] + "..." if len(result["document"]) > 200 else result["document"],
                    "full_content": result["document"],
                    "content": result["document"],
                    "document_id": result["metadata"].get("document_id", ""),
                    "metadata": result["metadata"]
                }
                for result in search_results
            ],
            "query": question,
            "retrieved_chunks": search_results
        }

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

    def create_collection(self, collection_name: str) -> bool:
        return self.vector_store.create_collection(collection_name)