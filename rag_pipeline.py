import openai
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
from xml_processor import XMLProcessor
from vector_store import VectorStore

load_dotenv()


class RAGPipeline:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.vector_store = VectorStore(persist_directory)
        self.xml_processor = XMLProcessor()

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.openai_client = openai.OpenAI(api_key=api_key)

    def add_xml_document(self, xml_content: str, filename: str = None) -> str:
        document_data = self.xml_processor.extract_text_and_metadata(xml_content, filename)
        return self.vector_store.add_document(document_data)

    def add_xml_file(self, file_path: str) -> str:
        document_data = self.xml_processor.process_file(file_path)
        return self.vector_store.add_document(document_data)

    def search_documents(self, query: str, n_results: int = 5, where: Optional[Dict] = None) -> List[Dict[str, Any]]:
        return self.vector_store.search(query, n_results, where)

    def generate_answer(self, query: str, context_results: List[Dict[str, Any]], model: str = "gpt-3.5-turbo") -> str:
        context_text = "\n\n".join([
            f"Source: {result['metadata'].get('filename', 'unknown')} (chunk {result['metadata'].get('chunk_index', 0)})\n"
            f"Content: {result['document']}"
            for result in context_results
        ])

        system_prompt = """You are a helpful assistant that answers questions based on XML document content.
Use only the provided context to answer questions. If the context doesn't contain enough information
to answer the question, say so clearly. Always cite which source document your answer comes from."""

        user_prompt = f"""Context from XML documents:
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

    def query(self, question: str, n_results: int = 5, where: Optional[Dict] = None, model: str = "gpt-3.5-turbo") -> Dict[str, Any]:
        search_results = self.search_documents(question, n_results, where)

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
                    "chunk_index": result["metadata"].get("chunk_index", 0),
                    "distance": result["distance"],
                    "content_preview": result["document"][:200] + "..." if len(result["document"]) > 200 else result["document"]
                }
                for result in search_results
            ],
            "query": question
        }

    def delete_document(self, document_id: str):
        self.vector_store.delete_document(document_id)

    def list_documents(self) -> List[Dict[str, Any]]:
        return self.vector_store.list_documents()

    def get_document_count(self) -> int:
        return self.vector_store.get_document_count()