import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
from xml_processor import XMLProcessor
from json_processor import JSONProcessor
from text_processor import TextProcessor
from vector_store import VectorStore

load_dotenv()


class HuggingFaceRAGPipeline:
    def __init__(self, persist_directory: str = "./chroma_db", model_name: str = "microsoft/DialoGPT-medium"):
        self.vector_store = VectorStore(persist_directory)
        self.xml_processor = XMLProcessor()
        self.json_processor = JSONProcessor()
        self.text_processor = TextProcessor()

        # Initialize HuggingFace model
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            print(f"Loading model {model_name} on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                use_safetensors=True,
            )

            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            print(f"Successfully loaded {model_name}")

        except Exception as e:
            print(f"Error loading {model_name}: {str(e)}")
            print("Falling back to microsoft/DialoGPT-medium")
            # Fallback to a smaller, more reliable model
            self.model_name = "microsoft/DialoGPT-medium"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def add_xml_document(self, xml_content: str, filename: str = None, collection_name: str = None, chunk_size: int = 1000, chunk_overlap: int = 200) -> str:
        document_data = self.xml_processor.extract_text_and_metadata(xml_content, filename)
        return self.vector_store.add_document(document_data, collection_name)

    def add_json_document(self, json_content: str, filename: str = None, collection_name: str = None, chunk_size: int = 1000, chunk_overlap: int = 200) -> str:
        # Update chunk configuration for this upload
        self.json_processor.update_chunk_config(chunk_size, chunk_overlap)
        document_data = self.json_processor.extract_text_and_metadata(json_content, filename)
        return self.vector_store.add_document(document_data, collection_name)

    def add_text_document(self, text_content: str, filename: str = None, collection_name: str = None, chunk_size: int = 1000, chunk_overlap: int = 200) -> str:
        # Update chunk configuration for this upload
        self.text_processor.update_chunk_config(chunk_size, chunk_overlap)
        document_data = self.text_processor.extract_text_and_metadata(text_content, filename)
        return self.vector_store.add_document(document_data, collection_name)

    def add_xml_file(self, file_path: str, collection_name: str = None, chunk_size: int = 1000, chunk_overlap: int = 200) -> str:
        document_data = self.xml_processor.process_file(file_path)
        return self.vector_store.add_document(document_data, collection_name)

    def add_json_file(self, file_path: str, collection_name: str = None, chunk_size: int = 1000, chunk_overlap: int = 200) -> str:
        # Update chunk configuration for this upload
        self.json_processor.update_chunk_config(chunk_size, chunk_overlap)
        document_data = self.json_processor.process_file(file_path)
        return self.vector_store.add_document(document_data, collection_name)

    def add_text_file(self, file_path: str, collection_name: str = None, chunk_size: int = 1000, chunk_overlap: int = 200) -> str:
        # Update chunk configuration for this upload
        self.text_processor.update_chunk_config(chunk_size, chunk_overlap)
        document_data = self.text_processor.process_file(file_path)
        return self.vector_store.add_document(document_data, collection_name)

    def search_documents(self, query: str, n_results: int = 5, where: Optional[Dict] = None, collection_name: str = None) -> List[Dict[str, Any]]:
        return self.vector_store.search(query, n_results, where, collection_name)

    def generate_answer(self, query: str, context_results: List[Dict[str, Any]], max_length: int = 512) -> str:
        context_text = "\n\n".join([
            f"Source: {result['metadata'].get('filename', 'unknown')} (chunk {result['metadata'].get('chunk_index', 0)})\n"
            f"Content: {result['document']}"
            for result in context_results
        ])

        system_prompt = """You are a helpful assistant that answers questions based on document content (XML, JSON, and text files).
Use only the provided context to answer questions. If the context doesn't contain enough information
to answer the question, say so clearly. Always cite which source document your answer comes from."""

        # Format the prompt for the model
        prompt = f"""System: {system_prompt}

Context from documents:
{context_text}

Question: {query}

Answer:"""

        try:
            # Tokenize the input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = inputs.to(self.device)

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_length,
                    num_return_sequences=1,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs)
                )

            # Decode the response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the answer part (after "Answer:")
            if "Answer:" in generated_text:
                answer = generated_text.split("Answer:")[-1].strip()
            else:
                answer = generated_text[len(prompt):].strip()

            # Clean up the answer
            if not answer or len(answer) < 10:
                return "I couldn't generate a proper answer based on the provided context. The context might not contain enough relevant information to answer your question."

            return answer

        except Exception as e:
            return f"Error generating response with HuggingFace model: {str(e)}"

    def query(self, question: str, n_results: int = 5, where: Optional[Dict] = None, max_length: int = 512, collection_name: str = None) -> Dict[str, Any]:
        search_results = self.search_documents(question, n_results, where, collection_name)

        if not search_results:
            return {
                "answer": "No relevant documents found for your question.",
                "sources": [],
                "query": question,
                "model_used": f"HuggingFace: {self.model_name}"
            }

        answer = self.generate_answer(question, search_results, max_length)

        return {
            "answer": answer,
            "sources": [
                {
                    "filename": result["metadata"].get("filename", "unknown"),
                    "document_type": result["metadata"].get("document_type", "unknown"),
                    "chunk_index": result["metadata"].get("chunk_index", 0),
                    "distance": result["distance"],
                    "relevance_score": round(1 - (result["distance"] / 2), 3),
                    "content_preview": result["document"][:200] + "..." if len(result["document"]) > 200 else result["document"],
                    "full_content": result["document"],
                    "content": result["document"],
                    "document_id": result["metadata"].get("document_id", ""),
                    "metadata": result["metadata"]
                }
                for result in search_results
            ],
            "query": question,
            "retrieved_chunks": search_results,
            "model_used": f"HuggingFace: {self.model_name}"
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

    def create_collection(self, collection_name: str, embedding_model: str = None) -> bool:
        return self.vector_store.create_collection(collection_name, embedding_model)

    def get_available_embedding_models(self) -> Dict[str, Dict[str, str]]:
        return self.vector_store.get_available_embedding_models()

    def get_collection_info(self, collection_name: str = None) -> Dict[str, Any]:
        return self.vector_store.get_collection_info(collection_name)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "model_type": "HuggingFace Transformers"
        }