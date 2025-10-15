import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
from document_retriever import DocumentRetriever

load_dotenv()


class HuggingFaceRAGPipeline:
    def __init__(self, persist_directory: str = "./chroma_db", model_name: str = "microsoft/DialoGPT-medium"):
        self.retriever = DocumentRetriever(persist_directory)

        # Initialize HuggingFace pipeline
        self.model_name = model_name
        self.device = 0 if torch.cuda.is_available() else -1  # 0 for GPU, -1 for CPU

        try:
            print(f"Loading text generation pipeline with model {model_name}...")

            # Load tokenizer and model separately with safetensors enabled
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_safetensors=True,
                trust_remote_code=True
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                use_safetensors=True,
                device_map="auto",
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True
            )

            # Create text generation pipeline with pre-loaded model and tokenizer
            self.generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                return_full_text=False,  # Only return generated text, not the input
                pad_token_id=50256  # Common pad token ID, will be overridden if tokenizer has one
            )

            # Set pad token if tokenizer doesn't have one
            if self.generator.tokenizer.pad_token is None:
                self.generator.tokenizer.pad_token = self.generator.tokenizer.eos_token

            print(f"Successfully loaded {model_name}")

        except Exception as e:
            print(f"Error loading {model_name}: {str(e)}")
            print("Falling back to microsoft/DialoGPT-medium")

            # Fallback to a smaller, more reliable model
            self.model_name = "microsoft/DialoGPT-medium"

            # Load tokenizer and model separately with safetensors
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_safetensors=True
            )

            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                use_safetensors=True,
                device_map="auto"
            )

            self.generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                return_full_text=False
            )

            if self.generator.tokenizer.pad_token is None:
                self.generator.tokenizer.pad_token = self.generator.tokenizer.eos_token

    def add_xml_document(self, xml_content: str, filename: str = None, collection_name: str = None, chunk_size: int = 1000, chunk_overlap: int = 200) -> str:
        return self.retriever.add_xml_document(xml_content, filename, collection_name, chunk_size, chunk_overlap)

    def add_json_document(self, json_content: str, filename: str = None, collection_name: str = None, chunk_size: int = 1000, chunk_overlap: int = 200) -> str:
        return self.retriever.add_json_document(json_content, filename, collection_name, chunk_size, chunk_overlap)

    def add_text_document(self, text_content: str, filename: str = None, collection_name: str = None, chunk_size: int = 1000, chunk_overlap: int = 200) -> str:
        return self.retriever.add_text_document(text_content, filename, collection_name, chunk_size, chunk_overlap)

    def add_xml_file(self, file_path: str, collection_name: str = None, chunk_size: int = 1000, chunk_overlap: int = 200) -> str:
        return self.retriever.add_xml_file(file_path, collection_name, chunk_size, chunk_overlap)

    def add_json_file(self, file_path: str, collection_name: str = None, chunk_size: int = 1000, chunk_overlap: int = 200) -> str:
        return self.retriever.add_json_file(file_path, collection_name, chunk_size, chunk_overlap)

    def add_text_file(self, file_path: str, collection_name: str = None, chunk_size: int = 1000, chunk_overlap: int = 200) -> str:
        return self.retriever.add_text_file(file_path, collection_name, chunk_size, chunk_overlap)

    def search_documents(self, query: str, n_results: int = 5, where: Optional[Dict] = None, collection_name: str = None, use_reranker: bool = False) -> List[Dict[str, Any]]:
        return self.retriever.search_documents(query, n_results, where, collection_name, use_reranker)

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
            print(prompt)
            # Use the pipeline for text generation
            generated_outputs = self.generator(
                prompt,
                max_new_tokens=max_length,
                temperature=0.1,
                do_sample=True,
                truncation=True,
                pad_token_id=self.generator.tokenizer.eos_token_id,
                eos_token_id=self.generator.tokenizer.eos_token_id,
                num_return_sequences=1
            )

            # Extract the generated text
            if generated_outputs and len(generated_outputs) > 0:
                answer = generated_outputs[0]['generated_text'].strip()
            else:
                answer = ""

            # Clean up the answer
            if not answer or len(answer) < 10:
                return "I couldn't generate a proper answer based on the provided context. The context might not contain enough relevant information to answer your question."

            return answer

        except Exception as e:
            return f"Error generating response with HuggingFace pipeline: {str(e)}"

    def query(self, question: str, n_results: int = 5, where: Optional[Dict] = None, max_length: int = 512, collection_name: str = None, use_reranker: bool = False) -> Dict[str, Any]:
        search_results = self.search_documents(question, n_results, where, collection_name, use_reranker)

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
            "sources": self.retriever._format_sources(search_results, collection_name),
            "query": question,
            "retrieved_chunks": search_results,
            "model_used": f"HuggingFace: {self.model_name}",
            "reranker_used": use_reranker
        }

    def delete_document(self, document_id: str, collection_name: str = None):
        self.retriever.delete_document(document_id, collection_name)

    def list_documents(self, collection_name: str = None, page: int = 1, per_page: int = 10,
                      search: str = None, document_type: str = None) -> Dict[str, Any]:
        return self.retriever.list_documents(collection_name, page, per_page, search, document_type)

    def get_document_count(self, collection_name: str = None) -> int:
        return self.retriever.get_document_count(collection_name)

    def get_document_content(self, document_id: str, collection_name: str = None):
        return self.retriever.get_document_content(document_id, collection_name)

    def list_collections(self) -> List[str]:
        return self.retriever.list_collections()

    def create_collection(self, collection_name: str, embedding_model: str = None, chunk_size: int = 1000, chunk_overlap: int = 200) -> bool:
        return self.retriever.create_collection(collection_name, embedding_model, chunk_size, chunk_overlap)

    def delete_collection(self, collection_name: str) -> bool:
        return self.retriever.delete_collection(collection_name)

    def get_available_embedding_models(self) -> Dict[str, Dict[str, str]]:
        return self.retriever.get_available_embedding_models()

    def get_collection_info(self, collection_name: str = None) -> Dict[str, Any]:
        return self.retriever.get_collection_info(collection_name)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": "GPU" if self.device == 0 else "CPU",
            "model_type": "HuggingFace Text Generation Pipeline"
        }