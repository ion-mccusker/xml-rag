import json
from typing import Dict, List, Any, Optional, Union
import re
from pathlib import Path


class JSONProcessor:
    def __init__(self):
        pass

    def extract_text_and_metadata(self, json_content: str, filename: str = None) -> Dict[str, Any]:
        try:
            data = json.loads(json_content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON content: {str(e)}")

        # Check if this matches the specific format: {"title": "...", "content": "...", "metadata": {...}}
        if self._is_title_content_format(data):
            return self._process_title_content_format(data, filename)

        # Use original processing for other JSON formats
        metadata = self._extract_metadata(data, filename)
        text_chunks = self._extract_text_chunks(data)

        return {
            "metadata": metadata,
            "text_chunks": text_chunks,
            "full_text": " ".join(text_chunks),
            "document_type": "json"
        }

    def _extract_metadata(self, data: Any, filename: str = None) -> Dict[str, Any]:
        metadata = {
            "filename": filename or "unknown",
            "document_type": "json",
            "data_type": type(data).__name__,
            "total_keys": self._count_keys(data) if isinstance(data, dict) else 0,
            "total_items": len(data) if isinstance(data, (list, dict)) else 0,
            "max_depth": self._get_max_depth(data),
            "structure_summary": self._get_structure_summary(data)
        }

        if isinstance(data, dict):
            common_meta_fields = ['title', 'name', 'description', 'author', 'date', 'created', 'modified', 'version', 'id']
            for field in common_meta_fields:
                if field in data and isinstance(data[field], (str, int, float)):
                    metadata[field] = str(data[field])

        return metadata

    def _extract_text_chunks(self, data: Any, path: str = "", chunks: List[str] = None) -> List[str]:
        if chunks is None:
            chunks = []

        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key

                if isinstance(value, str) and len(value.strip()) > 10:
                    clean_text = self._clean_text(value.strip())
                    if clean_text:
                        chunk_with_context = f"{key}: {clean_text}"
                        chunks.append(chunk_with_context)
                elif isinstance(value, (dict, list)):
                    self._extract_text_chunks(value, current_path, chunks)
                elif isinstance(value, (int, float, bool)) and value is not None:
                    chunks.append(f"{key}: {str(value)}")

        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_path = f"{path}[{i}]" if path else f"[{i}]"

                if isinstance(item, str) and len(item.strip()) > 10:
                    clean_text = self._clean_text(item.strip())
                    if clean_text:
                        chunks.append(f"Item {i}: {clean_text}")
                elif isinstance(item, (dict, list)):
                    self._extract_text_chunks(item, current_path, chunks)
                elif isinstance(item, (int, float, bool)) and item is not None:
                    chunks.append(f"Item {i}: {str(item)}")

        elif isinstance(data, str) and len(data.strip()) > 10:
            clean_text = self._clean_text(data.strip())
            if clean_text:
                chunks.append(clean_text)

        return chunks

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def _count_keys(self, data: Any) -> int:
        if isinstance(data, dict):
            count = len(data)
            for value in data.values():
                if isinstance(value, (dict, list)):
                    count += self._count_keys(value)
            return count
        elif isinstance(data, list):
            count = 0
            for item in data:
                if isinstance(item, (dict, list)):
                    count += self._count_keys(item)
            return count
        return 0

    def _get_max_depth(self, data: Any, current_depth: int = 0) -> int:
        if isinstance(data, dict):
            if not data:
                return current_depth
            return max(self._get_max_depth(value, current_depth + 1) for value in data.values())
        elif isinstance(data, list):
            if not data:
                return current_depth
            return max(self._get_max_depth(item, current_depth + 1) for item in data)
        else:
            return current_depth

    def _get_structure_summary(self, data: Any) -> str:
        if isinstance(data, dict):
            if not data:
                return "empty_object"

            key_types = {}
            for key, value in data.items():
                value_type = type(value).__name__
                if value_type in key_types:
                    key_types[value_type] += 1
                else:
                    key_types[value_type] = 1

            summary_parts = [f"{count}_{type_name}" for type_name, count in key_types.items()]
            return f"object({','.join(summary_parts)})"

        elif isinstance(data, list):
            if not data:
                return "empty_array"

            item_types = {}
            for item in data:
                item_type = type(item).__name__
                if item_type in item_types:
                    item_types[item_type] += 1
                else:
                    item_types[item_type] = 1

            summary_parts = [f"{count}_{type_name}" for type_name, count in item_types.items()]
            return f"array({','.join(summary_parts)})"
        else:
            return type(data).__name__

    def process_file(self, file_path: str) -> Dict[str, Any]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return self.extract_text_and_metadata(content, path.name)

    def _is_title_content_format(self, data: Any) -> bool:
        """Check if the JSON matches the specific format: {"title": "...", "content": "...", "metadata": {...}}"""
        if not isinstance(data, dict):
            return False

        required_keys = {"title", "content"}
        has_required = required_keys.issubset(data.keys())

        # Check that title and content are strings
        if has_required:
            title_is_string = isinstance(data.get("title"), str)
            content_is_string = isinstance(data.get("content"), str)
            metadata_is_dict = isinstance(data.get("metadata", {}), dict)

            return title_is_string and content_is_string and metadata_is_dict

        return False

    def _process_title_content_format(self, data: Dict[str, Any], filename: str = None) -> Dict[str, Any]:
        """Process JSON documents with the specific title/content/metadata format"""
        title = data.get("title", "").strip()
        content = data.get("content", "").strip()
        doc_metadata = data.get("metadata", {})

        # Build metadata combining document info with nested metadata
        metadata = {
            "filename": filename or "unknown",
            "document_type": "json",
            "format_type": "title_content_metadata",
            "title": title,
            "content_length": len(content),
            "has_nested_metadata": bool(doc_metadata)
        }

        # Add nested metadata fields (convert complex types to strings)
        for key, value in doc_metadata.items():
            if isinstance(value, (str, int, float, bool)) and value is not None:
                metadata[f"meta_{key}"] = str(value)
            elif isinstance(value, (list, dict)):
                metadata[f"meta_{key}"] = str(value)

        # Create text chunks from the content
        text_chunks = []

        # Add title as first chunk if present
        if title:
            text_chunks.append(f"Title: {title}")

        # Process the main content
        if content:
            # Clean the content
            clean_content = self._clean_text(content)
            if clean_content:
                # Split content into chunks if it's very long
                if len(clean_content) > 1000:
                    # Split into sentences and group them
                    sentences = re.split(r'[.!?]+\s+', clean_content)
                    current_chunk = ""

                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) < 800:
                            current_chunk += sentence + ". "
                        else:
                            if current_chunk.strip():
                                text_chunks.append(f"Content: {current_chunk.strip()}")
                            current_chunk = sentence + ". "

                    # Add remaining content
                    if current_chunk.strip():
                        text_chunks.append(f"Content: {current_chunk.strip()}")
                else:
                    text_chunks.append(f"Content: {clean_content}")

        # Add metadata as searchable content if it contains meaningful text
        for key, value in doc_metadata.items():
            if isinstance(value, str) and len(value.strip()) > 10:
                clean_meta_text = self._clean_text(value.strip())
                if clean_meta_text:
                    text_chunks.append(f"{key}: {clean_meta_text}")

        # Ensure we have at least some content
        if not text_chunks:
            text_chunks = [f"Document: {title}" if title else "Empty document"]

        return {
            "metadata": metadata,
            "text_chunks": text_chunks,
            "full_text": " ".join(text_chunks),
            "document_type": "json"
        }