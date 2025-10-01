import re
from typing import Dict, List, Any, Optional
from pathlib import Path
import hashlib
from langchain.text_splitter import RecursiveCharacterTextSplitter


class TextProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def extract_text_and_metadata(self, text_content: str, filename: str = None) -> Dict[str, Any]:
        if not text_content or not text_content.strip():
            raise ValueError("Empty text content provided")

        metadata = self._extract_metadata(text_content, filename)
        text_chunks = self._create_chunks(text_content)

        return {
            "metadata": metadata,
            "text_chunks": text_chunks,
            "full_text": text_content,
            "full_content": text_content,  # Store original content for filesystem
            "document_type": "text"
        }

    def _extract_metadata(self, content: str, filename: str = None) -> Dict[str, Any]:
        print("Extracting metadata")
        lines = content.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]

        metadata = {
            "filename": filename or "unknown",
            "document_type": "text",
            "total_lines": len(lines),
            "non_empty_lines": len(non_empty_lines),
            "total_characters": len(content),
            "total_words": len(content.split()),
            "paragraphs": self._count_paragraphs(content),
            "content_hash": hashlib.md5(content.encode()).hexdigest()[:8]
        }

        language_hints = self._detect_language_hints(content)
        if language_hints:
            metadata["language_hints"] = ", ".join(language_hints)

        structure_info = self._analyze_structure(content)
        metadata.update(structure_info)

        title = self._extract_title(content, filename)
        if title:
            metadata["title"] = title

        return metadata

    def _create_chunks(self, content: str) -> List[str]:
        """Use LangChain's RecursiveCharacterTextSplitter for better chunking"""
        if not content.strip():
            return []

        chunks = self.text_splitter.split_text(content)
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def update_chunk_config(self, chunk_size: int, chunk_overlap: int):
        """Update chunk configuration and recreate text splitter"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def _split_into_paragraphs(self, content: str) -> List[str]:
        paragraphs = re.split(r'\n\s*\n', content)
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_large_paragraph(self, paragraph: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                if len(sentence) <= self.chunk_size:
                    current_chunk = sentence
                else:
                    word_chunks = self._split_by_words(sentence)
                    chunks.extend(word_chunks[:-1])
                    current_chunk = word_chunks[-1] if word_chunks else ""

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _split_by_words(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        current_chunk = ""

        for word in words:
            if len(current_chunk) + len(word) + 1 <= self.chunk_size:
                if current_chunk:
                    current_chunk += " " + word
                else:
                    current_chunk = word
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = word

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        if len(chunks) <= 1 or self.chunk_overlap <= 0:
            return chunks

        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                prev_chunk = chunks[i-1]
                overlap_text = self._get_overlap_text(prev_chunk, self.chunk_overlap)
                if overlap_text:
                    overlapped_chunk = overlap_text + "\n\n" + chunk
                    overlapped_chunks.append(overlapped_chunk)
                else:
                    overlapped_chunks.append(chunk)

        return overlapped_chunks

    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        if len(text) <= overlap_size:
            return text

        words = text.split()
        if len(words) <= 10:
            return text

        overlap_words = words[-10:]
        return " ".join(overlap_words)

    def _count_paragraphs(self, content: str) -> int:
        paragraphs = re.split(r'\n\s*\n', content)
        return len([p for p in paragraphs if p.strip()])

    def _detect_language_hints(self, content: str) -> List[str]:
        hints = []

        sample = content[:1000].lower()

        common_patterns = {
            "code": [r'def\s+\w+\(', r'function\s+\w+\(', r'class\s+\w+', r'import\s+\w+', r'#include\s*<'],
            "email": [r'\w+@\w+\.\w+'],
            "urls": [r'https?://\S+'],
            "structured": [r'^\s*[-*+]\s+', r'^\s*\d+\.\s+'],
            "technical": [r'\b(?:API|JSON|XML|HTTP|SQL|HTML)\b']
        }

        for hint_type, patterns in common_patterns.items():
            for pattern in patterns:
                if re.search(pattern, sample, re.MULTILINE | re.IGNORECASE):
                    hints.append(hint_type)
                    break

        return list(set(hints))

    def _analyze_structure(self, content: str) -> Dict[str, Any]:
        lines = content.split('\n')

        header_lines = []
        for line in lines[:10]:
            line = line.strip()
            if line and (line.isupper() or line.startswith('#') or len(line.split()) <= 8):
                header_lines.append(line)

        bullet_count = len(re.findall(r'^\s*[-*+•]\s+', content, re.MULTILINE))
        numbered_count = len(re.findall(r'^\s*\d+\.\s+', content, re.MULTILINE))

        return {
            "potential_headers": " | ".join(header_lines[:3]) if header_lines else "",
            "bullet_points": bullet_count,
            "numbered_items": numbered_count,
            "has_structure": bullet_count > 0 or numbered_count > 0 or len(header_lines) > 0
        }

    def _extract_title(self, content: str, filename: str = None) -> Optional[str]:
        lines = content.split('\n')
        for line in lines[:5]:
            line = line.strip()
            if line and len(line.split()) <= 12 and len(line) <= 100:
                if not line.lower().startswith(('the', 'this', 'a ', 'an ')) and '.' not in line:
                    return line

        if filename:
            name_without_ext = Path(filename).stem
            if len(name_without_ext.split()) <= 8:
                return name_without_ext.replace('_', ' ').replace('-', ' ').title()

        return None

    def process_file(self, file_path: str) -> Dict[str, Any]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
            except Exception as e:
                raise ValueError(f"Unable to read file with supported encodings: {str(e)}")

        return self.extract_text_and_metadata(content, path.name)