import xml.etree.ElementTree as ET
from lxml import etree
from typing import Dict, List, Any, Optional
import re
from pathlib import Path


class XMLProcessor:
    def __init__(self):
        pass

    def extract_text_and_metadata(self, xml_content: str, filename: str = None) -> Dict[str, Any]:
        try:
            root = etree.fromstring(xml_content.encode('utf-8'))
        except Exception as e:
            raise ValueError(f"Invalid XML content: {str(e)}")

        metadata = self._extract_metadata(root, filename)
        text_chunks = self._extract_text_chunks(root)

        return {
            "metadata": metadata,
            "text_chunks": text_chunks,
            "full_text": " ".join(text_chunks)
        }

    def _extract_metadata(self, root: etree.Element, filename: str = None) -> Dict[str, Any]:
        metadata = {
            "filename": filename or "unknown",
            "root_tag": root.tag,
            "namespace": root.nsmap.get(None, ""),
            "attributes": dict(root.attrib),
            "element_count": len(list(root.iter())),
            "depth": self._get_max_depth(root)
        }

        common_meta_tags = ['title', 'author', 'date', 'description', 'subject', 'creator']
        for tag in common_meta_tags:
            elements = root.xpath(f".//{tag}", namespaces=root.nsmap)
            if elements:
                metadata[tag] = elements[0].text

        return metadata

    def _extract_text_chunks(self, root: etree.Element) -> List[str]:
        chunks = []

        for elem in root.iter():
            if elem.text and elem.text.strip():
                clean_text = self._clean_text(elem.text.strip())
                if clean_text and len(clean_text) > 10:
                    chunks.append(clean_text)

            if elem.tail and elem.tail.strip():
                clean_text = self._clean_text(elem.tail.strip())
                if clean_text and len(clean_text) > 10:
                    chunks.append(clean_text)

        return chunks

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def _get_max_depth(self, element: etree.Element, depth: int = 0) -> int:
        if len(element) == 0:
            return depth
        return max(self._get_max_depth(child, depth + 1) for child in element)

    def process_file(self, file_path: str) -> Dict[str, Any]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return self.extract_text_and_metadata(content, path.name)