"""Text processing utilities."""
import re
from typing import List, Dict, Any
from PyPDF2 import PdfReader

class TextProcessor:
    """Process and split text documents."""
    
    def detect_section_headers(self, text: str) -> List[tuple]:
        """
        Detect potential section headers in text.
        Returns a list of (start_pos, header_text) tuples.
        """
        patterns = [
            r'^Chapter\s+\d+[\.\:]\s+.+$',  # Chapter headers
            r'^\d+\.\d+\s+.+$',             # Section headers like 1.2 Topic
            r'^Section\s+\d+[\.\:]\s+.+$',  # Section headers
            r'^[A-Z][A-Z\s]+$',             # ALL CAPS headers
        ]
        
        headers = []
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.MULTILINE):
                headers.append((match.start(), match.group()))
        
        return sorted(headers)
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[int, str]:
        """Extract text content from a PDF file, page by page."""
        try:
            reader = PdfReader(pdf_path)
            pages = {}
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    pages[i] = page_text
            return pages
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return {}
    
    def split_by_sections(self, pages: Dict[int, str]) -> List[Dict[str, Any]]:
        """
        Split text into sections based on detected headers.
        Returns a list of section dictionaries with header, content, and page info.
        """
        sections = []
        current_section = {"header": "Introduction", "content": "", "start_page": 0}
        
        for page_num, page_text in sorted(pages.items()):
            # If this is the first page, set it as the start page
            if not sections and not current_section["content"]:
                current_section["start_page"] = page_num
            
            headers = self.detect_section_headers(page_text)
            
            if not headers:
                # No headers on this page, add to current section
                current_section["content"] += f"\n\n{page_text}"
            else:
                # For each header on the page
                last_pos = 0
                for pos, header_text in headers:
                    # Add content before this header to current section
                    if pos > 0:
                        current_section["content"] += f"\n\n{page_text[last_pos:pos]}"
                    
                    # Save current section if it has content
                    if current_section["content"].strip():
                        sections.append(current_section)
                    
                    # Start a new section
                    current_section = {
                        "header": header_text,
                        "content": "",
                        "start_page": page_num
                    }
                    
                    last_pos = pos + len(header_text)
                
                # Add remaining content after the last header
                if last_pos < len(page_text):
                    current_section["content"] += page_text[last_pos:]
        
        # Add the final section if it has content
        if current_section["content"].strip():
            sections.append(current_section)
        
        return sections
    
    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a document and split it into sections."""
        if file_path.lower().endswith('.pdf'):
            pages = self.extract_text_from_pdf(file_path)
            return self.split_by_sections(pages)
        else:
            # For now, just handle PDFs
            raise ValueError("Only PDF files are supported at the moment")
