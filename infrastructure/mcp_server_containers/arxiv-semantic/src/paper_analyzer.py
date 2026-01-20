"""
Paper analyzer module for deep analysis of ArXiv papers.
Extracts structured information including sections, figures, equations, and references.
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import arxiv
import pymupdf4llm
import fitz  # PyMuPDF
from PIL import Image
import io

logger = logging.getLogger(__name__)


class PaperAnalyzer:
    """Analyzes ArXiv papers to extract structured information."""

    def __init__(self, papers_dir: str = None):
        """
        Initialize the paper analyzer.

        Args:
            papers_dir: Directory to store downloaded papers
        """
        self.papers_dir = papers_dir or os.environ.get('PAPERS_DIR', '/workspace/papers')
        os.makedirs(self.papers_dir, exist_ok=True)

    def download_paper(self, paper_id: str) -> Tuple[arxiv.Result, str]:
        """
        Download a paper from ArXiv.

        Args:
            paper_id: ArXiv paper ID

        Returns:
            Tuple of (paper metadata, pdf path)
        """
        try:
            # Search for the paper
            search = arxiv.Search(id_list=[paper_id])
            paper = next(search.results())

            # Download PDF
            pdf_path = os.path.join(self.papers_dir, f"{paper_id.replace('/', '_')}.pdf")
            if not os.path.exists(pdf_path):
                paper.download_pdf(dirpath=self.papers_dir, filename=f"{paper_id.replace('/', '_')}.pdf")
                logger.info(f"Downloaded paper {paper_id} to {pdf_path}")
            else:
                logger.info(f"Paper {paper_id} already exists at {pdf_path}")

            return paper, pdf_path

        except Exception as e:
            logger.error(f"Failed to download paper {paper_id}: {e}")
            raise

    def extract_full_text(self, pdf_path: str) -> str:
        """
        Extract full text from PDF using pymupdf4llm.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text in markdown format
        """
        try:
            md_text = pymupdf4llm.to_markdown(pdf_path)
            return md_text
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}")
            # Fallback to basic extraction
            return self._basic_text_extraction(pdf_path)

    def _basic_text_extraction(self, pdf_path: str) -> str:
        """
        Basic text extraction fallback using PyMuPDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text
        """
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            logger.error(f"Basic text extraction failed: {e}")
            return ""

    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extract paper sections from text.

        Args:
            text: Full paper text

        Returns:
            Dictionary of section titles and content
        """
        sections = {}
        current_section = "Introduction"
        current_content = []

        # Common section patterns
        section_patterns = [
            r'^#+\s+(.+)$',  # Markdown headers
            r'^(\d+\.?\s+[A-Z].+)$',  # Numbered sections
            r'^([A-Z][A-Z\s]+)$',  # All caps sections
            r'^(Abstract|Introduction|Related Work|Methodology|Methods|Results|Discussion|Conclusion|References).*$'
        ]

        lines = text.split('\n')
        for line in lines:
            is_section = False
            for pattern in section_patterns:
                match = re.match(pattern, line.strip())
                if match:
                    # Save previous section
                    if current_content:
                        sections[current_section] = '\n'.join(current_content).strip()
                    current_section = match.group(1).strip()
                    current_content = []
                    is_section = True
                    break

            if not is_section:
                current_content.append(line)

        # Save last section
        if current_content:
            sections[current_section] = '\n'.join(current_content).strip()

        return sections

    def extract_figures(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract figures and their captions from PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of figure information
        """
        figures = []
        try:
            doc = fitz.open(pdf_path)

            for page_num, page in enumerate(doc):
                # Extract images
                image_list = page.get_images(full=True)

                for img_index, img in enumerate(image_list):
                    try:
                        # Extract image
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)

                        # Convert to PIL Image
                        img_data = pix.tobytes("png")
                        image = Image.open(io.BytesIO(img_data))

                        # Try to find caption (simplified approach)
                        text = page.get_text()
                        caption = self._extract_figure_caption(text, img_index + 1)

                        # Save image
                        img_filename = f"fig_{page_num}_{img_index}.png"
                        img_path = os.path.join(self.papers_dir, img_filename)
                        image.save(img_path)

                        figures.append({
                            'page': page_num + 1,
                            'index': img_index,
                            'path': img_path,
                            'caption': caption,
                            'width': image.width,
                            'height': image.height
                        })

                        pix = None

                    except Exception as e:
                        logger.warning(f"Failed to extract image {img_index} from page {page_num}: {e}")

            doc.close()

        except Exception as e:
            logger.error(f"Failed to extract figures from {pdf_path}: {e}")

        return figures

    def _extract_figure_caption(self, text: str, fig_num: int) -> str:
        """
        Extract figure caption from text.

        Args:
            text: Page text
            fig_num: Figure number

        Returns:
            Extracted caption or empty string
        """
        # Look for common caption patterns
        patterns = [
            rf'Figure\s+{fig_num}[:\.]?\s*([^\n]+(?:\n[^\n]+)*?)(?=\n\n|\nFigure|\n\d+\.|\Z)',
            rf'Fig\.\s+{fig_num}[:\.]?\s*([^\n]+(?:\n[^\n]+)*?)(?=\n\n|\nFig\.|\n\d+\.|\Z)',
            rf'FIG\.\s+{fig_num}[:\.]?\s*([^\n]+(?:\n[^\n]+)*?)(?=\n\n|\nFIG\.|\n\d+\.|\Z)'
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()

        return ""

    def extract_equations(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract equations from text.

        Args:
            text: Full paper text

        Returns:
            List of equations with context
        """
        equations = []

        # LaTeX equation patterns
        patterns = [
            (r'\$\$(.+?)\$\$', 'display'),  # Display math
            (r'\\\[(.+?)\\\]', 'display'),  # Display math alternative
            (r'\\begin\{equation\}(.+?)\\end\{equation\}', 'display'),  # Equation environment
            (r'\\begin\{align\}(.+?)\\end\{align\}', 'display'),  # Align environment
            (r'\$(.+?)\$', 'inline')  # Inline math
        ]

        for pattern, eq_type in patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            for match in matches:
                equation = match.group(1).strip()
                # Skip very short inline equations (likely variables)
                if eq_type == 'inline' and len(equation) < 3:
                    continue

                # Get surrounding context
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = text[start:end]

                equations.append({
                    'equation': equation,
                    'type': eq_type,
                    'context': context,
                    'position': match.start()
                })

        # Sort by position
        equations.sort(key=lambda x: x['position'])
        return equations

    def extract_references(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract references from paper.

        Args:
            text: Full paper text

        Returns:
            List of references
        """
        references = []

        # Find references section
        ref_section = None
        section_patterns = [
            r'References\n(.+)',
            r'REFERENCES\n(.+)',
            r'Bibliography\n(.+)',
            r'\\section\*?\{References\}(.+)'
        ]

        for pattern in section_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                ref_section = match.group(1)
                break

        if not ref_section:
            return references

        # Extract individual references
        # Simple pattern for numbered references
        ref_pattern = r'\[(\d+)\]\s*([^\[\]]+?)(?=\[\d+\]|\Z)'
        matches = re.finditer(ref_pattern, ref_section)

        for match in matches:
            ref_num = match.group(1)
            ref_text = match.group(2).strip()

            # Try to extract title, authors, year
            title_match = re.search(r'"([^"]+)"', ref_text)
            year_match = re.search(r'\((\d{4})\)', ref_text) or re.search(r'(\d{4})', ref_text)

            references.append({
                'number': ref_num,
                'text': ref_text,
                'title': title_match.group(1) if title_match else None,
                'year': year_match.group(1) if year_match else None
            })

        return references

    def extract_key_findings(self, sections: Dict[str, str]) -> List[str]:
        """
        Extract key findings from paper sections.

        Args:
            sections: Dictionary of paper sections

        Returns:
            List of key findings
        """
        findings = []

        # Look for findings in specific sections
        relevant_sections = ['Conclusion', 'Results', 'Discussion', 'Summary', 'Abstract']

        for section_name in relevant_sections:
            for key, content in sections.items():
                if section_name.lower() in key.lower():
                    # Extract sentences that might be findings
                    sentences = content.split('.')
                    for sentence in sentences:
                        sentence = sentence.strip()
                        # Look for finding indicators
                        indicators = ['we found', 'we show', 'demonstrates', 'reveals',
                                     'indicates', 'suggests', 'achieve', 'improve',
                                     'outperform', 'significant']
                        if any(ind in sentence.lower() for ind in indicators):
                            if len(sentence) > 20:  # Filter out very short sentences
                                findings.append(sentence + '.')

        return findings[:10]  # Return top 10 findings

    def analyze_pdf(
        self,
        paper_id: str,
        pdf_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze a PDF that is already available locally."""

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Extract text and derived artifacts
        logger.info(f"Extracting text from {paper_id}")
        full_text = self.extract_full_text(pdf_path)

        logger.info(f"Extracting sections from {paper_id}")
        sections = self.extract_sections(full_text)

        logger.info(f"Extracting figures from {paper_id}")
        figures = self.extract_figures(pdf_path)

        logger.info(f"Extracting equations from {paper_id}")
        equations = self.extract_equations(full_text)

        logger.info(f"Extracting references from {paper_id}")
        references = self.extract_references(full_text)

        logger.info(f"Extracting key findings from {paper_id}")
        key_findings = self.extract_key_findings(sections)

        meta = metadata or {}

        analysis = {
            'paper_id': paper_id,
            'title': meta.get('title'),
            'authors': meta.get('authors', []),
            'abstract': meta.get('abstract'),
            'published': meta.get('published'),
            'categories': meta.get('categories', []),
            'pdf_url': meta.get('pdf_url'),
            'pdf_path': pdf_path,
            'full_text': full_text,
            'sections': sections,
            'figures': figures,
            'equations': equations,
            'references': references,
            'key_findings': key_findings,
            'statistics': {
                'num_sections': len(sections),
                'num_figures': len(figures),
                'num_equations': len(equations),
                'num_references': len(references),
                'text_length': len(full_text)
            }
        }

        return analysis

    def analyze_paper(self, paper_id: str) -> Dict[str, Any]:
        """
        Perform complete analysis of a paper.

        Args:
            paper_id: ArXiv paper ID

        Returns:
            Complete analysis results
        """
        try:
            paper, pdf_path = self.download_paper(paper_id)

            metadata = {
                'title': paper.title,
                'authors': [author.name for author in paper.authors],
                'abstract': paper.summary,
                'published': paper.published.isoformat(),
                'categories': paper.categories,
                'pdf_url': paper.pdf_url
            }

            analysis = self.analyze_pdf(paper_id, pdf_path, metadata)
            analysis['pdf_url'] = paper.pdf_url

            logger.info(f"Completed analysis of paper {paper_id}")
            return analysis

        except Exception as e:
            logger.error(f"Failed to analyze paper {paper_id}: {e}")
            raise
