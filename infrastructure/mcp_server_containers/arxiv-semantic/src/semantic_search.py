"""
Semantic search module using ChromaDB for ArXiv papers.
Inspired by Zotero MCP's semantic capabilities.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np
from embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


class SemanticSearchEngine:
    """Manages semantic search using ChromaDB for ArXiv papers."""

    def __init__(self, db_path: str = None, embedding_manager: EmbeddingManager = None):
        """
        Initialize the semantic search engine.

        Args:
            db_path: Path to ChromaDB database
        """
        self.db_path = db_path or os.environ.get('CHROMA_DB_PATH', '/workspace/chromadb')
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.client = None
        self.collection = None
        self._initialize_db()

    def _initialize_db(self):
        """Initialize ChromaDB client and collection."""
        try:
            os.makedirs(self.db_path, exist_ok=True)

            # Initialize ChromaDB client with persistence
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Create or get collection for ArXiv papers
            self.collection = self.client.get_or_create_collection(
                name="arxiv_papers",
                metadata={"description": "ArXiv paper embeddings for semantic search"}
            )

            logger.info(f"ChromaDB initialized at {self.db_path}")
            logger.info(f"Collection 'arxiv_papers' has {self.collection.count()} documents")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def use_database(self, db_path: Optional[str] = None):
        """Switch to a different ChromaDB path."""
        if not db_path or db_path == self.db_path:
            return

        self.db_path = db_path
        self._initialize_db()

    def add_paper(
        self,
        paper_id: str,
        title: str,
        abstract: str,
        content_chunks: List[Dict[str, Any]],
        metadata: Dict[str, Any] = None
    ):
        """
        Add a paper to the semantic search index.

        Args:
            paper_id: ArXiv paper ID
            title: Paper title
            abstract: Paper abstract
            content_chunks: List of content chunks from the paper
            metadata: Additional metadata
        """
        try:
            documents = []
            embeddings_list = []
            metadatas = []
            ids = []

            # Prepare abstract for indexing
            abstract_text = f"Title: {title}\n\nAbstract: {abstract}"
            documents.append(abstract_text)
            ids.append(f"{paper_id}_abstract")

            meta = self._normalize_metadata({
                'paper_id': paper_id,
                'title': title,
                'type': 'abstract',
                'indexed_at': datetime.now().isoformat()
            })
            if metadata:
                meta.update(self._normalize_metadata(metadata))
            metadatas.append(meta)

            # Add content chunks
            for i, chunk in enumerate(content_chunks):
                chunk_text = chunk.get('text', '')
                if not chunk_text.strip():
                    continue

                documents.append(chunk_text)
                ids.append(f"{paper_id}_chunk_{i}")

                chunk_meta = self._normalize_metadata({
                    'paper_id': paper_id,
                    'title': title,
                    'type': 'content',
                    'chunk_id': i,
                    'chunk_start': chunk.get('start') if chunk.get('start') is not None else 0,
                    'chunk_end': chunk.get('end') if chunk.get('end') is not None else 0,
                    'indexed_at': datetime.now().isoformat()
                })
                if metadata:
                    chunk_meta.update(self._normalize_metadata(metadata))
                metadatas.append(chunk_meta)

            # Generate embeddings
            logger.info(f"Generating embeddings for {len(documents)} documents from paper {paper_id}")
            embeddings = self.embedding_manager.generate_embeddings(documents)

            # Add to ChromaDB
            logger.debug(f"Sample metadata for {paper_id}: {metadatas[:1]}")
            self.collection.add(
                documents=documents,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
                ids=ids
            )

            logger.info(f"Added paper {paper_id} with {len(documents)} chunks to index")

        except Exception as e:
            logger.error(f"Failed to add paper {paper_id}: {e}")
            raise

    @staticmethod
    def _normalize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize metadata values to ChromaDB-compatible types."""

        normalized = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                normalized[key] = value
            elif isinstance(value, (list, tuple, set)):
                normalized[key] = ', '.join(str(item) for item in value)
            elif isinstance(value, dict):
                normalized[key] = json.dumps(value)
            else:
                normalized[key] = str(value)
        return normalized

    def search(
        self,
        query: str,
        n_results: int = 10,
        filter_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search on indexed papers.

        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of search results with scores and metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_manager.generate_query_embedding(query)

            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=filter_metadata if filter_metadata else None
            )

            # Format results
            formatted_results = []
            if results and results['ids']:
                for i in range(len(results['ids'][0])):
                    result = {
                        'id': results['ids'][0][i],
                        'score': 1 - results['distances'][0][i],  # Convert distance to similarity
                        'document': results['documents'][0][i] if results['documents'] else None,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {}
                    }
                    formatted_results.append(result)

            return formatted_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def search_similar_papers(
        self,
        paper_id: str,
        n_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find papers similar to a given paper.

        Args:
            paper_id: ArXiv paper ID
            n_results: Number of similar papers to return

        Returns:
            List of similar papers
        """
        try:
            # Get the paper's abstract embedding
            abstract_result = self.collection.get(
                ids=[f"{paper_id}_abstract"],
                include=['embeddings', 'metadatas']
            )

            if not abstract_result['ids']:
                raise ValueError(f"Paper {paper_id} not found in index")

            # Search for similar papers
            embedding = abstract_result['embeddings'][0]
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=n_results + 1,  # +1 to exclude self
                where={"type": "abstract"}  # Only search abstracts
            )

            # Format and filter results (exclude self)
            formatted_results = []
            if results and results['ids']:
                for i in range(len(results['ids'][0])):
                    if results['metadatas'][0][i].get('paper_id') != paper_id:
                        result = {
                            'paper_id': results['metadatas'][0][i].get('paper_id'),
                            'title': results['metadatas'][0][i].get('title'),
                            'score': 1 - results['distances'][0][i],
                            'metadata': results['metadatas'][0][i]
                        }
                        formatted_results.append(result)

            return formatted_results[:n_results]

        except Exception as e:
            logger.error(f"Similar paper search failed: {e}")
            raise

    def get_paper_chunks(
        self,
        paper_id: str,
        chunk_type: str = None
    ) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific paper.

        Args:
            paper_id: ArXiv paper ID
            chunk_type: Optional filter for chunk type ('abstract' or 'content')

        Returns:
            List of paper chunks
        """
        try:
            where_clause = {"paper_id": paper_id}
            if chunk_type:
                where_clause["type"] = chunk_type

            results = self.collection.get(
                where=where_clause,
                include=['documents', 'metadatas']
            )

            chunks = []
            if results and results['ids']:
                for i in range(len(results['ids'])):
                    chunk = {
                        'id': results['ids'][i],
                        'text': results['documents'][i],
                        'metadata': results['metadatas'][i]
                    }
                    chunks.append(chunk)

            # Sort chunks by chunk_id if they're content chunks
            if chunk_type == 'content':
                chunks.sort(key=lambda x: x['metadata'].get('chunk_id', 0))

            return chunks

        except Exception as e:
            logger.error(f"Failed to get chunks for paper {paper_id}: {e}")
            raise

    def delete_paper(self, paper_id: str):
        """
        Remove a paper from the index.

        Args:
            paper_id: ArXiv paper ID to remove
        """
        try:
            # Get all chunk IDs for this paper
            results = self.collection.get(
                where={"paper_id": paper_id},
                include=[]
            )

            if results and results['ids']:
                # Delete all chunks
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} chunks for paper {paper_id}")
            else:
                logger.warning(f"Paper {paper_id} not found in index")

        except Exception as e:
            logger.error(f"Failed to delete paper {paper_id}: {e}")
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the indexed papers.

        Returns:
            Dictionary with index statistics
        """
        try:
            total_count = self.collection.count()

            # Get unique papers
            all_metadata = self.collection.get(include=['metadatas'])['metadatas']
            unique_papers = set()
            abstract_count = 0
            content_count = 0

            for meta in all_metadata:
                unique_papers.add(meta.get('paper_id'))
                if meta.get('type') == 'abstract':
                    abstract_count += 1
                elif meta.get('type') == 'content':
                    content_count += 1

            return {
                'total_documents': total_count,
                'unique_papers': len(unique_papers),
                'abstracts': abstract_count,
                'content_chunks': content_count,
                'db_path': self.db_path
            }

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            raise
