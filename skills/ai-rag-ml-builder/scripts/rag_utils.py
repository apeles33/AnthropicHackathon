"""
RAG Utilities for Vector Database and Embeddings
Provides simple, self-contained implementations for RAG systems
"""

import numpy as np
import json
from typing import List, Dict, Tuple, Any
from collections import defaultdict


class SimpleEmbedder:
    """
    Simple TF-IDF-based embedder for demonstrations
    No external APIs required
    """
    
    def __init__(self):
        self.vocabulary = {}
        self.idf = {}
        self.fitted = False
    
    def fit(self, documents: List[str]):
        """Fit the embedder on a corpus of documents"""
        # Build vocabulary and document frequency
        doc_freq = defaultdict(int)
        all_words = set()
        
        for doc in documents:
            words = set(doc.lower().split())
            for word in words:
                doc_freq[word] += 1
                all_words.add(word)
        
        # Create vocabulary
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(all_words))}
        
        # Calculate IDF
        num_docs = len(documents)
        for word in self.vocabulary:
            self.idf[word] = np.log(num_docs / (doc_freq[word] + 1))
        
        self.fitted = True
    
    def embed(self, text: str) -> np.ndarray:
        """Convert text to embedding vector"""
        if not self.fitted:
            raise ValueError("Embedder must be fitted first")
        
        # Create zero vector
        vector = np.zeros(len(self.vocabulary))
        
        # Count term frequencies
        words = text.lower().split()
        word_counts = defaultdict(int)
        for word in words:
            if word in self.vocabulary:
                word_counts[word] += 1
        
        # Calculate TF-IDF
        for word, count in word_counts.items():
            if word in self.vocabulary:
                idx = self.vocabulary[word]
                tf = count / len(words) if words else 0
                vector[idx] = tf * self.idf.get(word, 0)
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts"""
        return [self.embed(text) for text in texts]


class VectorDatabase:
    """
    Simple in-memory vector database with similarity search
    """
    
    def __init__(self):
        self.vectors = []
        self.metadata = []
        self.embedder = SimpleEmbedder()
    
    def add_documents(self, documents: List[str], metadata: List[Dict[str, Any]] = None):
        """Add documents to the vector database"""
        if not documents:
            return
        
        # Fit embedder if first time
        if not self.embedder.fitted:
            self.embedder.fit(documents)
        
        # Embed and store
        embeddings = self.embedder.embed_batch(documents)
        self.vectors.extend(embeddings)
        
        if metadata is None:
            metadata = [{"text": doc} for doc in documents]
        self.metadata.extend(metadata)
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Search for most similar documents"""
        if not self.vectors:
            return []
        
        # Embed query
        query_vector = self.embedder.embed(query)
        
        # Calculate cosine similarities
        similarities = []
        for i, vec in enumerate(self.vectors):
            sim = np.dot(query_vector, vec)
            similarities.append((i, sim))
        
        # Sort and get top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:top_k]
        
        # Return metadata and scores
        results = []
        for idx, score in top_results:
            results.append((self.metadata[idx], float(score)))
        
        return results
    
    def save(self, filepath: str):
        """Save database to file"""
        data = {
            "vectors": [v.tolist() for v in self.vectors],
            "metadata": self.metadata,
            "vocabulary": self.embedder.vocabulary,
            "idf": self.embedder.idf
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    def load(self, filepath: str):
        """Load database from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.vectors = [np.array(v) for v in data["vectors"]]
        self.metadata = data["metadata"]
        self.embedder.vocabulary = data["vocabulary"]
        self.embedder.idf = data["idf"]
        self.embedder.fitted = True


class ConversationMemory:
    """
    Simple conversation memory for chatbots
    Stores conversation history with semantic search
    """
    
    def __init__(self, max_history: int = 50):
        self.max_history = max_history
        self.history = []
        self.vector_db = VectorDatabase()
    
    def add_exchange(self, user_message: str, assistant_message: str):
        """Add a conversation exchange"""
        exchange = {
            "user": user_message,
            "assistant": assistant_message,
            "timestamp": len(self.history)
        }
        self.history.append(exchange)
        
        # Add to vector DB for semantic search
        combined_text = f"User: {user_message}\nAssistant: {assistant_message}"
        self.vector_db.add_documents([combined_text], [exchange])
        
        # Trim if exceeds max
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def search_history(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search conversation history"""
        results = self.vector_db.search(query, top_k=top_k)
        return [meta for meta, score in results]
    
    def get_recent(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get recent conversation history"""
        return self.history[-n:] if self.history else []


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    
    return chunks


def generate_dummy_documents(num_docs: int = 20, topic: str = "general") -> List[Dict[str, str]]:
    """
    Generate dummy documents for testing
    """
    topics_data = {
        "general": [
            "Introduction to machine learning and artificial intelligence concepts",
            "Deep learning architectures and neural networks explained",
            "Natural language processing techniques and applications",
            "Computer vision and image recognition systems",
            "Data science best practices and methodologies",
            "Statistical analysis and predictive modeling",
            "Big data processing and distributed computing",
            "Cloud computing platforms and services overview",
            "Software engineering principles and design patterns",
            "Database management systems and SQL fundamentals"
        ],
        "medical": [
            "Cardiovascular disease prevention and treatment options",
            "Diabetes management and blood sugar monitoring",
            "Cancer screening guidelines and early detection",
            "Mental health disorders and therapeutic approaches",
            "Infectious disease control and vaccination programs",
            "Nutrition and dietary recommendations for health",
            "Exercise physiology and fitness training methods",
            "Pharmaceutical drug interactions and side effects",
            "Medical imaging techniques and diagnostic tools",
            "Telemedicine and digital health innovations"
        ],
        "business": [
            "Strategic planning and business development frameworks",
            "Financial analysis and investment strategies",
            "Marketing automation and customer acquisition",
            "Supply chain management and logistics optimization",
            "Human resources management and talent development",
            "Project management methodologies and tools",
            "Risk management and compliance regulations",
            "Digital transformation and technology adoption",
            "Mergers and acquisitions processes and strategies",
            "Corporate governance and stakeholder relations"
        ]
    }
    
    templates = topics_data.get(topic, topics_data["general"])
    
    documents = []
    for i in range(num_docs):
        template = templates[i % len(templates)]
        doc = {
            "id": f"doc_{i+1}",
            "title": f"Document {i+1}: {template.split()[0:3]}",
            "content": f"{template}. " * (5 + (i % 3)),  # Vary length
            "category": topic,
            "timestamp": f"2024-{1 + (i % 12):02d}-{1 + (i % 28):02d}"
        }
        documents.append(doc)
    
    return documents


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)
