from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple, Union, Optional
import pickle
from pathlib import Path

class SimilaritySearcher:
    """
    Optimized similarity search using sentence-transformers.
    Pre-computes embeddings for the corpus for fast repeated queries.
    Supports caching embeddings to disk for persistent storage.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.corpus_embeddings = None
        self.corpus = None
        self.model_name = model_name
    
    def fit(self, corpus: List[str], cache_path: Optional[str] = None) -> 'SimilaritySearcher':
        """
        Pre-compute embeddings for the corpus, with optional caching.
        
        Args:
            corpus: List of texts to search through
            cache_path: Optional path to save/load embeddings cache (.pkl file)
        """
        self.corpus = corpus
        
        # Try to load from cache
        if cache_path and Path(cache_path).exists():
            print(f"Loading embeddings from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
                # Verify cache is valid (same corpus size and model)
                if (cached.get('corpus_size') == len(corpus) and 
                    cached.get('model_name') == self.model_name):
                    self.corpus_embeddings = cached['embeddings']
                    print(f"Loaded {len(corpus)} embeddings from cache")
                    return self
                else:
                    print("Cache invalid (corpus size or model changed), recomputing...")
        
        # Compute embeddings
        print(f"Computing embeddings for {len(corpus)} texts...")
        self.corpus_embeddings = self.model.encode(corpus, convert_to_numpy=True, show_progress_bar=True)
        # Normalize for cosine similarity
        self.corpus_embeddings = self.corpus_embeddings / np.linalg.norm(self.corpus_embeddings, axis=1, keepdims=True)
        
        # Save to cache if path provided
        if cache_path:
            print(f"Saving embeddings to cache: {cache_path}")
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'embeddings': self.corpus_embeddings,
                    'corpus_size': len(corpus),
                    'model_name': self.model_name
                }, f)
        
        return self
    
    def search(self, query: str, top_n: int = 5, bottom_m: int = 5) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """
        Search for most and least similar items.
        
        Args:
            query: The content to search for
            top_n: Number of most relevant items to return
            bottom_m: Number of least relevant items to return
            
        Returns:
            Tuple of (top_n most similar, bottom_m least similar)
            Each item is a tuple of (text, similarity_score)
        """
        if self.corpus_embeddings is None:
            raise ValueError("Must call fit() with corpus before searching")
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Compute cosine similarities (dot product since normalized)
        similarities = np.dot(self.corpus_embeddings, query_embedding.T).flatten()
        
        # Get indices sorted by similarity
        sorted_indices = np.argsort(similarities)
        
        # Top N (highest similarity - at the end of sorted array)
        top_indices = sorted_indices[-top_n:][::-1]
        top_results = [(self.corpus[i], similarities[i]) for i in top_indices]
        
        # Bottom M (lowest similarity - at the start of sorted array)
        bottom_indices = sorted_indices[:bottom_m]
        bottom_results = [(self.corpus[i], similarities[i]) for i in bottom_indices]
        
        return top_results, bottom_results


# Convenience function for one-off searches
def similarity_search(
    query: str,
    corpus: List[str],
    top_n: int = 5,
    bottom_m: int = 5,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """
    One-off similarity search. For repeated searches on the same corpus,
    use SimilaritySearcher class instead for better performance.
    
    Args:
        query: The content to search for
        corpus: List of texts to search through
        top_n: Number of most relevant items to return
        bottom_m: Number of least relevant items to return
        model_name: Name of the sentence-transformer model
        
    Returns:
        Tuple of (top_n most similar, bottom_m least similar)
        Each item is a tuple of (text, similarity_score)
    """
    searcher = SimilaritySearcher(model_name)
    searcher.fit(corpus)
    return searcher.search(query, top_n, bottom_m)