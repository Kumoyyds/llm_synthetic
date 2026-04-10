from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple, Union, Optional, Dict
import pickle
from pathlib import Path
import threading

# Global lock for thread-safe model operations
_model_lock = threading.Lock()

# Shared model cache - singleton pattern for thread safety
_model_cache: Dict[str, SentenceTransformer] = {}


def _get_shared_model(model_name: str, device: str = None) -> SentenceTransformer:
    """
    Get or create a shared SentenceTransformer model instance.
    Thread-safe singleton pattern to avoid multiple model instantiations.
    """
    global _model_cache
    with _model_lock:
        if model_name not in _model_cache:
            print(f"Loading model: {model_name} (device: {device or 'auto'})")
            _model_cache[model_name] = SentenceTransformer(model_name, device=device)
        return _model_cache[model_name]


class SimilaritySearcher:
    """
    Optimized similarity search using sentence-transformers.
    Pre-computes embeddings for the corpus for fast repeated queries.
    Supports caching embeddings to disk for persistent storage.
    Cache stores text->embedding mapping for incremental updates.
    Uses a shared model instance for thread safety.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = None):
        """
        Args:
            model_name: Name of the sentence-transformer model
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model = _get_shared_model(model_name, device)
        self.corpus_embeddings = None
        self.corpus = None
        self.model_name = model_name
        self._embedding_cache: Dict[str, np.ndarray] = {}  # text -> normalized embedding
    
    def fit(self, corpus: List[str], cache_path: Optional[str] = None) -> 'SimilaritySearcher':
        """
        Pre-compute embeddings for the corpus, with optional caching.
        Supports incremental updates: cached embeddings are reused,
        new texts are computed and added to the cache.
        
        Args:
            corpus: List of texts to search through
            cache_path: Optional path to save/load embeddings cache (.pkl file)
        """
        self.corpus = corpus
        cache_updated = False
        
        # Try to load from cache
        if cache_path and Path(cache_path).exists():
            # print(f"Loading embeddings cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
                # Check model compatibility
                if cached.get('model_name') == self.model_name:
                    self._embedding_cache = cached.get('text_to_embedding', {})
                    # print(f"Loaded {len(self._embedding_cache)} cached embeddings")
                else:
                    # print(f"Cache model mismatch ({cached.get('model_name')} vs {self.model_name}), starting fresh...")
                    self._embedding_cache = {}
        
        # Identify texts that need embedding
        texts_to_compute = [text for text in corpus if text not in self._embedding_cache]
        cached_count = len(corpus) - len(texts_to_compute)
        
        if texts_to_compute:
            print(f"Computing embeddings for {len(texts_to_compute)} new texts ({cached_count} from cache)...")
            with _model_lock:
                new_embeddings = self.model.encode(texts_to_compute, convert_to_numpy=True, show_progress_bar=True)
            # Normalize for cosine similarity
            new_embeddings = new_embeddings / np.linalg.norm(new_embeddings, axis=1, keepdims=True)
            
            # Update cache dict with new embeddings
            for text, emb in zip(texts_to_compute, new_embeddings):
                self._embedding_cache[text] = emb
            cache_updated = True
        else:
            # print(f"All {len(corpus)} embeddings loaded from cache")
            pass
        
        # Build corpus_embeddings array in corpus order
        self.corpus_embeddings = np.array([self._embedding_cache[text] for text in corpus])
        
        # Save updated cache if needed
        if cache_path and cache_updated:
            print(f"Saving updated cache ({len(self._embedding_cache)} embeddings): {cache_path}")
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'text_to_embedding': self._embedding_cache,
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
        
        # Encode query (thread-safe)
        with _model_lock:
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