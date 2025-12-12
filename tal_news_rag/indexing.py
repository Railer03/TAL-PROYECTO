import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import pickle
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IndexingEngine:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.metadata = []

    def create_embeddings(self, texts):
        """Generates embeddings for a list of texts."""
        logging.info(f"Generating embeddings for {len(texts)} texts using {self.model_name}...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings

    def build_index(self, embeddings, metadata_list):
        """Builds the FAISS index."""
        logging.info("Building FAISS index...")
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        self.metadata = metadata_list
        logging.info(f"Index built with {self.index.ntotal} vectors.")

    def save_index(self, index_path, metadata_path):
        """Saves the index and metadata to disk."""
        logging.info(f"Saving index to {index_path}...")
        faiss.write_index(self.index, index_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        logging.info("Index and metadata saved.")

    def load_index(self, index_path, metadata_path):
        """Loads the index and metadata from disk."""
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError("Index or metadata file not found.")
        
        logging.info(f"Loading index from {index_path}...")
        self.index = faiss.read_index(index_path)
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        logging.info(f"Loaded index with {self.index.ntotal} vectors.")

    def search(self, query_text, k=5):
        """Searches the index for the query text."""
        query_vector = self.model.encode([query_text])
        distances, indices = self.index.search(np.array(query_vector).astype('float32'), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                result = self.metadata[idx]
                result['score'] = float(distances[0][i])
                results.append(result)
        return results

if __name__ == "__main__":
    # Test
    engine = IndexingEngine()
    texts = ["Hola mundo", "Noticia de Chile", "Inteligencia Artificial"]
    meta = [{"id": 1}, {"id": 2}, {"id": 3}]
    emb = engine.create_embeddings(texts)
    engine.build_index(emb, meta)
    res = engine.search("Chile")
    print(res)
