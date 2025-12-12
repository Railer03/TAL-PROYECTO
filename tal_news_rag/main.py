import os
import logging
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from ingestion import DataIngestion
from indexing import IndexingEngine
from search import IntelligentSearch
from generation import Generator

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "../dataset/dataset_proyecto_chile_septiembre2025.csv")
INDEX_PATH = os.path.join(BASE_DIR, "news_index.faiss")
METADATA_PATH = os.path.join(BASE_DIR, "news_metadata.pkl")

def main():
    # 1. Initialize Components
    ingestor = DataIngestion(DATASET_PATH)
    indexer = IndexingEngine()
    
    # 2. Check if Index exists
    if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
        logging.info("Loading existing index...")
        indexer.load_index(INDEX_PATH, METADATA_PATH)
    else:
        logging.info("Index not found. Starting ingestion pipeline...")
        ingestor.load_data()
        # Limit to 1000 for prototype speed, remove limit for full run
        df = ingestor.clean_and_enrich(limit=1000) 
        
        # Prepare data for indexing
        texts = df['full_content'].tolist()
        metadata = df.to_dict(orient='records')
        
        # Create Embeddings and Index
        embeddings = indexer.create_embeddings(texts)
        indexer.build_index(embeddings, metadata)
        indexer.save_index(INDEX_PATH, METADATA_PATH)

    # 3. Initialize Search and Generation
    searcher = IntelligentSearch(indexer)
    
    # Determine Provider
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GOOGLE_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    if openai_key:
        logging.info("Using OpenAI Provider")
        generator = Generator(api_key=openai_key, provider="openai", secondary_key=groq_key)
    elif gemini_key:
        logging.info("Using Google Gemini Provider (with Groq backup)")
        generator = Generator(api_key=gemini_key, provider="gemini", secondary_key=groq_key)
    elif groq_key:
        logging.info("Using Groq Provider")
        generator = Generator(api_key=groq_key, provider="groq")
    else:
        logging.warning("No API Key found. Using Mock Provider.")
        generator = Generator(provider="mock")

    # 4. Interactive Loop or One-shot
    print("\n" + "="*50)
    print("Bienvenido al Sistema de Noticias 'Sophia Search'")
    print("="*50)
    
    if len(sys.argv) > 1:
        # One-shot mode
        query = " ".join(sys.argv[1:])
        print(f"\nModo automático. Buscando: '{query}'")
        results = searcher.search(query)
        
        if results:
            print(f"Encontrados {len(results)} artículos relevantes. Generando respuesta...")
            answer = generator.generate_answer(query, results)
            print("\n" + "-"*30)
            print("RESPUESTA:")
            print(answer)
            print("-"*30)
        else:
            print("No se encontraron noticias relevantes.")
    else:
        while True:
            query = input("\nIngresa tu pregunta (o 'salir'): ")
            if query.lower() in ['salir', 'exit', 'quit']:
                break
            
            print(f"\nBuscando noticias sobre: '{query}'...")
            results = searcher.search(query)
            
            if not results:
                print("No se encontraron noticias relevantes.")
                continue
                
            print(f"Encontrados {len(results)} artículos relevantes. Generando respuesta...")
            answer = generator.generate_answer(query, results)
            
            print("\n" + "-"*30)
            print("RESPUESTA:")
            print(answer)
            print("-"*30)

if __name__ == "__main__":
    main()
