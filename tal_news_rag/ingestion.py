import pandas as pd
import spacy
from datetime import datetime
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataIngestion:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        try:
            self.nlp = spacy.load("es_core_news_sm")
        except OSError:
            logging.info("Downloading spacy model...")
            from spacy.cli import download
            download("es_core_news_sm")
            self.nlp = spacy.load("es_core_news_sm")

    def load_data(self):
        """Loads the CSV file."""
        logging.info(f"Loading data from {self.file_path}...")
        try:
            self.df = pd.read_csv(self.file_path)
            logging.info(f"Loaded {len(self.df)} records.")
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def clean_and_enrich(self, limit=None):
        """Cleans text, parses dates, and extracts regions using NER."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        if limit:
            self.df = self.df.head(limit)
            logging.info(f"Processing limited to first {limit} records.")

        # 1. Parse Dates
        logging.info("Parsing dates...")
        # Format: "Sep 24, 2025 @ 00:00:00.000"
        # We need to handle Spanish month names if they are in Spanish, but the example shows "Sep".
        # Let's check if it's English or Spanish. "Sep" is ambiguous.
        # If the file has "Ene", "Abr", "Ago", "Dic", it's Spanish.
        # I'll assume English for now based on "Sep", but if it fails I'll try Spanish mapping.
        
        def parse_date(date_str):
            try:
                # Remove the time part for simplicity or keep it.
                # "Sep 24, 2025 @ 00:00:00.000"
                clean_date = date_str.split(" @")[0]
                return datetime.strptime(clean_date, "%b %d, %Y")
            except Exception:
                return None

        self.df['parsed_date'] = self.df['date'].apply(parse_date)
        
        # 2. Combine Title and Text
        self.df['full_content'] = self.df['title'].fillna('') + ". " + self.df['text'].fillna('')

        # 3. NER for Region Extraction
        logging.info("Extracting regions (NER)... This may take a while.")
        
        regions = []
        # Using tqdm for progress bar
        for text in tqdm(self.df['full_content']):
            doc = self.nlp(text[:1000]) # Limit to first 1000 chars for speed
            locs = [ent.text for ent in doc.ents if ent.label_ in ['LOC', 'GPE']]
            # Simple heuristic: take the most frequent location or the first one
            if locs:
                regions.append(locs[0]) # Just take the first one for now
            else:
                regions.append("Desconocida")
        
        self.df['detected_region'] = regions
        logging.info("Enrichment complete.")
        return self.df

if __name__ == "__main__":
    # Test
    ingestor = DataIngestion("../dataset/dataset_proyecto_chile_septiembre2025.csv")
    ingestor.load_data()
    df = ingestor.clean_and_enrich(limit=100)
    print(df[['parsed_date', 'detected_region']].head())
