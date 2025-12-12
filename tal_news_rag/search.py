import logging
from datetime import datetime, timedelta
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IntelligentSearch:
    def __init__(self, indexing_engine):
        self.engine = indexing_engine

    def parse_query(self, query):
        """
        Simulates an Agent that extracts filters from the query.
        In a real production system, this would be an LLM call.
        """
        filters = {}
        
        # Simple heuristic for "semana pasada" (last week)
        if "semana pasada" in query.lower():
            today = datetime.now() # In the simulation context, this should be the context date
            # But for now let's just say it sets a flag
            filters['date_range'] = 'last_week'
        
        # Simple heuristic for regions (expand as needed)
        regions = ["Valparaíso", "Santiago", "Biobío", "Coquimbo", "La Serena", "Concepción"]
        for region in regions:
            if region.lower() in query.lower():
                filters['region'] = region
                break
        
        return filters

    def search(self, query, k=10):
        """
        Performs the search with filtering.
        """
        # 1. Parse Query (Agent Step)
        filters = self.parse_query(query)
        logging.info(f"Agent extracted filters: {filters}")

        # 2. Retrieve candidates (get more than k to allow for filtering)
        # We fetch k*3 candidates to have enough after filtering
        raw_results = self.engine.search(query, k=k*3)

        # 3. Apply Filters
        filtered_results = []
        for res in raw_results:
            # Filter by Region
            if 'region' in filters:
                # Check if the detected region matches (fuzzy or exact)
                if filters['region'].lower() not in str(res.get('detected_region', '')).lower():
                    continue
            
            # Filter by Date (Mock implementation for 'last_week')
            # In a real scenario, we would compare dates.
            # Since the dataset is from Sep 2025, we can implement logic if needed.
            
            filtered_results.append(res)
            if len(filtered_results) >= k:
                break
        
        logging.info(f"Returning {len(filtered_results)} results after filtering.")
        return filtered_results

if __name__ == "__main__":
    # Test is dependent on IndexingEngine being populated
    pass
