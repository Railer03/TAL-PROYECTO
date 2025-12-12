import os
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Generator:
    def __init__(self, api_key=None, provider="openai", secondary_key=None):
        self.api_key = api_key
        self.secondary_key = secondary_key
        self.provider = provider
        self.client = None
        self.model = None
        self.groq_client = None
        
        self._initialize_provider()

    def _initialize_provider(self):
        if self.provider == "openai" and self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                logging.warning("OpenAI library not installed.")
        elif self.provider == "gemini" and self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-2.5-flash')
            except ImportError:
                logging.warning("Google Generative AI library not installed.")
        
        # Always initialize Groq if key is available (as backup or primary)
        if self.secondary_key or (self.provider == "groq" and self.api_key):
            groq_key = self.secondary_key if self.secondary_key else self.api_key
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=groq_key)
            except ImportError:
                logging.warning("Groq library not installed.")

    def _call_groq(self, prompt):
        logging.info("Using Groq (Llama 3.3) for generation...")
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Eres un asistente útil y veraz."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Error calling Groq: {e}")
            return "Error generando respuesta con Groq."

    def generate_answer(self, query, context_chunks):
        """
        Generates an answer using the LLM and the retrieved context.
        """
        # 1. Construct Prompt
        context_text = ""
        for i, chunk in enumerate(context_chunks):
            source = chunk.get('media_outlet', 'Desconocido')
            date = chunk.get('date', 'Fecha desconocida')
            title = chunk.get('title', '').strip()
            text = chunk.get('full_content', '')
            context_text += (
                f"[{i+1}] Fuente: {source} | Fecha: {date} | Título: {title}\n"
                f"{text}\n\n"
            )

        prompt = f"""
        Eres un asistente de noticias inteligente llamado Sophia.
        Usa solo el contexto dado para responder. Si la respuesta no está en el contexto, di que no tienes información suficiente.
        Cita las fuentes usando los números [1], [2], etc. Si citas algo, al final agrega la línea "Fuentes:" listando solo las referencias usadas ([n] medio, fecha, título). Si no usas ninguna, omite la sección de fuentes.

        Contexto:
        {context_text}

        Pregunta: {query}

        Respuesta:
        """

        # 2. Call LLM
        if self.provider == "openai" and self.client:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Eres un asistente útil y veraz."},
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.choices[0].message.content
            except Exception as e:
                logging.error(f"Error calling OpenAI: {e}")
                return "Lo siento, hubo un error al generar la respuesta."
        elif self.provider == "gemini" and self.model:
            retries = 3
            for attempt in range(retries):
                try:
                    response = self.model.generate_content(prompt)
                    return response.text
                except Exception as e:
                    if "429" in str(e):
                        logging.warning(f"Gemini Rate Limit hit. Switching to Groq backup...")
                        if self.groq_client:
                            self.provider = "groq" # Switch permanently for this session
                            return self._call_groq(prompt)
                        
                        wait_time = 5 * (attempt + 1)
                        logging.warning(f"No backup available. Waiting {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logging.error(f"Error calling Gemini: {e}")
                        if self.groq_client:
                            logging.info("Switching to Groq due to error.")
                            self.provider = "groq"
                            return self._call_groq(prompt)
                        return "Lo siento, hubo un error al generar la respuesta con Gemini."
            return "Lo siento, el servicio está saturado."
            
        elif self.provider == "groq" and self.groq_client:
            return self._call_groq(prompt)
            
        else:
            # Mock response for testing without API key
            logging.warning("No API key provided or provider not supported. Returning mock response.")
            return f"**Respuesta Simulada:**\nBasado en las noticias recuperadas, aquí tienes un resumen...\n(Contexto usado: {len(context_chunks)} artículos)\n\n{context_text[:500]}..."

if __name__ == "__main__":
    gen = Generator(provider="mock")
    print(gen.generate_answer("¿Qué pasó?", [{"full_content": "Algo pasó.", "media_outlet": "Emol", "date": "Hoy"}]))
