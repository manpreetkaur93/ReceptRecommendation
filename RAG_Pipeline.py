# RAG_Pipeline.py
import os
import re
import pandas as pd
import joblib
import nltk
from dotenv import load_dotenv
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from openai import OpenAI

# Ladda NLTK-resurser
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.synonyms = {
            'chicken': ['poultry', 'hen', 'chicken breast', 'kyckling'],
            'beef': ['ground beef', 'sirloin', 'roast beef', 'nötkött'],
            'potato': ['potatoes', 'spuds', 'yukon gold', 'potatis'],
            'zucchini': ['courgette', 'squash', 'zucchin', 'zuchini']  # Fånga stavfel
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self._preprocess(text) for text in X]

    def _preprocess(self, text):
        text = str(text).lower()
        
        # Ersätt svenska och stavfel
        for key, synonyms in self.synonyms.items():
            for synonym in synonyms:
                text = re.sub(rf'\b{re.escape(synonym)}\b', key, text)
        
        # Rensa specialtecken
        text = re.sub(r'[^a-z0-9\s,-]', '', text)
        
        # Tokenisera och filtrera
        tokens = nltk.word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if len(token) > 2]
        
        return ' '.join(tokens) if tokens else ''

class RecipeRAG:
    def __init__(self, model_path, data_path):
        load_dotenv()
        self.pipeline = joblib.load(model_path)
        self.df = pd.read_csv(data_path)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def retrieve(self, query, top_k=5, similarity_threshold=0.3):
        processed_query = self.pipeline['preprocessor'].transform([query])
        query_vec = self.pipeline['tfidf'].transform(processed_query)
        distances, indices = self.pipeline['knn'].kneighbors(query_vec, n_neighbors=top_k*2)
        
        results = self.df.iloc[indices[0]].copy()
        results['similarity'] = 1 - distances[0]
        
        # Filtrera bort låg similarity
        filtered = results[results['similarity'] > similarity_threshold]
        return filtered.sort_values('similarity', ascending=False).head(top_k)

class RecipeChatbot:
    def __init__(self, rag_system):
        self.rag = rag_system
        # Add common thank you phrases
        self.thank_you_phrases = [
            "thank", "thanks", "thankyou", "thank you", "thx", 
            "appreciate", "grateful", "tack", "tackar"
        ]
        
    def handle_message(self, user_input):
        # Check if input is just a thank you message
        if self._is_thank_you(user_input):
            return "You're welcome! Feel free to ask if you need more recipe ideas in the future."
        
        # Continue with normal recipe recommendation flow
        recipes = self.rag.retrieve(user_input, top_k=3)
        
        prompt = f"""User query: {user_input}
        Relevant recipes (English only): {recipes[['name', 'ingredients']].to_dict()}
        Formulate a helpful response in English with:
        - 3 recipe suggestions max
        - Ignore non-English content
        - No zucchini recipes unless explicitly requested"""
        
        response = self.rag.client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an English-only recipe assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200
        )
        return response.choices[0].message.content
    
    def _is_thank_you(self, text):
        """Check if the message is just a thank you"""
        text = text.lower().strip()
        
        # If the message is just a thank you (possibly with some polite words)
        for phrase in self.thank_you_phrases:
            if phrase in text and len(text.split()) < 6:
                return True
                
        return False
