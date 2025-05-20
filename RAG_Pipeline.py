import os
import pandas as pd
import joblib
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from dotenv import load_dotenv
from openai import OpenAI

# Ladda miljövariabler
load_dotenv()

# Konfigurera NLTK
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.synonyms = {
            'chicken': ['poultry', 'hen', 'chicken breast'],
            'beef': ['ground beef', 'sirloin', 'roast beef'],
            'potato': ['potatoes', 'spuds', 'yukon gold']
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self._preprocess(text) for text in X]

    def _preprocess(self, text):
        text = str(text).lower()
        for key, synonyms in self.synonyms.items():
            for synonym in synonyms:
                text = re.sub(rf'\b{re.escape(synonym)}\b', key, text)
        text = re.sub(r'[^\w\s,-]', '', text)
        tokens = nltk.word_tokenize(text)
        return ' '.join([self.lemmatizer.lemmatize(token) for token in tokens])

class RecipeRAG:
    def __init__(self, model_path, data_path):
        self.pipeline = joblib.load(model_path)
        self.df = pd.read_csv(data_path)
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def retrieve(self, query, top_k=5):  # ← DENNA METOD MÅSTE FINNAS
        processed_query = self.pipeline['preprocessor'].transform([query])
        query_vec = self.pipeline['tfidf'].transform(processed_query)
        distances, indices = self.pipeline['knn'].kneighbors(query_vec, n_neighbors=top_k)
        
        results = self.df.iloc[indices[0]].copy()
        results['similarity'] = 1 - distances[0]
        return results


class RecipeChatbot:
    def __init__(self, rag_system):
        self.rag = rag_system
        
    def handle_message(self, user_input):
        recipes = self.rag.retrieve(user_input, top_k=3)
        prompt = f"""Användaren frågar: {user_input}
        Relevanta recept: {recipes[['name', 'ingredients']].to_dict()}
        Formulera ett hjälpsamt svar på svenska."""
        
        response = self.rag.client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        return response.choices[0].message.content
