# test_model.py
import pandas as pd
import joblib
import re
import os
import nltk
from sklearn.model_selection import train_test_split

# Fixa NLTK data path för miljöer med begränsade rättigheter
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', download_dir=nltk_data_path)
    nltk.download('punkt', download_dir=nltk_data_path)

from nltk.stem import WordNetLemmatizer

# --------------------------
# Konfiguration
# --------------------------
DATA_FILE = "recipes_with_ingredients_and_tags.csv"
MODEL_FILE = "models/full_pipeline.pkl"

# --------------------------
# Preprocess-funktion
# --------------------------
ingredient_synonyms = {
    'chicken': ['poultry', 'hen', 'chicken breast'],
    'beef': ['ground beef', 'sirloin', 'roast beef'],
    'potato': ['potatoes', 'spuds', 'yukon gold'],
    'soy sauce': ['soy', 'soy sauce', 'shoyu'],
    'rice': ['rice', 'jasmine rice', 'basmati'],
    'garlic': ['garlic', 'garlic cloves']
}

lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = str(text).lower()
    for key, synonyms in ingredient_synonyms.items():
        for synonym in synonyms:
            text = re.sub(r'\b(?:%s)\b' % '|'.join(map(re.escape, synonyms)), key, text)
    text = re.sub(r'[^\w\s,-]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

# --------------------------
# Huvudtest
# --------------------------
def main():
    # Kontrollera filer
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Datafil {DATA_FILE} saknas")
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Modellfil {MODEL_FILE} saknas")

    # Ladda data
    df = pd.read_csv(DATA_FILE)
    df['processed'] = df['ingredients'].apply(preprocess) + ' ' + df['tag_name'].apply(preprocess)
    
    # Ladda modell
    pipeline = joblib.load(MODEL_FILE)
    
    # Testa med 3 exempel
    test_queries = [
        ("chicken, rice, teriyaki sauce", ["chicken", "rice", "teriyaki sauce"]),
        ("beef and potatoes", ["beef", "potato"]),
        ("pasta with garlic", ["pasta", "garlic"])
    ]
    
    for query, ingredients in test_queries:
        print(f"\n🔍 Testar: '{query}'")
        try:
            # Generera rekommendationer
            query_vec = pipeline['tfidf'].transform([preprocess(query)])
            distances, indices = pipeline['knn'].kneighbors(query_vec)
            
            # Utvärdera resultat
            results = df.iloc[indices[0]]
            correct = 0
            for _, row in results.iterrows():
                recipe_ings = set(preprocess(row['ingredients']).split())
                if all(ing in recipe_ings for ing in ingredients):
                    correct += 1
            
            print(f"Resultat: {correct}/{len(results)} korrekta")
            print("Topp 3 rekommendationer:")
            print(results[['name', 'ingredients']].head(3))
            
        except Exception as e:
            print(f"Fel vid test: {str(e)}")

if __name__ == "__main__":
    main()
