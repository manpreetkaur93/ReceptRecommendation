from flask import Flask, request, render_template
import pandas as pd
import json
import joblib
import os
import re
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('punkt')

app = Flask(__name__)

# ---- SAMMA PREPROCESSING SOM I NOTEBOOK ---- #
ingredient_synonyms = {
    'chicken': ['poultry', 'hen', 'chicken breast'],
    'beef': ['ground beef', 'sirloin', 'roast beef'],
    'potato': ['potatoes', 'spuds', 'yukon gold']
}

lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = str(text).lower()
    for key, synonyms in ingredient_synonyms.items():
        for synonym in synonyms:
            text = re.sub(r'\b' + re.escape(synonym) + r'\b', key, text)
    text = re.sub(r'[^\w\s,-]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)
# -------------------------------------------- #

# Sökvägar
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "recipes_with_ingredients_and_tags.csv")
INSTRUCTIONS_PATH = os.path.join(BASE_DIR, "ingredient_and_instructions.json")
PIPELINE_PATH = os.path.join(BASE_DIR, "models", "full_pipeline.pkl")

# Ladda data och modeller
try:
    df = pd.read_csv(DATA_PATH)
    
    with open(INSTRUCTIONS_PATH, 'r', encoding='utf-8') as f:
        instructions_data = json.load(f)
    
    pipeline = joblib.load(PIPELINE_PATH)
    print("✅ Data och modeller laddade")
except Exception as e:
    print(f"Fel vid inläsning: {e}")
    exit()

def get_recipe_instructions(slug):
    if slug in instructions_data:
        steps = instructions_data[slug].get("instructions", [])
        return [step["display_text"] for step in steps if "display_text" in step]
    return None

def get_recommendations(query, top_n=10):
    processed_query = preprocess(query)
    query_vec = pipeline['tfidf'].transform([processed_query])
    distances, indices = pipeline['knn'].kneighbors(query_vec, n_neighbors=top_n)
    
    results = df.iloc[indices[0]]
    results['similarity'] = 1 - distances[0]
    results['instructions'] = results['slug'].apply(
        lambda x: get_recipe_instructions(x) if df[df['slug'] == x]['has_instructions'].any() else None
    )
    
    return results[['name', 'ingredients', 'thumbnail_url', 'similarity', 'instructions']]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['ingredients']
        try:
            results = get_recommendations(user_input)
            return render_template('results.html', 
                                recommendations=results.to_dict(orient='records'),
                                query=user_input)
        except Exception as e:
            return render_template('error.html', error=str(e))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
