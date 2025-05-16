from flask import Flask, request, render_template
import pandas as pd
import json
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# Load data and models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "recipes_with_ingredients_and_tags.csv")
INSTRUCTIONS_PATH = os.path.join(BASE_DIR, "ingredient_and_instructions.json")
TFIDF_MODEL_PATH = os.path.join(BASE_DIR, "models", "tfidf_model.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "models", "tfidf_matrix.pkl")

try:
    # Load recipe data
    df = pd.read_csv(DATA_PATH)
    
    # Load instructions data
    with open(INSTRUCTIONS_PATH, 'r', encoding='utf-8') as f:
        instructions_data = json.load(f)
    
    # Load models
    tfidf = joblib.load(TFIDF_MODEL_PATH)
    tfidf_matrix = joblib.load(TFIDF_MATRIX_PATH)
    print("Data and models loaded successfully")
except Exception as e:
    print(f"Error loading data or models: {e}")
    exit()

def get_recipe_instructions(slug):
    """Get instructions for a recipe by slug, returns a list of strings."""
    if slug in instructions_data:
        steps = instructions_data[slug].get("instructions", [])
        # Extract only the display_text from each step dict
        return [step["display_text"] for step in steps if "display_text" in step]
    return None

def get_recommendations(query, top_n=10):
    """Get recipe recommendations based on ingredients query"""
    processed_query = ' '.join(query.lower().replace(',', ' ').split())
    query_vec = tfidf.transform([processed_query])
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[-top_n:][::-1]
    
    # Get the basic recipe data
    results = df.iloc[top_indices][['name', 'ingredients', 'description', 'thumbnail_url', 'slug', 'has_instructions']].fillna("")
    
    # Convert to list of dictionaries for template
    recipes = results.to_dict(orient="records")
    
    # Add instructions to each recipe
    for recipe in recipes:
        if recipe['has_instructions']:
            recipe['instructions'] = get_recipe_instructions(recipe['slug'])
        else:
            recipe['instructions'] = None
    
    return recipes

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['ingredients']
        recommendations = get_recommendations(user_input)
        return render_template('results.html', recommendations=recommendations)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
