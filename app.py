from flask import Flask, request, render_template
import pandas as pd
import json
import joblib
import os

app = Flask(__name__)

# Sökvägar
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "recipes_with_ingredients_and_tags.csv")
INSTRUCTIONS_PATH = os.path.join(BASE_DIR, "ingredient_and_instructions.json")
TFIDF_PATH = os.path.join(BASE_DIR, "models", "tfidf_model.pkl")
KNN_PATH = os.path.join(BASE_DIR, "models", "knn_model.pkl")

# Ladda data och modeller
try:
    # Ladda receptdata
    df = pd.read_csv(DATA_PATH)
    
    # Ladda instruktionsdata
    with open(INSTRUCTIONS_PATH, 'r', encoding='utf-8') as f:
        instructions_data = json.load(f)
    
    # Ladda modeller
    tfidf = joblib.load(TFIDF_PATH)
    knn = joblib.load(KNN_PATH)
    
    print("✅ Data och modeller laddade")
except Exception as e:
    print(f"Fel vid inläsning: {e}")
    exit()

def get_recipe_instructions(slug):
    """Hämta instruktioner för ett recept baserat på slug"""
    if slug in instructions_data:
        steps = instructions_data[slug].get("instructions", [])
        return [step["display_text"] for step in steps if "display_text" in step]
    return None

def get_recommendations(query, top_n=10):
    """Generera rekommendationer med KNN"""
    # Förbearbeta och transformera query
    processed_query = ' '.join(query.lower().replace(',', ' ').split())
    query_vec = tfidf.transform([processed_query])
    
    # Hämta närmaste grannar
    distances, indices = knn.kneighbors(query_vec, n_neighbors=top_n)
    
    # Skapa resultatlista
    recommendations = []
    for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
        recipe = df.iloc[idx].copy()
        recipe['similarity'] = 1 - distance  # Konvertera avstånd till likhet
        recipe['instructions'] = get_recipe_instructions(recipe['slug']) if recipe['has_instructions'] else None
        recommendations.append(recipe)
    
    return pd.DataFrame(recommendations)[['name', 'ingredients', 'thumbnail_url', 'similarity', 'instructions']]

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
