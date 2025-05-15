from flask import Flask, request, render_template
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# Sökvägar
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "recipes_with_ingredients_and_tags.csv")
TFIDF_MODEL_PATH = os.path.join(BASE_DIR, "models", "tfidf_model.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "models", "tfidf_matrix.pkl")

# Ladda data och modeller
try:
    df = pd.read_csv(DATA_PATH)
    tfidf = joblib.load(TFIDF_MODEL_PATH)
    tfidf_matrix = joblib.load(TFIDF_MATRIX_PATH)
except Exception as e:
    print(f"Fel vid inläsning: {e}")
    exit()

# Rekommendationsfunktion
def get_recommendations(query, top_n=10):
    processed_query = ' '.join(query.lower().replace(',', ' ').split())
    query_vec = tfidf.transform([processed_query])
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[-top_n:][::-1]
    return df.iloc[top_indices][['name', 'ingredients', 'description', 'thumbnail_url']].fillna("")

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['ingredients']
        results = get_recommendations(user_input)
        return render_template('results.html', recommendations=results)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
