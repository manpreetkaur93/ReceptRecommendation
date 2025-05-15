from flask import Flask, request, render_template
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import os
# Add to top of app.py to check versions
import sklearn
print(f"scikit-learn version: {sklearn.__version__}")

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
    
    # Add these debug lines
    print(f"Model path: {TFIDF_MODEL_PATH}")
    print(f"Matrix path: {TFIDF_MATRIX_PATH}")
    print(f"IDF vector exists: {hasattr(tfidf, 'idf_')}")
    print(f"TF-IDF type: {type(tfidf).__name__}")
    print(f"Matrix shape: {tfidf_matrix.shape}")
    
except Exception as e:
    print(f"Error loading models: {e}")
    exit()


# Rekommendationsfunktion
def get_recommendations(query, top_n=10):
    try:
        # Clean the query text
        processed_query = ' '.join(query.lower().replace(',', ' ').split())
        
        # Verify TF-IDF vectorizer status
        print(f"Before transform - IDF vector exists: {hasattr(tfidf, 'idf_')}")
        
        # Transform the query
        query_vec = tfidf.transform([processed_query])
        
        # Calculate similarity
        sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = sim_scores.argsort()[-top_n:][::-1]
        
        # Return results
        return df.iloc[top_indices][['name', 'ingredients', 'description', 'thumbnail_url']].fillna("")
    except Exception as e:
        print(f"Error in get_recommendations: {e}")
        # Return empty DataFrame with correct columns as fallback
        return pd.DataFrame(columns=['name', 'ingredients', 'description', 'thumbnail_url'])


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
