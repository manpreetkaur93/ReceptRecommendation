from flask import Flask, request, render_template, jsonify
import pandas as pd
import json
import joblib
import os
import nltk
from dotenv import load_dotenv
from RAG_Pipeline import RecipeRAG, RecipeChatbot, TextPreprocessor  # Lägg till explicit import

load_dotenv()

app = Flask(__name__)

# ---- INITIERING AV RAG-SYSTEM ---- #
def initialize_rag_system():
    """Initiera RAG-systemet vid appstart"""
    try:
        rag = RecipeRAG(
            model_path="models/full_pipeline.pkl",
            data_path="recipes_with_ingredients_and_tags.csv"
        )
        return RecipeChatbot(rag)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize RAG system: {str(e)}")

chatbot = initialize_rag_system()

# ---- DATAHANTERING ---- #
def load_resources():
    """Ladda databasresurser och pipeline"""
    resources = {}
    try:
        resources['df'] = pd.read_csv("recipes_with_ingredients_and_tags.csv")
        
        with open("ingredient_and_instructions.json", 'r', encoding='utf-8') as f:
            resources['instructions'] = json.load(f)
        
        loaded = joblib.load("models/full_pipeline.pkl")
        resources['pipeline'] = {
            'preprocessor': loaded['preprocessor'],
            'tfidf': loaded['tfidf'],
            'knn': loaded['knn']
        }
        
    except Exception as e:
        raise RuntimeError(f"Resource loading failed: {str(e)}")
    
    return resources

app_resources = load_resources()

# ... resten av din app.py förblir oförändrad ...


# ---- ROUTES ---- #
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            user_input = request.form['ingredients']
            results = get_recommendations(user_input)
            return render_template(
                'results.html',
                recommendations=results.to_dict(orient='records'),
                query=user_input
            )
        except Exception as e:
            return render_template('error.html', error=str(e))
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def handle_chat():
    try:
        user_message = request.json.get('message', '')
        if not user_message.strip():
            return jsonify({"error": "Empty message"}), 400
            
        bot_response = chatbot.handle_message(user_message)
        return jsonify({"response": bot_response})
    
    except Exception as e:
        app.logger.error(f"Chat error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


# ---- HJÄLPFUNKTIONER ---- #
def get_recipe_instructions(slug):
    """Hämta instruktioner för ett recept"""
    return [
        step["display_text"] 
        for step in app_resources['instructions'].get(slug, {}).get('instructions', [])
        if "display_text" in step
    ]

def get_recommendations(query, top_n=10):
    """Generera rekommendationer baserat på ingredienser"""
    try:
        # Förprocessera query
        processed_query = app_resources['pipeline']['preprocessor'].transform([query])
        
        # Hämta resultat från modellen
        query_vec = app_resources['pipeline']['tfidf'].transform(processed_query)
        distances, indices = app_resources['pipeline']['knn'].kneighbors(query_vec, n_neighbors=top_n)
        
        # Bearbeta resultat
        results = app_resources['df'].iloc[indices[0]].copy()
        results['similarity'] = 1 - distances[0]
        results['instructions'] = results['slug'].apply(
            lambda x: get_recipe_instructions(x) 
            if app_resources['df'].loc[app_resources['df']['slug'] == x, 'has_instructions'].any()
            else None
        )
        
        return results[['name', 'ingredients', 'thumbnail_url', 'similarity', 'instructions']]
    
    except Exception as e:
        raise RuntimeError(f"Recommendation failed: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
