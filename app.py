from flask import Flask, request, render_template, jsonify
import pandas as pd
import json
import joblib
import os
import re
import nltk
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv
from RAG_Pipeline import TextPreprocessor, RecipeRAG, RecipeChatbot

# Ladda miljövariabler
load_dotenv()

# Konfigurera NLTK
nltk.download('wordnet')
nltk.download('punkt')

app = Flask(__name__)

# ---- INITIERING AV MODELLER ---- #
def initialize_rag_system():
    """Initiera RAG-systemet en gång vid appstart"""
    rag = RecipeRAG(
        model_path="models/full_pipeline.pkl",
        data_path="recipes_with_ingredients_and_tags.csv"
    )
    return RecipeChatbot(rag)

chatbot = initialize_rag_system()

# ---- PREPROCESSING ---- #
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
            text = re.sub(rf'\b{re.escape(synonym)}\b', key, text)
    text = re.sub(r'[^\w\s,-]', '', text)
    tokens = nltk.word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(token) for token in tokens])

# ---- DATAHANTERING ---- #
def load_resources():
    """Ladda alla nödvändiga resurser"""
    resources = {
        'df': pd.read_csv("recipes_with_ingredients_and_tags.csv"),
        'instructions': None,
        'pipeline': None
    }
    
    with open("ingredient_and_instructions.json", 'r', encoding='utf-8') as f:
        resources['instructions'] = json.load(f)
    
    resources['pipeline'] = joblib.load("models/full_pipeline.pkl")
    return resources

# Initiera resurser vid appstart
app_resources = load_resources()

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
            app.logger.error(f"Error in index route: {str(e)}")
            return render_template('error.html', error=str(e))
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def handle_chat():
    try:
        user_message = request.json.get('message', '')
        if not user_message:
            return jsonify({"error": "Empty message"}), 400
            
        bot_response = chatbot.handle_message(user_message)
        return jsonify({"response": bot_response})
    
    except Exception as e:
        app.logger.error(f"Chat error: {str(e)}")
        return jsonify({"error": "Failed to process request"}), 500

# ---- HJÄLPFUNKTIONER ---- #
def get_recipe_instructions(slug):
    return [
        step["display_text"] 
        for step in app_resources['instructions'].get(slug, {}).get('instructions', [])
        if "display_text" in step
    ]

def get_recommendations(query, top_n=10):
    """Hämta rekommendationer från modellen"""
    processed_query = preprocess(query)
    pipeline = app_resources['pipeline']
    
    # Transformera och hämta resultat
    query_vec = pipeline['tfidf'].transform([processed_query])
    distances, indices = pipeline['knn'].kneighbors(query_vec, n_neighbors=top_n)
    
    # Bearbeta resultat
    results = app_resources['df'].iloc[indices[0]]
    results['similarity'] = 1 - distances[0]
    results['instructions'] = results['slug'].apply(
        lambda x: get_recipe_instructions(x) 
        if app_resources['df'].loc[app_resources['df']['slug'] == x, 'has_instructions'].any()
        else None
    )
    
    return results[['name', 'ingredients', 'thumbnail_url', 'similarity', 'instructions']]

if __name__ == '__main__':
    app.run(debug=True)
