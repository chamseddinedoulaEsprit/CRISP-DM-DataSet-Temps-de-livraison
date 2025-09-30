from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

app = Flask(__name__)

# -----------------------------
# Dictionnaires de mapping pour les colonnes catégorielles
# -----------------------------
STORE_CATEGORIES_MAP = {
    'Restaurant': 0, 'Grocery': 1, 'Pharmacy': 2, 'Retail': 3,
    'Fast Food': 4, 'Coffee Shop': 5, 'Convenience Store': 6,
    'Electronics': 7, 'Clothing': 8
}

WEATHER_CONDITIONS_MAP = {
    'Clear': 0, 'Cloudy': 1, 'Rainy': 2, 'Snowy': 3,
    'Foggy': 4, 'Stormy': 5
}

PRICE_RANGES_MAP = {
    'Budget': 0, 'Mid-range': 1, 'Premium': 2, 'Luxury': 3
}

# -----------------------------
# Charger les modèles
# -----------------------------
def load_models():
    try:
        lightgbm_model = joblib.load('lightgbm_model.pkl')
        dbscan_model = joblib.load('dbscan_model.pkl')
        print("✅ Tous les fichiers de modèles sont présents.")
        return lightgbm_model, dbscan_model
    except FileNotFoundError:
        print("❌ Modèles manquants.")
        return None, None

lightgbm_model, dbscan_model = load_models()

# -----------------------------
# Routes Flask
# -----------------------------
@app.route('/')
def index():
    return render_template('index.html',
                           store_categories=list(STORE_CATEGORIES_MAP.keys()),
                           weather_conditions=list(WEATHER_CONDITIONS_MAP.keys()),
                           price_ranges=list(PRICE_RANGES_MAP.keys()))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Construire le DataFrame avec toutes les colonnes en float
        features = pd.DataFrame([{
            'market_id': float(data.get('market_id', 1)),
            'store_id': float(data.get('store_id', 1)),
            'store_primary_category': float(STORE_CATEGORIES_MAP.get(
                data.get('store_primary_category', 'Restaurant'), 0)),
            'total_items': float(data.get('total_items', 1)),
            'subtotal': float(data.get('subtotal', 10.0)),
            'num_distinct_items': float(data.get('num_distinct_items', 1)),
            'total_onshift_partners': float(data.get('total_onshift_partners', 1)),
            'day_of_week_numeric': float(data.get('day_of_week_numeric', 1)),
            'hour_of_day': float(data.get('hour_of_day', 12)),
            'price_range': float(PRICE_RANGES_MAP.get(data.get('price_range', 'Mid-range'), 1)),
            'weather_condition': float(WEATHER_CONDITIONS_MAP.get(data.get('weather_condition', 'Clear'), 0)),
            'temperature': float(data.get('temperature', 20.0))
        }])
        
        # Conversion en float32
        features_encoded = preprocess_features(features)
        
        # Vérification des colonnes pour LightGBM
        if lightgbm_model is not None:
            model_columns = list(lightgbm_model.feature_name_)
            if list(features_encoded.columns) != model_columns:
                return jsonify({
                    'success': False,
                    'error': f"Colonnes incorrectes pour LightGBM. Attendu: {model_columns}, reçu: {list(features_encoded.columns)}"
                }), 400
        
        predictions = {}

        # -----------------------------
        # Prédiction LightGBM
        # -----------------------------
        if lightgbm_model is not None:
            try:
                delivery_time_pred = lightgbm_model.predict(features_encoded)[0]
                predictions['delivery_time_minutes'] = round(float(delivery_time_pred), 2)
                predictions['delivery_time_formatted'] = format_delivery_time(delivery_time_pred)
            except Exception as e:
                predictions['delivery_time_error'] = f"Erreur LightGBM: {str(e)}"

        # -----------------------------
        # Prédiction DBSCAN
        # -----------------------------
        if dbscan_model is not None:
            try:
                cluster_pred = dbscan_model.fit_predict(features_encoded)[0]
                predictions['cluster'] = int(cluster_pred)
                predictions['cluster_interpretation'] = interpret_cluster(cluster_pred)
            except Exception as e:
                predictions['cluster_error'] = f"Erreur DBSCAN: {str(e)}"

        return jsonify({'success': True, 'predictions': predictions})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# -----------------------------
# Préprocessing
# -----------------------------
def preprocess_features(df):
    """Toutes les colonnes sont déjà float encodé, on force juste float32"""
    return df.astype(np.float32)

# -----------------------------
# Fonctions utilitaires
# -----------------------------
def format_delivery_time(minutes):
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours}h {mins}min" if hours > 0 else f"{mins}min"

def interpret_cluster(cluster_id):
    interpretations = {
        -1: "Commande atypique (outlier)",
        0: "Commande standard",
        1: "Commande rapide",
        2: "Commande complexe",
        3: "Commande premium"
    }
    return interpretations.get(cluster_id, f"Cluster {cluster_id}")

# -----------------------------
# Health check
# -----------------------------
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'lightgbm_loaded': lightgbm_model is not None,
        'dbscan_loaded': dbscan_model is not None,
        'timestamp': datetime.now().isoformat()
    })

# -----------------------------
# Lancer l'application
# -----------------------------
if __name__ == '__main__':
    print("🚀 Démarrage de l'application de prédiction de livraison")
    print("="*60)
    print("✅ Tous les fichiers de modèles sont présents." if lightgbm_model and dbscan_model else "❌ Modèles manquants")
    print("🌐 Serveur: http://0.0.0.0:5000")
    print("🔧 Mode debug: True")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000)
