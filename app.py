from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# -----------------------------
# Dictionnaires de mapping pour les colonnes catégorielles
# -----------------------------
STORE_IDS = [
    'd43ab110ab2489d6b9b2caa394bf920f', '757b505cfd34c64c85ca5b5690ee5293',
    'faacbcd5bf1d018912c116bf2783e9a1', '45c48cce2e2d7fbdea1afc51c7c6ad26',
    'c9f0f895fb98ab9159f51fd0297e236d', 'dc5689792e08eb2e219dce49e64c885b',
    'fbd7939d674997cdb4692d34de8633c4', 'f7177163c833dff4b38fc8d2872f1ec6',
    'f15d337c70078947cfe1b5d6f0ed3f13', 'a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6',
    'b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7', 'c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8'
]
STORE_ID_MAP = {store_id: idx for idx, store_id in enumerate(STORE_IDS)}

STORE_CATEGORIES = [
    'american', 'pizza', 'mexican', 'burger', 'japanese',
    'sandwich', 'chinese', 'dessert', 'italian', 'chicken', 'cafe'
]
STORE_CATEGORIES_MAP = {category: idx for idx, category in enumerate(STORE_CATEGORIES)}

PRICE_RANGES = ['Budget', 'Mid-range', 'Premium', 'Luxury']
PRICE_RANGES_MAP = {price: idx for idx, price in enumerate(PRICE_RANGES)}

WEATHER_CONDITIONS = ['Clear', 'Cloudy', 'Rainy', 'Snowy', 'Foggy', 'Stormy']
WEATHER_CONDITIONS_MAP = {condition: idx for idx, condition in enumerate(WEATHER_CONDITIONS)}

MARKET_ID_TO_CITY = {
    1.0: ('Delhi NCR', (19.07, 72.87)),
    2.0: ('Mumbai MMR', (28.61, 77.21)),
    3.0: ('Bengaluru', (12.97, 77.59)),
    4.0: ('Chennai', (13.08, 80.27)),
    5.0: ('Hyderabad', (17.38, 78.48)),
    6.0: ('Kolkata', (22.57, 88.36))
}

STORE_MAPPING = {
    'd43ab110ab2489d6b9b2caa394bf920f': {'name': 'Pizza Hut Delhi', 'city': 'Delhi NCR', 'category': 'pizza'},
    '757b505cfd34c64c85ca5b5690ee5293': {'name': 'Burger King Mumbai', 'city': 'Mumbai MMR', 'category': 'burger'},
    'faacbcd5bf1d018912c116bf2783e9a1': {'name': "Domino's Bangalore", 'city': 'Bengaluru', 'category': 'pizza'},
    '45c48cce2e2d7fbdea1afc51c7c6ad26': {'name': 'McDonald Chennai', 'city': 'Chennai', 'category': 'burger'},
    'c9f0f895fb98ab9159f51fd0297e236d': {'name': 'KFC Hyderabad', 'city': 'Hyderabad', 'category': 'chicken'},
    'dc5689792e08eb2e219dce49e64c885b': {'name': 'Subway Kolkata', 'city': 'Kolkata', 'category': 'sandwich'},
    'fbd7939d674997cdb4692d34de8633c4': {'name': 'Cafe Coffee Day Delhi', 'city': 'Delhi NCR', 'category': 'cafe'},
    'f7177163c833dff4b38fc8d2872f1ec6': {'name': 'Pizza Corner Mumbai', 'city': 'Mumbai MMR', 'category': 'pizza'},
    'f15d337c70078947cfe1b5d6f0ed3f13': {'name': 'Burger Hub Bangalore', 'city': 'Bengaluru', 'category': 'burger'},
    'a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6': {'name': 'Taco Bell Delhi', 'city': 'Delhi NCR', 'category': 'mexican'},
    'b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7': {'name': 'Sushi Place Mumbai', 'city': 'Mumbai MMR', 'category': 'japanese'},
    'c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8': {'name': 'Pasta House Bangalore', 'city': 'Bengaluru', 'category': 'italian'}
}

# -----------------------------
# Charger les modèles et le scaler
# -----------------------------
def load_models_and_scaler():
    try:
        lightgbm_model = joblib.load('lightgbm_model.pkl')
        dbscan_model = joblib.load('dbscan_model.pkl')
        try:
            scaler = joblib.load('scaler.pkl')
            print("✅ Tous les fichiers (modèles et scaler) sont présents.")
            return lightgbm_model, dbscan_model, scaler
        except FileNotFoundError:
            print("⚠️ Scaler non trouvé (scaler.pkl manquant). Prédictions sans normalisation.")
            return lightgbm_model, dbscan_model, None
        except Exception as e:
            print(f"❌ Erreur lors du chargement de scaler.pkl: {str(e)}")
            return lightgbm_model, dbscan_model, None
    except FileNotFoundError as e:
        print(f"❌ Modèle manquant: {str(e)}")
        return None, None, None

lightgbm_model, dbscan_model, scaler = load_models_and_scaler()

# -----------------------------
# Routes Flask
# -----------------------------
@app.route('/')
def index():
    if lightgbm_model is None or dbscan_model is None:
        return jsonify({'error': 'Modèles non chargés. Vérifiez les fichiers lightgbm_model.pkl et dbscan_model.pkl.'}), 500
    return render_template('index.html',
                           store_ids=STORE_IDS,
                           store_mapping=STORE_MAPPING,
                           store_categories=STORE_CATEGORIES,
                           price_ranges=PRICE_RANGES,
                           weather_conditions=WEATHER_CONDITIONS,
                           market_ids=MARKET_ID_TO_CITY)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Input data:", data)  # Log input data
        
        # Obtenir le market_id à partir du store_id
        store_id = data.get('store_id', STORE_IDS[0])
        market_id = None
        for mid, (city, _) in MARKET_ID_TO_CITY.items():
            if STORE_MAPPING.get(store_id, {}).get('city') == city:
                market_id = mid
                break
        if market_id is None:
            market_id = float(data.get('market_id', 1.0))
        
        # Construire le DataFrame avec les colonnes attendues
        features = pd.DataFrame([{
            'market_id': market_id,
            'store_id': float(STORE_ID_MAP.get(store_id, 0)),
            'store_primary_category': float(STORE_CATEGORIES_MAP.get(
                data.get('store_primary_category', STORE_MAPPING.get(store_id, {}).get('category', 'american')), 0)),
            'total_items': float(data.get('total_items', 1)),
            'subtotal': float(data.get('subtotal', 10.0)),
            'num_distinct_items': float(data.get('num_distinct_items', 1)),
            'total_onshift_partners': float(data.get('total_onshift_partners', 1)),
            'day_of_week_numeric': float(data.get('day_of_week_numeric', 1)),
            'hour_of_day': float(data.get('hour_of_day', 12)),
            'price_range': float(PRICE_RANGES_MAP.get(data.get('price_range', 'Mid-range'), 1)),
            'weather_condition': float(WEATHER_CONDITIONS_MAP.get(
                data.get('weather_condition', 'Clear'), 0)),
            'temperature': float(data.get('temperature', 20.0))
        }])
        
        # Conversion en float32 et normalisation
        features_encoded = preprocess_features(features)
        print("Features encoded:", features_encoded.to_dict())  # Log features
        
        # Vérification des colonnes pour LightGBM
        if lightgbm_model is not None:
            model_columns = [
                'market_id', 'store_id', 'store_primary_category', 'total_items',
                'subtotal', 'num_distinct_items', 'total_onshift_partners',
                'day_of_week_numeric', 'hour_of_day', 'price_range',
                'weather_condition', 'temperature'
            ]
            if list(features_encoded.columns) != model_columns:
                return jsonify({
                    'success': False,
                    'error': f"Colonnes incorrectes pour LightGBM. Attendu: {model_columns}, reçu: {list(features_encoded.columns)}"
                }), 400
        
        predictions = {}

        # Prédiction LightGBM
        if lightgbm_model is not None:
            try:
                delivery_time_pred = lightgbm_model.predict(features_encoded)[0]
                predictions['delivery_time_minutes'] = round(float(delivery_time_pred), 2)
                predictions['delivery_time_formatted'] = format_delivery_time(delivery_time_pred)
            except Exception as e:
                predictions['delivery_time_error'] = f"Erreur LightGBM: {str(e)}"

        # Prédiction DBSCAN
        if dbscan_model is not None:
            try:
                cluster_pred = dbscan_model.fit_predict(features_encoded)[0]
                print("DBSCAN cluster:", cluster_pred)  # Log cluster output
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
    """Scale features using the loaded scaler and convert to float32"""
    if scaler is None:
        print("⚠️ Aucun scaler disponible. Utilisation des données non normalisées (résultats imprécis possibles).")
        return df.astype(np.float32)
    scaled_df = pd.DataFrame(scaler.transform(df), columns=df.columns)
    return scaled_df.astype(np.float32)

# -----------------------------
# Fonctions utilitaires
# -----------------------------
def format_delivery_time(minutes):
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours}h {mins}min" if hours > 0 else f"{mins}min"

def interpret_cluster(cluster_id):
    interpretations = {
        -1: "Commande standard",
        0: "Commande standard",
        1: "Commande rapide"
    }
    return interpretations.get(cluster_id, f"Cluster inconnu {cluster_id}")

# -----------------------------
# Health check
# -----------------------------
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'lightgbm_loaded': lightgbm_model is not None,
        'dbscan_loaded': dbscan_model is not None,
        'scaler_loaded': scaler is not None,
        'timestamp': datetime.now().isoformat()
    })

# -----------------------------
# Lancer l'application
# -----------------------------
if __name__ == '__main__':
    print("🚀 Démarrage de l'application de prédiction de livraison")
    print("="*60)
    print("✅ Tous les fichiers (modèles et scaler) sont présents." if lightgbm_model and dbscan_model and scaler else "⚠️ Fichiers manquants")
    print("🌐 Serveur: http://0.0.0.0:5000")
    print("🔧 Mode debug: True")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000)