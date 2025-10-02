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
# Charger les modèles, le scaler et le PCA
# -----------------------------
def load_models_and_scaler():
    try:
        lightgbm_model = joblib.load('lightgbm_model.pkl')
        dbscan_model = joblib.load('dbscan_model.pkl')
        pca_model = joblib.load('pca_model.pkl')
        try:
            scaler = joblib.load('scaler.pkl')
            print("✅ Tous les fichiers (modèles, PCA et scaler) sont présents.")
            return lightgbm_model, dbscan_model, pca_model, scaler
        except FileNotFoundError:
            print("⚠️ Scaler non trouvé (scaler.pkl manquant). Prédictions sans normalisation.")
            return lightgbm_model, dbscan_model, pca_model, None
        except Exception as e:
            print(f"❌ Erreur lors du chargement de scaler.pkl: {str(e)}")
            return lightgbm_model, dbscan_model, pca_model, None
    except FileNotFoundError as e:
        print(f"❌ Fichier modèle manquant: {str(e)}")
        # Retourner None pour tous les modèles si l'un d'eux est manquant
        return None, None, None, None

lightgbm_model, dbscan_model, pca_model, scaler = load_models_and_scaler()

# -----------------------------
# Routes Flask
# -----------------------------
@app.route('/')
def index():
    if lightgbm_model is None or dbscan_model is None or pca_model is None:
        return jsonify({'error': 'Modèles non chargés. Vérifiez les fichiers lightgbm_model.pkl, dbscan_model.pkl et pca_model.pkl.'}), 500
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
        
        # Construire le DataFrame avec les colonnes attendues pour LightGBM (10 features)
        features_lightgbm = pd.DataFrame([{
            'market_id': market_id,
            'store_id': float(STORE_ID_MAP.get(store_id, 0)),
            'store_primary_category': float(STORE_CATEGORIES_MAP.get(
                data.get('store_primary_category', STORE_MAPPING.get(store_id, {}).get('category', 'american')), 0)),
            'total_items': float(data.get('total_items', 1)),
            'subtotal': float(data.get('subtotal', 10.0)),
            'num_distinct_items': float(data.get('num_distinct_items', 1)),
            'total_onshift_partners': float(data.get('total_onshift_partners', 1)),
            'hour_of_day': float(data.get('hour_of_day', 12)),
            'price_range': float(PRICE_RANGES_MAP.get(data.get('price_range', 'Mid-range'), 1)),
            'temperature': float(data.get('temperature', 20.0))
        }])
        
        # Construire le DataFrame avec les colonnes attendues pour PCA/DBSCAN (13 features)
        features_pca_dbscan = pd.DataFrame([{
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
            'temperature': float(data.get('temperature', 20.0)),
            'is_weekend': float(1 if data.get('day_of_week_numeric', 1) in [5, 6] else 0)  # 5=Saturday, 6=Sunday
        }])
        
        print(f"LightGBM features shape: {features_lightgbm.shape}")
        print(f"PCA/DBSCAN features shape: {features_pca_dbscan.shape}")
        
        predictions = {}

        # Prédiction LightGBM
        if lightgbm_model is not None:
            try:
                # Préprocessing pour LightGBM
                features_lightgbm_processed = preprocess_features(features_lightgbm, model_type='lightgbm')
                print("LightGBM features encoded:", features_lightgbm_processed.to_dict())
                
                # Vérification des colonnes pour LightGBM
                expected_lightgbm_columns = [
                    'market_id', 'store_id', 'store_primary_category', 'total_items',
                    'subtotal', 'num_distinct_items', 'total_onshift_partners',
                    'hour_of_day', 'price_range', 'temperature'
                ]
                
                if list(features_lightgbm_processed.columns) != expected_lightgbm_columns:
                    print(f"⚠️ Ajustement des colonnes LightGBM. Attendu: {expected_lightgbm_columns}")
                    features_lightgbm_processed = features_lightgbm_processed[expected_lightgbm_columns]
                
                delivery_time_pred = lightgbm_model.predict(features_lightgbm_processed)[0]
                predictions['delivery_time_minutes'] = round(float(delivery_time_pred), 2)
                predictions['delivery_time_formatted'] = format_delivery_time(delivery_time_pred)
                predictions['lightgbm_features_used'] = len(features_lightgbm_processed.columns)
            except Exception as e:
                predictions['delivery_time_error'] = f"Erreur LightGBM: {str(e)}"

        # Prédiction DBSCAN avec réduction PCA
        if dbscan_model is not None and pca_model is not None:
            try:
                # Préprocessing pour PCA/DBSCAN
                features_pca_dbscan_processed = preprocess_features(features_pca_dbscan, model_type='pca_dbscan')
                print("PCA/DBSCAN features encoded:", features_pca_dbscan_processed.to_dict())
                
                # Vérification des colonnes pour PCA
                expected_pca_columns = [
                    'market_id', 'store_id', 'store_primary_category', 'total_items',
                    'subtotal', 'num_distinct_items', 'total_onshift_partners',
                    'day_of_week_numeric', 'hour_of_day', 'price_range',
                    'weather_condition', 'temperature', 'is_weekend'
                ]
                
                if list(features_pca_dbscan_processed.columns) != expected_pca_columns:
                    print(f"⚠️ Ajustement des colonnes PCA. Attendu: {expected_pca_columns}")
                    # S'assurer que nous avons les bonnes colonnes dans le bon ordre
                    features_pca_dbscan_processed = features_pca_dbscan_processed.reindex(columns=expected_pca_columns)
                
                # Réduction de dimensionnalité avec PCA
                features_pca = pca_model.transform(features_pca_dbscan_processed)
                print(f"PCA transformation: {features_pca_dbscan_processed.shape} -> {features_pca.shape}")
                
                # Prédiction du cluster avec DBSCAN sur les données réduites
                cluster_pred = dbscan_model.fit_predict(features_pca)[0]
                print("DBSCAN cluster:", cluster_pred)
                predictions['cluster'] = int(cluster_pred)
                predictions['cluster_interpretation'] = interpret_cluster(cluster_pred)
                predictions['pca_components'] = features_pca.shape[1]
                predictions['pca_dbscan_features_used'] = len(features_pca_dbscan_processed.columns)
            except Exception as e:
                predictions['cluster_error'] = f"Erreur DBSCAN/PCA: {str(e)}"
        elif dbscan_model is not None and pca_model is None:
            try:
                # Fallback: DBSCAN sans PCA
                features_dbscan_fallback = preprocess_features(features_pca_dbscan, model_type='pca_dbscan')
                cluster_pred = dbscan_model.fit_predict(features_dbscan_fallback)[0]
                print("DBSCAN cluster (sans PCA):", cluster_pred)
                predictions['cluster'] = int(cluster_pred)
                predictions['cluster_interpretation'] = interpret_cluster(cluster_pred)
                predictions['pca_warning'] = "PCA non utilisé - clustering sur données originales"
            except Exception as e:
                predictions['cluster_error'] = f"Erreur DBSCAN (sans PCA): {str(e)}"

        return jsonify({'success': True, 'predictions': predictions})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# -----------------------------
# Préprocessing
# -----------------------------
def preprocess_features(df, model_type='lightgbm'):
    """Scale features using the loaded scaler and convert to float32"""
    if scaler is None:
        print(f"⚠️ Aucun scaler disponible pour {model_type}. Utilisation des données non normalisées.")
        return df.astype(np.float32)
    
    # Appliquer la normalisation
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
        -1: "Commande standard (bruit)",
        0: "Commande standard",
        1: "Commande rapide",
        2: "Commande prioritaire"
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
        'pca_loaded': pca_model is not None,
        'scaler_loaded': scaler is not None,
        'timestamp': datetime.now().isoformat()
    })

# -----------------------------
# Route pour vérifier les composantes PCA
# -----------------------------
@app.route('/pca_info')
def pca_info():
    if pca_model is not None:
        return jsonify({
            'n_components': pca_model.n_components_,
            'explained_variance_ratio': pca_model.explained_variance_ratio_.tolist(),
            'total_variance_explained': sum(pca_model.explained_variance_ratio_),
            'n_features_expected': pca_model.n_features_in_
        })
    else:
        return jsonify({'error': 'Modèle PCA non chargé'}), 404

# -----------------------------
# Route pour vérifier les caractéristiques des modèles
# -----------------------------
@app.route('/model_info')
def model_info():
    lightgbm_features = 10  # Basé sur l'erreur
    pca_features = 13      # Basé sur l'erreur
    
    return jsonify({
        'lightgbm': {
            'expected_features': lightgbm_features,
            'features_list': [
                'market_id', 'store_id', 'store_primary_category', 'total_items',
                'subtotal', 'num_distinct_items', 'total_onshift_partners',
                'hour_of_day', 'price_range', 'temperature'
            ]
        },
        'pca_dbscan': {
            'expected_features': pca_features,
            'features_list': [
                'market_id', 'store_id', 'store_primary_category', 'total_items',
                'subtotal', 'num_distinct_items', 'total_onshift_partners',
                'day_of_week_numeric', 'hour_of_day', 'price_range',
                'weather_condition', 'temperature', 'is_weekend'
            ]
        }
    })

# -----------------------------
# Lancer l'application
# -----------------------------
if __name__ == '__main__':
    print("🚀 Démarrage de l'application de prédiction de livraison avec PCA")
    print("="*60)
    print("✅ Tous les fichiers (modèles, PCA et scaler) sont présents." if lightgbm_model and dbscan_model and pca_model and scaler else "⚠️ Fichiers manquants")
    
    # Afficher les informations sur les caractéristiques attendues
    print("📊 Caractéristiques attendues:")
    print("   - LightGBM: 10 features")
    print("   - PCA/DBSCAN: 13 features")
    
    print("🌐 Serveur: http://0.0.0.0:5000")
    print("🔧 Mode debug: True")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000)