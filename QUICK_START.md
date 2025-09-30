# 🚀 Guide de Démarrage Rapide

## Installation Express

### 1. Préparation
```bash
# Placez vos fichiers .pkl dans ce dossier
cp /chemin/vers/lightgbm_model.pkl .
cp /chemin/vers/dbscan_model.pkl .
```

### 2. Déploiement automatique
```bash
./deploy.sh
```

### 3. Démarrage manuel
```bash
pip install -r requirements.txt
python run.py
```

## 🐳 Déploiement Docker

### Option 1 : Docker Compose (Recommandé)
```bash
# Avec Nginx reverse proxy
docker-compose up -d

# Application seule
docker-compose up -d delivery-prediction-app
```

### Option 2 : Docker Direct
```bash
# Build
docker build -t delivery-prediction .

# Run
docker run -d -p 5000:5000 \
    -v $(pwd)/lightgbm_model.pkl:/app/lightgbm_model.pkl:ro \
    -v $(pwd)/dbscan_model.pkl:/app/dbscan_model.pkl:ro \
    --name delivery-app delivery-prediction
```

## 🧪 Tests

```bash
# Test complet
python test_app.py

# Test sur port personnalisé
python test_app.py 8000

# Health check rapide
curl http://localhost:5000/health
```

## 🌐 Accès

- **Application** : http://localhost:5000
- **API Health** : http://localhost:5000/health
- **Avec Nginx** : http://localhost:80

## 📱 Utilisation

1. Ouvrez l'interface web
2. Remplissez le formulaire avec vos données
3. Cliquez sur "Prédire"
4. Obtenez :
   - Temps de livraison prédit (minutes)
   - Classification par cluster

## 🔧 Configuration Rapide

### Variables d'environnement
```bash
export HOST=0.0.0.0
export PORT=5000
export DEBUG=False
```

### Fichier .env
```
HOST=0.0.0.0
PORT=5000
DEBUG=True
```

## ⚡ Commandes Utiles

```bash
# Arrêter Docker
docker-compose down

# Voir les logs
docker-compose logs -f

# Restart
docker-compose restart

# Status
docker-compose ps
```

## 🆘 Dépannage Express

| Problème | Solution |
|----------|----------|
| Port 5000 occupé | Changez `PORT=5001` dans .env |
| Modèles non trouvés | Vérifiez les fichiers .pkl dans le dossier |
| Erreur de prédiction | Adaptez `preprocess_features()` dans app.py |
| Erreur Docker | Vérifiez `docker-compose logs` |

## 📞 Test API Direct

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "market_id": 1,
    "store_id": 100,
    "store_primary_category": "Restaurant",
    "total_items": 3,
    "subtotal": 25.50,
    "num_distinct_items": 2,
    "total_onshift_partners": 5,
    "delivery_duration_min": 30,
    "day_of_week_numeric": 3,
    "hour_of_day": 19,
    "price_range": "Mid-range",
    "weather_condition": "Clear",
    "temperature": 22.5
  }'
```