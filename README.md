# Delivery Time Estimation Project

This project is focused on predicting delivery times for a leading intra-city logistics company in India. The project is implemented using the **CRISP-DM methodology** on a dataset of **172,000 rows** (`datadelevry.xlsx`).

---

## Project Overview

This delivery company (our client) is India's foremost marketplace for intra-city logistics, spearheading innovation in the nation's $40 billion intra-city logistics sector. With a mission to enhance the livelihoods of over **150,000 driver-partners**, our client ensures consistent earnings and fosters independence among its workforce. Currently, the company boasts a customer base exceeding **5 million**.

Collaborating with a diverse array of restaurants, the company facilitates the direct delivery of their goods to consumers. Leveraging a network of delivery partners sourced from various eateries, our client seeks to provide customers with **accurate estimated delivery times** based on factors such as their order, location, and available delivery partners.

---

## Associated Tasks

- **Regression**: Prediction of the delivery time estimation.

---

## Data Description

The dataset `datadelevry.xlsx` contains **172,000 rows**, each corresponding to a unique delivery. Each column represents a feature as described below:

| Column Name | Description |
|-------------|-------------|
| `market_id` | Integer ID for the market where the restaurant lies |
| `created_at` | Timestamp when the order was placed |
| `actual_delivery_time` | Timestamp when the order was delivered |
| `store_primary_category` | Category of the restaurant |
| `order_protocol` | Integer code for order protocol (e.g., via porter, call to restaurant, pre-booked, third party) |
| `total_items_subtotal` | Final price of the order |
| `num_distinct_items` | Number of distinct items in the order |
| `min_item_price` | Price of the cheapest item in the order |
| `max_item_price` | Price of the costliest item in the order |
| `total_onshift_partners` | Number of delivery partners on duty at the time the order was placed |
| `total_busy_partners` | Number of delivery partners attending other tasks |
| `total_outstanding_orders` | Total number of orders to be fulfilled at that moment |

---

## Methodology

The project follows the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology, which includes the following phases:

1. **Business Understanding** – Define project objectives and deliverables.
2. **Data Understanding** – Explore and describe the dataset.
3. **Data Preparation** – Clean, format, and transform data for analysis.
4. **Modeling** – Apply regression models to predict delivery times.
5. **Evaluation** – Assess model performance and validate results.
6. **Deployment** – Provide recommendations for integrating the predictive model.

---

## Usage

1. Load the dataset `datadelevry.xlsx` into your preferred Python environment (e.g., Pandas).
2. Explore and preprocess the data.
3. Train regression models to predict delivery times.
4. Evaluate model performance using appropriate metrics (e.g., RMSE, MAE).
5. Deploy the model to estimate delivery times for future orders.

---

## Requirements

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn (for visualization)
- Jupyter Notebook (optional)

---
# Application Web Flask - Prédiction de Livraison 🚚

Cette application web utilise deux modèles de machine learning pour analyser et prédire les livraisons :
- **LightGBM** : Prédiction du temps de livraison en minutes
- **DBSCAN** : Classification par clustering des commandes

## 📁 Structure du Projet

```
├── app.py                 # Application Flask principale
├── run.py                 # Script de démarrage
├── requirements.txt       # Dépendances Python
├── README.md             # Documentation
├── .env.example          # Variables d'environnement
├── templates/
│   └── index.html        # Interface utilisateur
├── lightgbm_model.pkl    # Modèle LightGBM (à ajouter)
└── dbscan_model.pkl      # Modèle DBSCAN (à ajouter)
```

## 🔧 Installation

### 1. Cloner ou télécharger le projet
```bash
# Si vous avez les fichiers dans un dossier
cd votre-dossier-projet
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Ajouter vos modèles
Placez vos fichiers de modèles dans le répertoire principal :
- `lightgbm_model.pkl`
- `dbscan_model.pkl`

### 4. Configuration (optionnel)
```bash
cp .env.example .env
# Éditez .env selon vos besoins
```

## 🚀 Démarrage

### Méthode 1 : Script de démarrage
```bash
python run.py
```

### Méthode 2 : Flask direct
```bash
python app.py
```

### Méthode 3 : Variables d'environnement
```bash
export FLASK_APP=app.py
export FLASK_ENV=development
flask run --host=0.0.0.0 --port=5000
```

## 🌐 Accès à l'Application

Une fois démarrée, l'application est accessible à :
- **Local** : http://localhost:5000
- **Réseau** : http://votre-ip:5000

## 📊 Fonctionnalités

### Interface Utilisateur
- **Formulaire intuitif** avec tous les champs nécessaires
- **Design responsive** compatible mobile/desktop
- **Validation des données** côté client et serveur
- **Affichage des résultats** en temps réel

### Prédictions Disponibles
1. **Temps de livraison** (LightGBM)
   - Prédiction en minutes
   - Format lisible (heures/minutes)

2. **Classification des commandes** (DBSCAN)
   - Identification du cluster
   - Interprétation du type de commande

### Champs de Données
L'application accepte ces 13 variables :
- `market_id` : ID du marché
- `store_id` : ID du magasin
- `store_primary_category` : Catégorie du magasin
- `total_items` : Nombre total d'articles
- `subtotal` : Sous-total de la commande
- `num_distinct_items` : Nombre d'articles distincts
- `total_onshift_partners` : Nombre de livreurs disponibles
- `delivery_duration_min` : Durée estimée de livraison
- `day_of_week_numeric` : Jour de la semaine (1-7)
- `hour_of_day` : Heure de la journée (0-23)
- `price_range` : Gamme de prix
- `weather_condition` : Condition météorologique
- `temperature` : Température en °C

## 🔧 Personnalisation

### Modifier les Catégories
Dans `app.py`, vous pouvez ajuster :
```python
STORE_CATEGORIES = ['Restaurant', 'Grocery', ...]
WEATHER_CONDITIONS = ['Clear', 'Cloudy', ...]
PRICE_RANGES = ['Budget', 'Mid-range', ...]
```

### Préprocessing des Données
La fonction `preprocess_features()` dans `app.py` doit être adaptée selon votre preprocessing original :
```python
def preprocess_features(df):
    # Adaptez cette fonction selon votre preprocessing
    # Encodage, normalisation, feature engineering, etc.
    return df_processed
```

### Interprétation des Clusters
Modifiez `interpret_cluster()` selon votre analyse DBSCAN :
```python
def interpret_cluster(cluster_id):
    interpretations = {
        -1: "Commande atypique",
        0: "Commande standard",
        # Ajoutez vos interprétations
    }
    return interpretations.get(cluster_id, f"Cluster {cluster_id}")
```

## 🐛 Résolution de Problèmes

### Modèles non trouvés
```
Erreur: Les fichiers de modèles n'ont pas été trouvés
```
**Solution** : Vérifiez que `lightgbm_model.pkl` et `dbscan_model.pkl` sont dans le répertoire principal.

### Erreur de prédiction
```
Erreur LightGBM/DBSCAN: ...
```
**Solutions** :
1. Vérifiez le preprocessing dans `preprocess_features()`
2. Assurez-vous que les colonnes correspondent à celles d'entraînement
3. Vérifiez les types de données

### Port déjà utilisé
```
Address already in use
```
**Solution** : Changez le port dans `.env` ou utilisez :
```bash
python run.py
# Ou définissez PORT=5001 dans les variables d'environnement
```

## 📡 API Endpoints

### GET /
Page principale avec le formulaire

### POST /predict
Endpoint de prédiction
```json
{
  "market_id": 1,
  "store_id": 1,
  "store_primary_category": "Restaurant",
  // ... autres champs
}
```

**Réponse** :
```json
{
  "success": true,
  "predictions": {
    "delivery_time_minutes": 25.5,
    "delivery_time_formatted": "25min",
    "cluster": 0,
    "cluster_interpretation": "Commande standard"
  }
}
```

### GET /health
Vérification de l'état de l'application

## 🛡️ Sécurité

- Validation des données d'entrée
- Gestion des erreurs robuste
- Pas d'exposition des détails internes
- Protection contre les injections

## 📈 Monitoring

L'endpoint `/health` permet de vérifier :
- État de l'application
- Chargement des modèles
- Horodatage

## 🚀 Déploiement en Production

### Avec Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Avec uWSGI
```bash
pip install uwsgi
uwsgi --http :5000 --wsgi-file app.py --callable app --processes 4
```

### Variables d'Environnement de Production
```bash
export FLASK_ENV=production
export DEBUG=False
```

## 📝 Notes Importantes

1. **Preprocessing** : Adaptez la fonction `preprocess_features()` selon votre pipeline original
2. **Modèles** : Assurez-vous que vos modèles sont compatibles avec les versions de libraries
3. **Sécurité** : En production, désactivez le mode debug
4. **Performance** : Pour de gros volumes, considérez une mise en cache des prédictions

## 📧 Support

Pour des questions techniques ou des améliorations, consultez les commentaires dans le code ou adaptez selon vos besoins spécifiques.

# 📁 Structure du Projet - Application Flask de Prédiction de Livraison

```
delivery-prediction-app/
│
├── 🐍 FICHIERS PYTHON
│   ├── app.py                    # Application Flask principale
│   ├── run.py                    # Script de démarrage avec vérifications
│   └── test_app.py              # Tests automatisés de l'application
│
├── 🌐 INTERFACE UTILISATEUR
│   └── templates/
│       └── index.html           # Interface web responsive avec Bootstrap
│
├── ⚙️ CONFIGURATION
│   ├── requirements.txt         # Dépendances Python
│   ├── .env.example            # Variables d'environnement (template)
│   └── nginx.conf              # Configuration Nginx (optionnel)
│
├── 🐳 DOCKER
│   ├── Dockerfile              # Image Docker de l'application
│   └── docker-compose.yml      # Orchestration multi-conteneurs
│
├── 📜 SCRIPTS
│   └── deploy.sh               # Script de déploiement automatisé
│
├── 📚 DOCUMENTATION
│   ├── README.md               # Documentation complète
│   ├── QUICK_START.md          # Guide de démarrage rapide
│   └── PROJECT_STRUCTURE.md    # Ce fichier
│
├── 📊 DONNÉES EXEMPLE
│   └── sample_data.json        # Exemples de données pour tests
│
└── 🤖 MODÈLES ML (À AJOUTER)
    ├── lightgbm_model.pkl      # Modèle LightGBM pour prédiction temps
    └── dbscan_model.pkl        # Modèle DBSCAN pour clustering
```

## 🎯 Composants Principaux

### 1. **Application Flask** (`app.py`)
- ✅ Endpoints pour prédiction et health check
- ✅ Chargement automatique des modèles
- ✅ Preprocessing des données
- ✅ Gestion d'erreurs robuste
- ✅ API REST JSON

### 2. **Interface Web** (`templates/index.html`)
- ✅ Design moderne avec Bootstrap 5
- ✅ Formulaire interactif avec validation
- ✅ Affichage des résultats en temps réel
- ✅ Interface responsive mobile/desktop
- ✅ Icons Font Awesome

### 3. **Scripts Utilitaires**
- ✅ `run.py` : Démarrage avec vérifications
- ✅ `test_app.py` : Tests automatisés
- ✅ `deploy.sh` : Déploiement automatisé

### 4. **Containerisation Docker**
- ✅ Multi-stage build optimisé
- ✅ Health checks intégrés
- ✅ Nginx reverse proxy
- ✅ Configuration de production

## 🔧 Flux de Données

```
Utilisateur → Interface Web → Flask App → Preprocessing → Modèles ML → Résultats
    ↑                                                                        ↓
    └─────────────────── Affichage des Prédictions ←─────────────────────────┘
```

## 📈 Fonctionnalités

### ✅ Prédictions Disponibles
- **LightGBM** : Temps de livraison en minutes
- **DBSCAN** : Classification par clusters

### ✅ Sécurité & Robustesse
- Validation des données d'entrée
- Gestion d'erreurs complète
- Logs structurés
- Health checks

### ✅ Déploiement
- Environnement virtuel Python
- Containerisation Docker
- Configuration Nginx
- Scripts automatisés

### ✅ Monitoring
- Health check endpoint
- Tests automatisés
- Logs détaillés

## 🚀 Options de Démarrage

### **1. Développement Local**
```bash
python run.py
```

### **2. Déploiement Automatisé**
```bash
./deploy.sh
```

### **3. Container Docker**
```bash
docker-compose up -d
```

### **4. Tests**
```bash
python test_app.py
```

## 📊 Variables d'Entrée

L'application accepte **13 variables** correspondant aux colonnes de votre dataset :

| Variable | Type | Description |
|----------|------|-------------|
| `market_id` | int | Identifiant du marché |
| `store_id` | int | Identifiant du magasin |
| `store_primary_category` | str | Catégorie du magasin |
| `total_items` | int | Nombre total d'articles |
| `subtotal` | float | Sous-total de la commande |
| `num_distinct_items` | int | Articles distincts |
| `total_onshift_partners` | int | Livreurs disponibles |
| `delivery_duration_min` | float | Durée estimée |
| `day_of_week_numeric` | int | Jour de la semaine (1-7) |
| `hour_of_day` | int | Heure (0-23) |
| `price_range` | str | Gamme de prix |
| `weather_condition` | str | Condition météo |
| `temperature` | float | Température (°C) |

## 🔄 Workflow de Développement

1. **Développement** → Tests locaux avec `test_app.py`
2. **Validation** → Déploiement avec `deploy.sh`
3. **Production** → Container Docker avec `docker-compose`
4. **Monitoring** → Health checks et logs

## 🛠️ Personnalisation

### Points de Personnalisation Principaux :
- **`preprocess_features()`** : Adapter selon votre preprocessing
- **Catégories** : Modifier les listes dans `app.py`
- **Interprétations** : Adapter `interpret_cluster()`
- **Style** : Modifier le CSS dans `index.html`

## 📝 Notes Importantes

- ⚠️ **Modèles requis** : Placez vos `.pkl` dans le répertoire racine
- ⚠️ **Preprocessing** : Adaptez selon votre pipeline original
- ⚠️ **Production** : Désactivez le mode debug
- ⚠️ **Sécurité** : Utilisez HTTPS en production

- # 🚀 Guide de Démarrage Rapide

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



