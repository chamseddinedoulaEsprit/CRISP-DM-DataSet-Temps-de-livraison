#!/bin/bash

# Script de déploiement pour l'application Flask de prédiction de livraison

echo "🚀 Déploiement de l'Application de Prédiction de Livraison"
echo "=========================================================="

# Couleurs pour les messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonction pour afficher des messages colorés
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Vérifier Python
log_info "Vérification de Python..."
if ! command -v python3 &> /dev/null; then
    log_error "Python 3 n'est pas installé"
    exit 1
fi
log_success "Python 3 trouvé: $(python3 --version)"

# Vérifier pip
log_info "Vérification de pip..."
if ! command -v pip3 &> /dev/null; then
    log_error "pip3 n'est pas installé"
    exit 1
fi
log_success "pip3 trouvé: $(pip3 --version)"

# Créer un environnement virtuel si nécessaire
if [ ! -d "venv" ]; then
    log_info "Création de l'environnement virtuel..."
    python3 -m venv venv
    log_success "Environnement virtuel créé"
else
    log_info "Environnement virtuel existant trouvé"
fi

# Activer l'environnement virtuel
log_info "Activation de l'environnement virtuel..."
source venv/bin/activate
log_success "Environnement virtuel activé"

# Mettre à jour pip
log_info "Mise à jour de pip..."
pip install --upgrade pip > /dev/null 2>&1
log_success "pip mis à jour"

# Installer les dépendances
log_info "Installation des dépendances..."
if pip install -r requirements.txt > /dev/null 2>&1; then
    log_success "Dépendances installées avec succès"
else
    log_error "Erreur lors de l'installation des dépendances"
    exit 1
fi

# Vérifier la présence des modèles
log_info "Vérification des fichiers de modèles..."
missing_models=()

if [ ! -f "lightgbm_model.pkl" ]; then
    missing_models+=("lightgbm_model.pkl")
fi

if [ ! -f "dbscan_model.pkl" ]; then
    missing_models+=("dbscan_model.pkl")
fi

if [ ${#missing_models[@]} -eq 0 ]; then
    log_success "Tous les fichiers de modèles sont présents"
else
    log_warning "Fichiers de modèles manquants:"
    for model in "${missing_models[@]}"; do
        echo "   - $model"
    done
    log_warning "L'application démarrera mais les prédictions pourraient ne pas fonctionner"
fi

# Créer le fichier .env si nécessaire
if [ ! -f ".env" ]; then
    log_info "Création du fichier .env..."
    cp .env.example .env
    log_success "Fichier .env créé à partir de .env.example"
fi

# Test de l'application
log_info "Test rapide de l'application..."
if python3 -c "from app import app; print('Import réussi')" > /dev/null 2>&1; then
    log_success "L'application peut être importée correctement"
else
    log_error "Erreur lors de l'import de l'application"
    exit 1
fi

echo ""
echo "=========================================================="
log_success "Déploiement terminé avec succès!"
echo ""
echo "📋 Prochaines étapes:"
echo "1. Ajoutez vos fichiers de modèles (.pkl) dans ce répertoire"
echo "2. Démarrez l'application avec: python3 run.py"
echo "3. Ou testez avec: python3 test_app.py"
echo ""
echo "🌐 L'application sera accessible à: http://localhost:5000"
echo "=========================================================="

# Proposer de démarrer l'application
read -p "Voulez-vous démarrer l'application maintenant? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "Démarrage de l'application..."
    python3 run.py
fi