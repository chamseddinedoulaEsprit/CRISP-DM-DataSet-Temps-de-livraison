#!/usr/bin/env python3
"""
Script de démarrage pour l'application Flask de prédiction de livraison
"""

import os
import sys
from app import app

def check_model_files():
    """Vérifier que les fichiers de modèles existent"""
    required_files = ['lightgbm_model.pkl', 'dbscan_model.pkl']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("⚠️  ATTENTION: Les fichiers de modèles suivants sont manquants:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nVeuillez placer vos fichiers .pkl dans le répertoire de l'application.")
        print("L'application démarrera mais les prédictions pourraient ne pas fonctionner.")
        print("-" * 60)
    else:
        print("✅ Tous les fichiers de modèles sont présents.")

def main():
    print("🚀 Démarrage de l'application de prédiction de livraison")
    print("=" * 60)
    
    # Vérifier les modèles
    check_model_files()
    
    # Configuration
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    debug = os.environ.get('DEBUG', 'True').lower() == 'true'
    
    print(f"🌐 Serveur: http://{host}:{port}")
    print(f"🔧 Mode debug: {debug}")
    print("=" * 60)
    
    # Démarrer l'application
    try:
        app.run(host=host, port=port, debug=debug)
    except KeyboardInterrupt:
        print("\n👋 Arrêt de l'application...")
    except Exception as e:
        print(f"❌ Erreur lors du démarrage: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()