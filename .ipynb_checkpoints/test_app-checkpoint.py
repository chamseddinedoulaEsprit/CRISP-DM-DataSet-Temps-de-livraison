#!/usr/bin/env python3
"""
Script de test pour l'application Flask de prédiction de livraison
"""

import requests
import json
import time
import sys

def test_health_endpoint(base_url):
    """Tester l'endpoint de santé"""
    print("🔍 Test de l'endpoint /health...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Endpoint /health OK")
            print(f"   Status: {data.get('status')}")
            print(f"   LightGBM chargé: {data.get('lightgbm_loaded')}")
            print(f"   DBSCAN chargé: {data.get('dbscan_loaded')}")
            return True
        else:
            print(f"❌ Erreur /health: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur de connexion /health: {e}")
        return False

def test_main_page(base_url):
    """Tester la page principale"""
    print("\n🔍 Test de la page principale...")
    try:
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200:
            print("✅ Page principale accessible")
            return True
        else:
            print(f"❌ Erreur page principale: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur de connexion page principale: {e}")
        return False

def test_prediction_endpoint(base_url):
    """Tester l'endpoint de prédiction avec des données d'exemple"""
    print("\n🔍 Test de l'endpoint /predict...")
    
    # Données d'exemple
    test_data = {
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
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict",
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Endpoint /predict accessible")
            
            if result.get('success'):
                predictions = result.get('predictions', {})
                print("📊 Prédictions reçues:")
                
                # Vérifier les prédictions LightGBM
                if 'delivery_time_minutes' in predictions:
                    print(f"   🕒 Temps de livraison: {predictions['delivery_time_minutes']} min")
                    print(f"   📝 Format: {predictions.get('delivery_time_formatted', 'N/A')}")
                elif 'delivery_time_error' in predictions:
                    print(f"   ⚠️ Erreur LightGBM: {predictions['delivery_time_error']}")
                
                # Vérifier les prédictions DBSCAN
                if 'cluster' in predictions:
                    print(f"   🎯 Cluster: {predictions['cluster']}")
                    print(f"   📖 Interprétation: {predictions.get('cluster_interpretation', 'N/A')}")
                elif 'cluster_error' in predictions:
                    print(f"   ⚠️ Erreur DBSCAN: {predictions['cluster_error']}")
                
                return True
            else:
                print(f"❌ Erreur dans la prédiction: {result.get('error')}")
                return False
        else:
            print(f"❌ Erreur /predict: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Détails: {error_data}")
            except:
                print(f"   Réponse: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur de connexion /predict: {e}")
        return False

def main():
    print("🧪 Test de l'Application Flask - Prédiction de Livraison")
    print("=" * 60)
    
    # URL de base (modifiable)
    base_url = "http://localhost:5000"
    
    # Vérifier si un autre port est spécifié
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
            base_url = f"http://localhost:{port}"
        except ValueError:
            print(f"Port invalide: {sys.argv[1]}. Utilisation du port par défaut 5000.")
    
    print(f"🌐 URL de test: {base_url}")
    print("-" * 60)
    
    # Attendre un peu que le serveur démarre
    print("⏳ Attente du démarrage du serveur (3 secondes)...")
    time.sleep(3)
    
    # Exécuter les tests
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Health endpoint
    if test_health_endpoint(base_url):
        tests_passed += 1
    
    # Test 2: Main page
    if test_main_page(base_url):
        tests_passed += 1
    
    # Test 3: Prediction endpoint
    if test_prediction_endpoint(base_url):
        tests_passed += 1
    
    # Résumé
    print("\n" + "=" * 60)
    print(f"📊 Résultats des Tests: {tests_passed}/{total_tests} réussis")
    
    if tests_passed == total_tests:
        print("🎉 Tous les tests sont passés! L'application fonctionne correctement.")
        sys.exit(0)
    else:
        print("⚠️ Certains tests ont échoué. Vérifiez la configuration.")
        sys.exit(1)

if __name__ == '__main__':
    main()