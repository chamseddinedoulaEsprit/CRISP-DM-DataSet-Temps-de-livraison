# Dockerfile pour l'application Flask de prédiction de livraison

FROM python:3.10-slim

# Métadonnées
LABEL maintainer="votre-email@example.com"
LABEL description="Application Flask pour la prédiction de temps de livraison"

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Répertoire de travail
WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copier les requirements et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application
COPY . .

# Créer un utilisateur non-root
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Exposer le port
EXPOSE 5000

# Vérification de santé
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Commande par défaut
CMD ["python", "run.py"]