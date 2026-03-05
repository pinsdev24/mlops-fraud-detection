#!/bin/bash

# Configuration
DATA_DIR="data/raw"
ZIP_FILE="$DATA_DIR/creditcardfraud.zip"
KAGGLE_URL="https://www.kaggle.com/api/v1/datasets/download/mlg-ulb/creditcardfraud"

echo "Création du répertoire $DATA_DIR si nécessaire..."
mkdir -p "$DATA_DIR"

echo "Téléchargement des données de fraude de carte de crédit depuis Kaggle..."
# L'option -L permet de suivre les redirections, -o spécifie le fichier de sortie
curl -L -o "$ZIP_FILE" "$KAGGLE_URL"

echo "Extraction des données..."
unzip -o "$ZIP_FILE" -d "$DATA_DIR"

echo "Nettoyage..."
rm "$ZIP_FILE"

echo "Téléchargement et extraction terminés avec succès. Les données se trouvent dans $DATA_DIR/"
