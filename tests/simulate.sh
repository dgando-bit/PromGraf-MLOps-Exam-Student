#!/bin/bash

# Configuration
URL="http://localhost:8080/predict"
COUNT=20

echo "Envoi de $COUNT requêtes à $URL..."

for ((i=1; i<=COUNT; i++))
do
   # Génération d'une heure aléatoire pour varier un peu
   RANDOM_HR=$(( ( RANDOM % 24 ) ))
   
   curl -s -X 'POST' \
     "$URL" \
     -H 'accept: application/json' \
     -H 'Content-Type: application/json' \
     -d "{
     \"temp\": 0.24,
     \"atemp\": 0.28,
     \"hum\": 0.8,
     \"windspeed\": 0.0,
     \"mnth\": 1,
     \"hr\": $RANDOM_HR,
     \"weekday\": 6,
     \"season\": 1,
     \"holiday\": 0,
     \"workingday\": 0,
     \"weathersit\": 1,
     \"dteday\": \"2024-01-01\"
   }" | jq '.predicted_count'

   echo " - Requête $i envoyée"
   sleep 0.2
done

echo "Terminé."