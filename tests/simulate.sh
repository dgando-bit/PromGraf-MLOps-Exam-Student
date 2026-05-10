#!/bin/bash
# =============================================================================
# tests/simulate.sh — Génération de trafic sur /predict
# Simule une utilisation réelle en variant les features à chaque requête.
# Usage : bash tests/simulate.sh [COUNT] [URL]
# =============================================================================

URL="${2:-http://localhost:8080/predict}"
COUNT="${1:-20}"

echo "🚴 Envoi de $COUNT requêtes à $URL..."
echo "-------------------------------------------"

SUCCESS=0
FAIL=0

for ((i=1; i<=COUNT; i++)); do

  # --- Features aléatoires pour simuler des conditions variées ---
  HR=$(( RANDOM % 24 ))
  WEEKDAY=$(( RANDOM % 7 ))
  SEASON=$(( (RANDOM % 4) + 1 ))          # 1-4
  WEATHERSIT=$(( (RANDOM % 3) + 1 ))      # 1-3
  HOLIDAY=$(( RANDOM % 2 ))               # 0 ou 1
  WORKINGDAY=$(( 1 - HOLIDAY ))           # cohérent avec holiday

  # Températures normalisées (0.0 - 1.0)
  TEMP=$(awk "BEGIN {printf \"%.2f\", $RANDOM/32767}")
  ATEMP=$(awk "BEGIN {printf \"%.2f\", $RANDOM/32767}")
  HUM=$(awk "BEGIN {printf \"%.2f\", $RANDOM/32767}")
  WIND=$(awk "BEGIN {printf \"%.2f\", ($RANDOM % 5000)/32767}")

  RESPONSE=$(curl -s -o /tmp/predict_response.json -w "%{http_code}" \
    -X POST "$URL" \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    -d "{
      \"temp\": $TEMP,
      \"atemp\": $ATEMP,
      \"hum\": $HUM,
      \"windspeed\": $WIND,
      \"mnth\": 1,
      \"hr\": $HR,
      \"weekday\": $WEEKDAY,
      \"season\": $SEASON,
      \"holiday\": $HOLIDAY,
      \"workingday\": $WORKINGDAY,
      \"weathersit\": $WEATHERSIT,
      \"dteday\": \"2024-01-01\"
    }")

  if [ "$RESPONSE" == "200" ]; then
    PREDICTED=$(cat /tmp/predict_response.json | python3 -m json.tool 2>/dev/null | grep predicted_count | awk '{print $2}' | tr -d ',')
    echo "  ✅ Requête $i/$COUNT — hr=$HR weekday=$WEEKDAY → predicted_count=$PREDICTED"
    SUCCESS=$(( SUCCESS + 1 ))
  else
    echo "  ❌ Requête $i/$COUNT — HTTP $RESPONSE"
    FAIL=$(( FAIL + 1 ))
  fi

  sleep 0.2
done

echo "-------------------------------------------"
echo "✅ Succès : $SUCCESS / $COUNT"
if [ "$FAIL" -gt 0 ]; then
  echo "❌ Échecs  : $FAIL / $COUNT"
fi
echo "📊 Vérifiez les métriques sur http://localhost:9090 (Prometheus)"
echo "   ou dans Grafana → dashboard 'API Performance'"