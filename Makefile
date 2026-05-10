# =============================================================================
# Makefile — Bike Sharing MLOps
# =============================================================================

.PHONY: all stop train evaluation fire-alert simulate error

# -----------------------------------------------------------------------------
# all : Démarre tous les services (API, Prometheus, Grafana, Node Exporter)
# -----------------------------------------------------------------------------
all:
	docker compose up --build -d

# -----------------------------------------------------------------------------
# stop : Arrête tous les services
# -----------------------------------------------------------------------------
stop:
	docker compose down

# -----------------------------------------------------------------------------
# train : Entraîne le modèle et sauvegarde bike_model.pkl + reference_data.parquet
# -----------------------------------------------------------------------------
train:
	docker compose run --rm bike-api python train.py

# -----------------------------------------------------------------------------
# evaluation : Exécute run_evaluation.py → met à jour les métriques Prometheus
# -----------------------------------------------------------------------------
evaluation:
	docker compose run --rm evaluation python run_evaluation.py

# -----------------------------------------------------------------------------
# fire-alert : Déclenche intentionnellement les alertes Grafana
# -----------------------------------------------------------------------------
fire-alert:
	@echo "🔥 Déclenchement intentionnel des alertes Grafana..."
	@curl -s -X POST http://localhost:8080/trigger-drift | python3 -m json.tool
	@echo ""
	@echo "✅ Métriques forcées :"
	@echo "   - evidently_drift_detected = 1  → alerte DriftDetected"
	@echo "   - model_rmse_score         = 999 → alerte RMSETooHigh"
	@echo ""
	@echo "📊 Vérifiez dans Grafana (http://localhost:3000) :"
	@echo "   - Dashboard 'Model Performance & Drift' → panel 'Data Drift Detected'"
	@echo "   - Alerting → Alert rules → DriftDetected et RMSETooHigh"
	@echo ""
	@echo "ℹ️  Pour réinitialiser les métriques : make evaluation"

# -----------------------------------------------------------------------------
# simulate : Génère du trafic de prédictions valides sur /predict
# -----------------------------------------------------------------------------
simulate:
	@chmod +x tests/simulate.sh
	@./tests/simulate.sh

# -----------------------------------------------------------------------------
# error : Génère intentionnellement des erreurs 422 et 404
#         pour tester le panel "Error Rate (%)" dans Grafana.
#
# Erreurs générées :
#   - 422 Unprocessable Entity → validation Pydantic (type invalide, champs manquants)
#   - 404 Not Found            → endpoint inexistant
#
# Vérification : Grafana → dashboard "API Performance" → panel "Error Rate (%)"
# La requête PromQL filtre status_code=~"4..|5.."
# -----------------------------------------------------------------------------
error:
	@echo "💥 Génération d'erreurs HTTP sur l'API..."
	@echo ""
	@echo "--- 5 requêtes 422 — champ temp invalide ---"
	@for i in 1 2 3 4 5; do \
		echo -n "  Requête $$i → "; \
		curl -s -o /dev/null -w "HTTP %{http_code}\n" \
			-X POST http://localhost:8080/predict \
			-H "Content-Type: application/json" \
			-d '{"temp": "invalide", "hr": 0}'; \
	done
	@echo ""
	@echo "--- 5 requêtes 422 — body vide (champs manquants) ---"
	@for i in 1 2 3 4 5; do \
		echo -n "  Requête $$i → "; \
		curl -s -o /dev/null -w "HTTP %{http_code}\n" \
			-X POST http://localhost:8080/predict \
			-H "Content-Type: application/json" \
			-d '{}'; \
	done
	@echo ""
	@echo "--- 5 requêtes 404 — endpoint inexistant ---"
	@for i in 1 2 3 4 5; do \
		echo -n "  Requête $$i → "; \
		curl -s -o /dev/null -w "HTTP %{http_code}\n" \
			http://localhost:8080/endpoint-inexistant; \
	done
	@echo ""
	@echo "✅ 15 erreurs envoyées : 10x 422 + 5x 404"
	@echo "📊 Grafana → API Performance → Error Rate (%) doit afficher un pic"
	@echo "ℹ️  Le panel capture status_code=~'4..|5..'"