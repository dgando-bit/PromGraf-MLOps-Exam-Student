all: 
	docker compose up --build -d

stop: 
	docker compose down

evaluation:
	docker compose up -d --build evaluation

train:
	docker compose run --rm bike-api python train.py

evaluation:
	docker compose run --rm evaluation python run_evaluation.py

fire-alert:
	@echo "🔥 Déclenchement intentionnel d'une alerte de dérive (Data Drift)..."
	@curl -s -X POST http://localhost:8000/trigger-drift | jq .
	@echo "\n✅ Requête envoyée. Vérifiez la métrique 'evidently_drift_detected' sur /metrics."

simulate:
	@chmod +x votre_script.sh
	@./tests/simulate.sh
