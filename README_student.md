# Examen du cours Prometheus & Grafana. (English version below)

### Structure du repo :

```
.
├── Makefile
├── README.md
├── data
├── deployment
│   ├── grafana
│   │   ├── dashboards
│   │   │   ├── API-dashboard.json
│   │   │   ├── Infra-dashboard.json
│   │   │   └── Model-dashboard.json
│   │   └── provisioning
│   │       ├── dashboards
│   │       │   └── dashboards.yaml
│   │       └── datasources
│   │           └── datasources.yaml
│   └── prometheus
│       ├── prometheus.yml
│       └── rules
│           └── alert_rules.yml
├── docker-compose.yml
├── models
│   ├── bike_model.pkl
│   └── reference_data.parquet
├── src
│   ├── api
│   │   ├── Dockerfile
│   │   ├── main.py
│   │   ├── requirements.txt
│   │   └── train.py
│   ├── api-metrics-dashboard.json
│   └── evaluation
│       ├── Dockerfile
│       ├── requirements.txt
│       └── run_evaluation.py
└── tests
    └── simulate.sh
```


### Guide de démarrage rapide

1. `make all` (Lancer l'API et le monitoring)
2. `make train` (Entraîner le modèle initial)
3. `make evaluation` (Lancer l'évaluation du modèle)
4. `make fire-alert` (Déclencher une alerte de dérive)
5. `make simulate` (Simuler une scénario de test)
6. `make stop` (Nettoyer l'environnement)

### --- Métrique bonus : détection de dérive Evidently ---

Pourquoi evidently_drift_detected ?
C'est une Gauge binaire (0 = pas de dérive, 1 = dérive détectée) issue du
rapport Evidently DataDriftPreset. Elle permet de déclencher une alerte
Grafana/Alertmanager dès qu'une dérive de distribution est détectée sur les
données courantes par rapport à la référence de janvier 2011. Combinée aux
Gauges RMSE/MAE/R2, elle fournit le signal le plus direct pour savoir si le
modèle doit être ré-entraîné (dérive data) ou si ses performances ont
simplement dégradé (dérive de performance). C'est typiquement la métrique
que l'on branche sur un alert rule "drift_detected == 1".