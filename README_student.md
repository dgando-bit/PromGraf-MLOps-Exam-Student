### Structure du repo :

```
.
├── Makefile
├── README.md
├── README_student.md
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

1. `make train`       — Entraîner le modèle initial
2. `make all`         — Lancer tous les services (API, Prometheus, Grafana, Node Exporter)
3. `make evaluation`  — Évaluer le modèle et mettre à jour les métriques Prometheus
4. `make simulate`    — Simuler un scénario de trafic réel sur /predict
5. `make error`       — Générer des erreurs HTTP (422/404) pour tester le panel Error Rate
6. `make fire-alert`  — Déclencher intentionnellement les alertes Grafana (drift + RMSE)
7. `make stop`        — Arrêter tous les services

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