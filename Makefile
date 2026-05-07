all: 
	docker compose up --build -d

stop: 
	docker compose down

evaluation:
	docker compose up -d --build evaluation

train:
	docker compose run --rm bike-api python train.py

evaluate:
	docker compose run --rm evaluation python run_evaluation.py