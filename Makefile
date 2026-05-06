all: 
	docker compose up --build -d

stop: 
	docker compose down

evaluation:
	docker compose up -d --build evaluation
