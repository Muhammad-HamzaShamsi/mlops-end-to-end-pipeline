.PHONY: install train test lint serve docker up down

install:
	pip install -r requirements-dev.txt

train:
	python -m src.train

test:
	pytest -q

lint:
	ruff check src tests

serve:
	uvicorn src.serve:app --reload --port 8000

docker:
	docker build -t mlops-api:latest .

up:
	docker compose up --build -d

down:
	docker compose down
