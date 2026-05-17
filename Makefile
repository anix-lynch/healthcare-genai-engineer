.PHONY: install serve demo test clean

install:
	pip install -r requirements.txt

serve:
	uvicorn app.main:app --reload --port 8000

# Fire one /ask through the TestClient and pretty-print the JSON.
# No uvicorn needed for `make demo` — keeps the smoke test self-contained.
demo:
	@python -c "from fastapi.testclient import TestClient; from app.main import app; \
import json; c = TestClient(app); \
r = c.post('/v1/ask', json={'query': '62yo male chest pain hypertension', 'k': 5, 'method': 'bm25'}); \
print(json.dumps(r.json(), indent=2))"

test:
	pytest tests/ -v

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache
