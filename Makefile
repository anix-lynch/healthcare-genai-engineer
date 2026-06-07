.PHONY: install serve demo test eval gate clean agent-eval adversarial-eval

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

# Run the 20-query golden set, write outputs/eval_summary.json
eval:
	python -m evaluation.ragas_runner

# Compare current eval vs baseline; exit 1 on regression past tolerance
gate:
	python -m evaluation.regression_gate

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache outputs/

# Agent execution eval — Bed Ops computes a disposition from live ER state;
# measures task-completion / decision-correctness / tool-call-success across
# 50 labelled scenarios (evaluation/agent_scenarios.json). See docs/agent_eval_design.md.
agent-eval:
	python -m evaluation.agent_eval

# Adversarial / safety slice — injection (known + paraphrased), benign precision,
# empty-input, oversize, control-char, empty-retrieval refusal, output-action guard.
# Reports honest injection recall (regex baseline is not exhaustive); gates the
# mechanical + precision checks. Writes outputs/adversarial_summary.json.
adversarial-eval:
	python -m evaluation.adversarial_eval
