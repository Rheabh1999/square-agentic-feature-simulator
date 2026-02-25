# Agentic AI Feature Launch Simulator (Square Sellers)

A Streamlit app where an LLM acts as a constrained agent:
- calls tools to compute baseline metrics
- detects fragility signals
- simulates a Square feature impact
- compares scenarios
- produces a PM-grade recommendation memo

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
