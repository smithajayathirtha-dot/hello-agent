# Hello Agent – CSV FAQ Agent

A small Streamlit app that answers natural-language questions **using only the data in uploaded CSV files**
(Week 0 mini project, Applied Agentic AI for SWEs).

## How it works

1. Upload one or more CSV files → each becomes a Pandas data frame and a 5-row preview is shown.
2. Type a question (or click an example).
3. A LangChain pandas data-frame agent (`gpt-4o-mini`, temperature 0) runs pandas code over the
   frames, guided by a strict system prompt:
   - answer only from the data, never from general knowledge
   - compute totals/averages with code for numeric questions
   - reply `I could not find this information in the uploaded files.` when the data has no answer
   - write a copy-paste-ready answer and cite the source file + row ID
4. The answer is shown with a copy button; toggle **Show agent reasoning steps** in the sidebar to see the code the agent ran.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then put your OpenAI key in .env (or paste it in the sidebar)
streamlit run app.py
```

Sample files are in `sample_data/`: `ecommerce_faqs.csv`, `credit_card_terms.csv`, `hospital_policy.csv`, `saas_docs.csv`.

## Try these

| Question | Expected |
|---|---|
| What is the return policy for electronics? | 30 days, original packaging, 15% restocking fee if seal broken |
| What are the visiting hours in the hospital? | 10:00 AM – 8:00 PM, max 2 visitors |
| What is the API rate limit for the free plan? | 1,000 requests/day, 429 error beyond |
| What is the average fee amount across all credit card terms? | computed from `Fee Amount` |
| Who won the 2022 World Cup? | `I could not find this information in the uploaded files.` |

## Notes

- `allow_dangerous_code=True` is required by the LangChain pandas agent because it executes model-written
  Python locally. That's fine for a local warm-up tool on your own files; don't expose it publicly as-is.
- `langchain-experimental` is in maintenance mode; it still works and is what the brief asks for.
