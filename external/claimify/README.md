# Claimify - High‑quality claim extraction from LLM outputs
Shhhh 🤫 !!! (It's Unofficial implementation from Microsoft but works amazingly!)


# Results?
You can verify the claim below against a Web Search or your Context!

**Inputs**:

```python
question = "What is the population of Delhi?" # The answer bellow is super random to show what happens
answer = "The largest city in India is the captial itself accomodating more than a Billion people over an area of 5000Sq Kms. It has more than 5 Lakhs people migrating each month from all over the world"

```

**Results**:

```🤫markdown
Extracted Claims:
1. Delhi is the largest city in India
2. Delhi has a population of more than one billion people [interpreting 'accommodating' as population count]
3. Delhi covers an area of approximately 5,000 square kilometers [interpreting '5000Sq Kms' as the city's total area]
4. Delhi is the capital of [the country of] India
5. Delhi receives more than 500,000 migrants each month [i.e., an incoming flow per month, not a cumulative total]
6. The migrants that Delhi receives [the incoming migrants referenced above] come from all over the world [i.e., originating from multiple countries worldwide]
```

# References
- [Official Paper](https://arxiv.org/abs/2502.10855)
- [Microsoft Blog](https://www.microsoft.com/en-us/research/blog/claimify-extracting-high-quality-claims-from-language-model-outputs/)

# Working
What it does:
- Given a question and an answer, extracts specific, verifiable, decontextualized factual claims.
- Single entrypoint: [`def extract_claims()`](src/claimify.py:59) on the pipeline class [`class Claimify()`](src/claimify.py:34).

How it works (process)
- 1) Selection — scan each sentence and keep only those with a specific, verifiable proposition. See [`def _split_sentences()`](src/claimify.py:151) and [`def _selection_stage()`](src/claimify.py:234).
- 2) Disambiguation — resolve names/acronyms and linguistic ambiguity using only local context. See [`def _disambiguation_stage()`](src/claimify.py:374).
- 3) Decomposition — break the disambiguated sentence into atomic, decontextualized propositions. See [`def _decomposition_stage()`](src/claimify.py:516).
- Prompts for each stage live in [`src/prompts.py`](src/prompts.py).

Quick usage
- Provide an LLM function with signature: `llm(prompt: str, temperature: float) -> str.`
- Example:

```python
from src.claimify import Claimify

def my_llm(prompt, temperature):
    # call your model here; must return string
    return "..."

c = Claimify(llm_function=my_llm)
claims = c.extract_claims(question="...", answer="...")
print(claims)
```

- Full runnable sample: [`def example_usage()`](src/claimify.py:645).

Tuning and design notes
- Hyperparameters (retries, context windows, completions) are defined in [`def __init__()`](src/claimify.py:35).
- Practical enhancements:
  - Robust sentence tokenization with NLTK and regex fallback: [`def _split_sentences()`](src/claimify.py:151).
  - Lenient parsing for disambiguation/decomposition outputs to handle minor LLM format drift.
  - Conservative logging for traceability.
- Temperature: the paper specifies 0/0.2, but some providers reject exact zeros. Use a small value; see comments at the top of [`src/claimify.py`](src/claimify.py).

License/Attribution
- This is an unofficial re‑implementation for research/engineering reference. Credit to the authors of Claimify.
