# Document Contradiction Detection

Automatic detection of document-level contradictions using automated claim-extraction and NLI models

### Requirements

Tested with Python `3.12`. Run the following to create the environment:

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Run the following command to download the required NLTK resources:
```
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### Environment variables (.env)

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Then edit `.env` and set at least the keys you need:

- `HF_TOKEN`: Hugging Face token for faster model downloads and higher rate limits (recommended for local models).
- `OPENAI_API_KEY`: required when using the `remote` backend.

### Claim Extractor configuration

You can change model names directly in `.env`:

- `CLAIM_MODEL_1`: local model run 1
- `CLAIM_MODEL_2`: local model run 2
- `CLAIM_MODEL_REMOTE`: remote model

Examples are in `.env.example`.

Optionally run the smoke test:

```bash
python scripts/smoke_test_claim_extractor.py
```