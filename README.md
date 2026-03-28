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