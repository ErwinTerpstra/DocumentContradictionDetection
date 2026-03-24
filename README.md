# Document Contradiction Detection

Automatic detection of document-level contradictions using automated claim-extraction and NLI models

### Requirements

Installation with Conda is recommended. Tested with Python `3.12`. Run the following to create the environment:

```
conda create -n contradetect python=3.12
conda activate contradetect
conda install ipykernel ipywidgets pytorch pytorch-cuda scikit-learn nltk transformers transformers[torch] tokenizers sentence_transformers huggingface_hub pandas matplotlib -c nvidia -c pytorch
```