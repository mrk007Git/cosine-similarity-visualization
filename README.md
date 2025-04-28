# Text Embedding Cosine Similarity

This project demonstrates how to compute semantic similarity between two biomedical terms using text embeddings and cosine similarity.

Given two input phrases ("heart attack" and "myocardial infarction"), the code generates embeddings using a pre-trained language model (`all-MiniLM-L6-v2`) and calculates the cosine similarity between the resulting vectors.

## Requirements
- Python 3.8+
- sentence-transformers

Install required packages with:

```bash
pip install -r requirements.txt
