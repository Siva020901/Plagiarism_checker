# Document Similarity Checker

This project provides a tool for checking the similarity between a given document and a set of documents in a corpus. The tool uses Natural Language Processing (NLP) techniques to preprocess text, extract synonyms, and calculate cosine similarity to detect potential plagiarism.

## Features

- Extract text from PDF and DOCX files.
- Tokenize and preprocess documents by including synonyms.
- Calculate cosine similarity between a given document and a corpus of documents.
- Detect potential plagiarism based on a similarity threshold.

## Requirements

- Python 3.6 or higher
- Required Python packages: `nltk`, `scikit-learn`, `python-docx`, `PyPDF2`

You can install the required packages using pip:

```bash
pip install nltk scikit-learn python-docx PyPDF2
