import nltk
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document
from PyPDF2 import PdfReader
import os

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

def get_synonyms(word):
    """Get a set of synonyms for a given word."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower())
    return synonyms

def preprocess_document(document):
    """Tokenize the document, find synonyms, and return a list of words."""
    words = nltk.word_tokenize(document.lower())
    processed_words = []
    for word in words:
        # Add word itself and its synonyms
        synonyms = get_synonyms(word)
        synonyms.add(word)
        processed_words.append(" ".join(synonyms))
    return " ".join(processed_words)

def calculate_similarity(doc1, doc2):
    """Calculate cosine similarity between two documents."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([doc1, doc2])
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity_matrix[0, 0]

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_docx(docx_path):
    """Extract text from a DOCX file."""
    doc = Document(docx_path)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return text

def extract_text_from_directory(directory_path):
    """Extract text from all PDF and DOCX files in a directory."""
    corpus = []
    for filename in os.listdir(directory_path): # Lists all files in the specified directory and constructs the full file path for each file.
        file_path = os.path.join(directory_path, filename)
        if filename.lower().endswith(".pdf"): #checks for pdf extension
            corpus.append(extract_text_from_pdf(file_path))
        elif filename.lower().endswith(".docx"): #checks for docx extension
            corpus.append(extract_text_from_docx(file_path))
    return corpus

def main():
    # Paths for the dfocuments in runtime
    input_document_path = input("Enter the path of the document: ")
    corpus_directory_path = input("\n Enter the path to the directory containing corpus documents: ")

    # Extract text from the input document
    if input_document_path.lower().endswith(".pdf"):
        input_text = extract_text_from_pdf(input_document_path)
    elif input_document_path.lower().endswith(".docx"):
        input_text = extract_text_from_docx(input_document_path)
    else:
        print("Unsupported input document format. Please provide a document PDF or DOCX extension.")
        return

    # Extract text from all documents in the corpus directory
    if not os.path.isdir(corpus_directory_path):
        print("Corpus directory not found.")
        return

    corpus = extract_text_from_directory(corpus_directory_path) #retrieve text data from documents located

    if not corpus: #if no document is present/processed
        print("No supported documents found in the corpus directory.") 
        return

    # Preprocess the input document
    processed_input_text = preprocess_document(input_text)


    # Comparing with each document in the corpus
    threshold = 0.5  # Setting the plagiarism threshold
    results = []
    for i, doc_text in enumerate(corpus): #with i being the index of the document and doc_text being the text of the document.
        processed_doc_text = preprocess_document(doc_text)
        similarity_score = calculate_similarity(processed_input_text, processed_doc_text)
        is_plagiarized = similarity_score >= threshold
        results.append((i, similarity_score, is_plagiarized))

    # Display results
    for i, similarity_score, is_plagiarized in results:
        print(f"\nSimilarity with document {i+1}: {similarity_score:.2f}")
        if is_plagiarized:
            print(f"  Plagiarized: The similarity score exceeds the threshold of {threshold}. \n")
        else:
            print(f"  Not Plagiarized: The similarity score is below the threshold of {threshold}.\n")

if __name__ == "__main__":
    main()
