import os
import spacy
import re
import wordninja
import csv
import torch
import pymupdf
import unicodedata
from langdetect import detect, LangDetectException
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the SpaCy model
nlp = spacy.load("en_core_web_sm")

# Load BERT-based tokenizer and model for causality detection
tokenizer = AutoTokenizer.from_pretrained("rasoultilburg/ssc_bert")
model = AutoModelForSequenceClassification.from_pretrained("rasoultilburg/ssc_bert")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Function to extract text from PDF
def extract_pdf_text(pdf_path):
    """
    Extracts text from a PDF file using PyMuPDF.
    """
    doc = pymupdf.open(pdf_path)
    text_list = []
    for page in doc:
        text_list.append(page.get_text())
    doc.close()
    return ''.join(text_list)


# Function to split text into sentences using SpaCy
def split_into_sentences(text):
    """
    Splits the provided text into individual sentences using SpaCy.
    """
    doc = nlp(text)
    return [sent.text for sent in doc.sents]


# Preprocessing function to clean and normalize sentences
def preprocess_text(sentences):
    """
    Applies preprocessing steps to clean and normalize a list of sentences.
    """
    preprocessed_sentences = []

    for sentence in sentences:
        sentence = re.sub(r'[\n\t\003]', ' ', sentence)
        
        try:
            if detect(sentence) != 'en':
                continue
        except LangDetectException:
            continue

        sentence = unicodedata.normalize('NFKD', sentence)
        sentence = sentence.replace(' ', '')  # partially fix split lettered words
        sentence = ' '.join(wordninja.split(sentence))
        sentence = re.sub(r'\s+', ' ', sentence).strip()

        if len(sentence) >= 30:
            preprocessed_sentences.append(sentence)

    return preprocessed_sentences


# Function to evaluate causality in sentences using BERT
def evaluate_causality(sentences):
    """
    Evaluates each sentence for causality in larger batches and returns causal sentences.
    """
    causal_sentences = []
    batch_size = 128

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)

        with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
            outputs = model(**tokens)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1).cpu().numpy()

        causal_sentences.extend([sentence for sentence, pred in zip(batch, predictions) if pred == 1])

    torch.cuda.empty_cache()
    return causal_sentences


# Full pipeline function to process a dataset
def process_dataset_pipeline(dataset_id, pdf_path, output_csv):
    """
    Full pipeline to process a dataset, extract sentences, apply preprocessing, and detect causal sentences.
    """
    document_id = os.path.basename(pdf_path)

    # Step 1: Extract text from the PDF
    text = extract_pdf_text(pdf_path)

    # Step 2: Split text into sentences
    sentences = split_into_sentences(text)

    # Step 3: Preprocess sentences
    preprocessed_sentences = preprocess_text(sentences)

    # Step 4: Detect causal sentences
    causal_sentences = evaluate_causality(preprocessed_sentences)

    # Step 5: Write causal sentences to a CSV file
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["dataset_id", "document_id", "causal_sentence"])
        for sentence in causal_sentences:
            writer.writerow([dataset_id, document_id, sentence])
