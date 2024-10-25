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


# Preprocessing function (updated to handle a list of sentences)
def preprocess_text(sentences):
    """
    Applies preprocessing steps to clean and normalize a list of sentences.
    """
    preprocessed_sentences = []

    for sentence in sentences:

        #Step 1: Remove control characters
        sentence = re.sub(r'[\n\t\003]', ' ', sentence)

        #Step 2: English language detection
        try:
            if detect(sentence) != 'en':
                continue
        except LangDetectException:
            continue

        #Step 3: Replace ligatures using Unicode normalization
        sentence = unicodedata.normalize('NFKD', sentence)

        #Step 4: Using wordninja to split words with more than 15 characters
        words = sentence.split() #tokenize sentence into words

        processed_words = []
        for word in words:
            if len(word) > 15:  #if a word is longer than 15 characters, split it using wordninja
                processed_words.extend(wordninja.split(word))
            else:
                processed_words.append(word)

        sentence = ' '.join(processed_words) #join the sentence with fixed list of words

        #Step 5: Fix words that are split across lines due to hyphens
        sentence = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', sentence)

        #Step 6: Ignore sentences with 6 or more consecutive split letters
        if re.search(r'(\b\w\s){6,}', sentence):
          continue

        #Step 7: Remove extra spaces
        sentence = re.sub(r'\s+', ' ', sentence).strip()

        #Step 8: Discard sentences with fewer than 30 characters
        if len(sentence) < 30:
            continue

        # Add the valid preprocessed sentence to the list
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