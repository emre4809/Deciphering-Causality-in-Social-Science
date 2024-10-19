import os
import spacy
import re
import wordninja
import csv
import torch
# from concurrent.futures import ProcessPoolExecutor
from langdetect import detect, LangDetectException
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pymupdf
import unicodedata
from timer import start_timer, end_timer





# start the timer
start_time = start_timer() #took me 361 seconds to run the current code (only thirddataset used in data)

nlp = spacy.load("en_core_web_sm")

tokenizer = AutoTokenizer.from_pretrained("rasoultilburg/ssc_bert")
model = AutoModelForSequenceClassification.from_pretrained("rasoultilburg/ssc_bert")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #if there's cuda, it will be used to run the model
model.to(device)

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Function to extract text from PDF
def extract_pdf_text(pdf_path):
    """
    Extracts text from a PDF file using PyMuPDF.
    Args:
        pdf_path (str): The file path to the PDF.
    Returns:
        str: The extracted text from the PDF, joined as a single string.
    """
    doc = pymupdf.open(pdf_path)
    text_list = []  # list to collect text from each page
    for page in doc:
        text_list.append(page.get_text())  # appending text from each page
    doc.close()
    
    # Join all collected page texts into a single string
    return ''.join(text_list)

# Function to split text into sentences using SpaCy
def split_into_sentences(text):
    """
    Splits the provided text into individual sentences using SpaCy.
    Args:
        text (str): The text to be split into sentences.
    Returns:
        list: A list of sentences.
    """
    doc = nlp(text)
    return [sent.text for sent in doc.sents]





# Preprocessing function (updated to handle a list of sentences)
def preprocess_text(sentences):
    """
    Applies preprocessing steps to clean and normalize a list of sentences.
    Args:
        sentences (list of str): The input sentences to preprocess.
    Returns:
        list: A list of preprocessed sentences, with invalid sentences filtered out.
    """
    preprocessed_sentences = []

    for sentence in sentences:

        # Step 1: Remove control characters
        sentence = re.sub(r'[\n\t\003]', ' ', sentence)

        # Step 2: English language detection
        try:
            if detect(sentence) != 'en':
                continue
        except LangDetectException:
            continue

        # Step 3: Replace ligatures using Unicode normalization
        sentence = unicodedata.normalize('NFKD', sentence)

        # Step 4: Split concatenated words
        sentence = sentence.replace(' ', '') #adding this part to partially fix split lettered words
        sentence = ' '.join(wordninja.split(sentence))

        # Step 5: Remove extra spaces
        sentence = re.sub(r'\s+', ' ', sentence).strip()

        # Step 6: Discard sentences with fewer than 30 characters
        if len(sentence) < 30:
            continue

        # Add the valid preprocessed sentence to the list
        preprocessed_sentences.append(sentence)

    return preprocessed_sentences

def evaluate_causality(sentences):
    """Evaluate each sentence for causality in larger batches and return causal sentences."""
    causal_sentences = []
    batch_size = 128

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]

        tokens = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)

        with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
            outputs = model(**tokens)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1).cpu().numpy()

        for sentence, pred in zip(batch, predictions):
            if pred == 1:
                causal_sentences.append(sentence)

        del tokens, outputs, logits
        torch.cuda.empty_cache()

    return causal_sentences

def process_dataset_pipeline(dataset_id, pdf_path, output_csv):
    document_id = os.path.basename(pdf_path)

    text = extract_pdf_text(pdf_path)

    sentences = split_into_sentences(text)

    preprocessed_sentences = preprocess_text(sentences)

    causal_sentences = evaluate_causality(preprocessed_sentences)

    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["dataset_id", "document_id", "causal_sentence"])
        for sentence in causal_sentences:
            writer.writerow([dataset_id, document_id, sentence])


input_folders = [
    # "sample_data/Coda_PDF",
    # "sample_data/Xavier_PDF",
    "data/ThirdDataset_PDF" # use thirddataset to test if its working
]

# Define the output CSV path
output_folder = "processed_csv_causal"
os.makedirs(output_folder, exist_ok=True)

for folder in input_folders:
    dataset_id = os.path.basename(folder)
    for pdf_file in os.listdir(folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(folder, pdf_file)

            output_csv = os.path.join(output_folder, f"causal_{os.path.splitext(pdf_file)[0]}.csv")

            process_dataset_pipeline(dataset_id, pdf_path, output_csv)

print("Pipeline execution completed. Check the 'processed_csv_causal' directory for results.")

#end the timer
end_timer(start_time)