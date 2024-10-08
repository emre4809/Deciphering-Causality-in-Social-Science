import os
import pymupdf
import time
import spacy

# loading spacy's english-based model (runs on CPU)
nlp = spacy.load("en_core_web_sm")


#function to extract text from PDFs
def extract_pdf_text(pdf_path):
    doc = pymupdf.open(pdf_path)
    text_list = []  #list to collect text from each page
    for page in doc:
        text_list.append(page.get_text())  #appending text from each page
    doc.close()
    
    #join all collected page texts into a single string
    return ''.join(text_list)

# split text into sentences using SpaCy
def split_into_sentences(text):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences


# timer 
start_time = time.time()

# directories
folders = [
    "data/Coda_PDF",
    "data/Xavier_PDF",
    "data/ThirdDataset_PDF"
]

# loop through all folders and extract text from each PDF
for folder in folders:
    for pdf_file in os.listdir(folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(folder, pdf_file)
            
            # step 1: extract text and time the process
            pdf_text = extract_pdf_text(pdf_path)
            
            # step 2: split text into sentences using SpaCy
            sentences = split_into_sentences(pdf_text)
            
            
            # # test to see if pdf-to-text is working
           
            # if pdf_file == "s00426-012-0476-2.pdf":
            #     print(f"\nProcessing {pdf_file}...\n")
                
            #     pdf_text = extract_pdf_text(pdf_path)
                
            #     print(f"\nExtracted text from {pdf_file}:\n")
            #     print(pdf_text)
            
            
            # # test to see if text-to-sentences is working
            
            # print(f"\nSentences from {pdf_file}:\n")
            # print(sentences)
            # # optionally, add a break if you want to inspect only ONE pdf for now
            # break
            
            
            

end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTotal time taken: {elapsed_time:.2f} seconds")
