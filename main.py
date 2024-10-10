import os
from pdf_extractor import extract_pdf_text
from sentence_splitter import split_into_sentences
from timer import start_timer, end_timer

def main():
    # starts the timer
    start_time = start_timer()

    # directories
    folders = [
        "data/Coda_PDF",
        "data/Xavier_PDF",
        "data/ThirdDataset_PDF" #ran the code only on thirddataset_pdf for test, took around 70s.
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
                
                # step 3: will be the preprocessing step
                
                
                # # test to see if text-to-sentences is working:
                
                # print(f"\nSentences from {pdf_file}:\n")
                # print(sentences)
                # # optionally, add a break if you want to inspect only one pdf
                # break

    
    # end the timer and print the elapsed time
    end_timer(start_time)

if __name__ == "__main__":
    main()
            