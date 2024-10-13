import os
from pdf_extractor import extract_pdf_text
from sentence_splitter import split_into_sentences
from timer import start_timer, end_timer

def main():
    # start the timer
    start_time = start_timer()

    # directories
    folders = [
        "data/Coda_PDF",
        "data/Xavier_PDF",
        "data/ThirdDataset_PDF"  # for testing, thirddataset is good since it has low amount of pdfs. comment out others and leave thirddataset only 
    ]

    # list to store the extracted text from each PDF
    all_texts = []  # all_texts will be: [“entire text of the first pdf”, “entire text of the second pdf”, …]

    # list to store the sentences for each PDF (list of lists)
    all_sentences = []  # all_sentences will be: [[“sentence1frompdf1”, “sentence2frompdf1, …], [“sentence1frompdf2”, “sentence2frompdf2”, “sentence3frompdf2”, …] …]

    # loop through all folders
    for folder in folders:
        for pdf_file in os.listdir(folder):
            if pdf_file.endswith(".pdf"):
                pdf_path = os.path.join(folder, pdf_file)
                
                # step 1: extract text from PDF
                pdf_text = extract_pdf_text(pdf_path)
                all_texts.append(pdf_text)  # store the full text for each PDF
                
                # step 2: split text into sentences using SpaCy
                sentences = split_into_sentences(pdf_text)
                all_sentences.append(sentences)  # store the list of sentences for each PDF
                
                # step 3: preprocessing will come here


    # at this point, "all_texts" is a list of full PDF texts,
    # and "all_sentences" is a list of lists, where each inner list contains sentences from one PDF.

    # we can now iterate over either "all_texts" or "all_sentences" for preprocessing. uncomment if you want to see the outputs
    # print("\nAll texts (PDF-to-text output):\n", all_texts)
    # print("\nAll sentences (Text-to-sentence output):\n", all_sentences)

    #end of the timer
    end_timer(start_time)

if __name__ == "__main__":
    main()
