import os
from definitions_preprocessing import process_dataset_pipeline
from timer import start_timer, end_timer

# start timer
start_time = start_timer()

# input folders
input_folders = [
    "data/Coda_PDF",
    "data/Xavier_PDF",
    "data/ThirdDataset_PDF"
]

# main output folder
output_folder = "processed_csv_causal"
os.makedirs(output_folder, exist_ok=True)

# process each PDF in the input folders
for folder in input_folders:
    dataset_id = os.path.basename(folder)
    output_subfolder = os.path.join(output_folder, dataset_id)  #put each dataset into a separate folder
    os.makedirs(output_subfolder, exist_ok=True)
    print(f"Processing dataset: {dataset_id}")
    for pdf_file in os.listdir(folder):
        if pdf_file.lower().endswith(".pdf"):   #added .lower() because of pdfs ending with ".PDF" instead of ".pdf"
            sanitized_name = pdf_file.replace("~", "")  #added this because pdfs with "~" were not getting processed
            pdf_path = os.path.join(folder, pdf_file)
            output_csv = os.path.join(output_subfolder, f"causal_{os.path.splitext(sanitized_name)[0]}.csv")
            
            # run the full pipeline on each PDF
            process_dataset_pipeline(dataset_id, pdf_path, output_csv)
            print(f"Processed: {pdf_file}") #uncomment to see progress of pdfs getting processed

print(f"Pipeline execution completed. Check the '{output_folder}' directory for results.")

# end timer
end_timer(start_time)