#Imports
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import os
import pandas as pd
import glob

#Cause effect extraction model setup
model_name = "tanfiona/unicausal-tok-baseline"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

nlp = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",
    # device=0  #Uncomment this line if you have a GPU
)

#Taken directly from model's huggingface page. Label 0 and 1 are not used, I-C: inner cause, I-E: inner effect
label_mapping = {
    # "LABEL_0": "B-C",
    # "LABEL_1": "B-E",
    "LABEL_2": "I-C",
    "LABEL_3": "I-E",
}

#Create output folder, name can be changed
output_folder = "extracted_causes_effects"
os.makedirs(output_folder, exist_ok=True)

#Taking causal sentences as input, returning the cause and effect in the sentence
input_folders = [
    "processed_csv_causal/Coda_PDF",
    "processed_csv_causal/Xavier_PDF",
    "processed_csv_causal/ThirdDataset_PDF"
]

#Process each dataset, replicating main.py 
for folder in input_folders:
    dataset_id = os.path.basename(folder)
    #Change the output subfolder name to datasetname_cause-effect
    output_subfolder = os.path.join(output_folder, f"{dataset_id}_cause-effect")
    os.makedirs(output_subfolder, exist_ok=True)
    print(f"Processing dataset: {dataset_id}")

    #Columns of new csv file, instead of causal sentence, there is causes and effects
    csv_files = glob.glob(os.path.join(folder, "*.csv"))    #Iterate over each CSV file
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)  #Read the CSV file into a DataFrame
        #Initialize lists to store the extracted data
        causes = [] 
        effects = []
        dataset_ids = []
        document_ids = []

        #Iterate over each row in the DataFrame
        for index, row in df.iterrows():
            sentence = row['causal_sentence']   #Extract the sentence and IDs from the row
            document_id = row['document_id']
            dataset_id_row = row['dataset_id']

            results = nlp(sentence)     #Use the NLP model to get token classifications for the sentence

            cause_found = False     #Initialize variables to store the first found cause and effect
            effect_found = False
            cause = ''
            effect = ''

            #Iterate over the model's results
            for res in results:
                if not cause_found and res['entity_group'] == 'LABEL_2':    #Check if the cause has not been found and the current token is labeled as 'LABEL_2' (I-C)
                    #Store the word as the cause
                    cause = res['word']
                    cause_found = True
                elif not effect_found and res['entity_group'] == 'LABEL_3':     #Check if the effect has not been found and the current token is labeled as 'LABEL_3' (I-E)
                    effect = res['word']    #Store the word as the effect
                    effect_found = True

                #If both cause and effect have been found, exit the loop early
                if cause_found and effect_found:
                    break

            #Append the extracted cause and effect to the respective lists
            causes.append(cause)
            effects.append(effect)
            dataset_ids.append(dataset_id_row)
            document_ids.append(document_id)

        #Create a new DataFrame with the extracted data
        new_df = pd.DataFrame({
            'dataset_id': dataset_ids,
            'document_id': document_ids,
            'cause': causes,
            'effect': effects
        })

        #Extract the document ID from the input CSV filename
        input_filename = os.path.basename(csv_file)  #eg., "causal_ENG00001.csv"
        #Remove the "causal_" prefix and ".csv" suffix to get the document ID
        doc_id_from_filename = input_filename[len("causal_"):-len(".csv")]

        #Create the new output with the CSV filename
        output_csv_filename = f"cause-effect_{doc_id_from_filename}.csv"    #eg., "cause-effect_ENG00001.csv"
        output_csv = os.path.join(output_subfolder, output_csv_filename)

        #Save to CSV
        new_df.to_csv(output_csv, index=False)
        print(f"Processed and saved: {output_csv}")

#Message indicating all datasets are done
print("All datasets have been processed.")
