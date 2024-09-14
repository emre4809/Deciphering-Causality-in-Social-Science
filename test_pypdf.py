# importing required modules
from pypdf import PdfReader

# creating a pdf reader object
reader = PdfReader('data/ssrn-4928865.pdf')

# printing number of pages in pdf file
print(len(reader.pages))

# getting a specific page from the pdf file
page = reader.pages[0]

# extracting text from page
extracted_text = page.extract_text()
print(extracted_text)



import re

def clean_text(text):
    # Remove figure/table references (e.g., "Figure 1", "Table 2")
    text = re.sub(r'(Figure|Table)\s+\d+', '', text)
    
    # Remove any remaining references to figures or tables
    text = re.sub(r'(Figure|Table)\s+[A-Za-z0-9]+', '', text)
    
    # Remove page numbers (common format: "Page 12", "12/34")
    text = re.sub(r'Page\s*\d+', '', text)
    text = re.sub(r'\d+/\d+', '', text)
    
    # Remove any text within square brackets (often references or notes)
    text = re.sub(r'\[.*?\]', '', text)
    
    # Remove multiple spaces, tabs, or newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Remove weird characters (non-ASCII characters can often be problematic)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    return text

# Example usage:
cleaned_text = clean_text(extracted_text)
print(cleaned_text)








