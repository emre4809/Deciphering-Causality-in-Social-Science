from pypdf import PdfReader

#example PDF paper to test pyPDF
reader = PdfReader('data/ssrn-4928865.pdf')

print(len(reader.pages))

page = reader.pages[0]

extracted_text = page.extract_text()
print(extracted_text)



import re

#manually cleaning the text
def clean_text(text):
    
    text = re.sub(r'(Figure|Table)\s+\d+', '', text)
    
    text = re.sub(r'(Figure|Table)\s+[A-Za-z0-9]+', '', text)
    
    text = re.sub(r'Page\s*\d+', '', text)
    text = re.sub(r'\d+/\d+', '', text)
    
    text = re.sub(r'\[.*?\]', '', text)
    
    text = re.sub(r'\s+', ' ', text)
    
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    return text

cleaned_text = clean_text(extracted_text)
print(cleaned_text)








