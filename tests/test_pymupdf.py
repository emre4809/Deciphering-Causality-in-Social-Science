#code for a single pdf-to-text done on a sample PDF

import pymupdf

doc = pymupdf.open("data/ssrn-4928865.pdf") # open a document


for page in doc: # iterate the document pages
    text = page.get_text() # get plain text (is in UTF-8)
print(text)

# close the document after extraction
doc.close()
