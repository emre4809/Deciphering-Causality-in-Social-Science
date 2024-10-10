import pymupdf

def extract_pdf_text(pdf_path):
    """
    Extracts text from a PDF file using PyMuPDF.

    Args:
        pdf_path (str): The file path to the PDF.

    Returns:
        str: The extracted text from the PDF, joined as a single string.
    """
    doc = pymupdf.open(pdf_path)
    text_list = []  #list to collect text from each page
    for page in doc:
        text_list.append(page.get_text())  #appending text from each page
    doc.close()
    
    #join all collected page texts into a single string
    return ''.join(text_list)