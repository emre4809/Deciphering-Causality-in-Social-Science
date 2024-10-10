import spacy

# loading spacy's english-based model (runs on CPU)
nlp = spacy.load("en_core_web_sm")

def split_into_sentences(text):
    """
    Splits the provided text into individual sentences using SpaCy.

    Args:
        text (str): The text to be split into sentences.

    Returns:
        list: A list of sentences.
    """
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences