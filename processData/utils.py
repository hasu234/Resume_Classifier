import fitz  # PyMuPDF
import re
import joblib

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file given its file path.

    Args:
        pdf_path (str): The file path of the PDF file.

    Returns:
        str: The extracted text from the PDF file.
    """
    text = ""
    pdf_document = fitz.open(pdf_path)
    
    for page_number in range(pdf_document.page_count):
        page = pdf_document[page_number]
        text += page.get_text()
    
    pdf_document.close()
    return text



def cleanText(text):
    """
    Cleans the given text by removing URLs, RT and cc, hashtags, mentions, punctuations, non-ASCII characters, and extra whitespace.

    Args:
    text (str): The text to be cleaned.

    Returns:
    str: The cleaned text.
    """
    text = re.sub('http\S+\s*', ' ', text)  # remove URLs
    text = re.sub('RT|cc', ' ', text)  # remove RT and cc
    text = re.sub('#\S+', '', text)  # remove hashtags
    text = re.sub('@\S+', '  ', text)  # remove mentions
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)  # remove punctuations
    text = re.sub(r'[^\x00-\x7f]',r' ', text) 
    text = re.sub('\s+', ' ', text)  # remove extra whitespace
    return text




def transformText(requiredText):
    """
    Transforms the given text into a matrix of TF-IDF features.

    Args:
    requiredText (list): A list of strings containing the text to be transformed.

    Returns:
    WordFeatures (sparse matrix): A sparse matrix of TF-IDF features.
    """
    word_vectorizer =  joblib.load('./model_weight/tfidf_vectorizer.joblib')
    WordFeatures = word_vectorizer.transform([requiredText])

    return WordFeatures
