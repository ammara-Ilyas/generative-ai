from PyPDF2 import PdfReader


def get_pdf_text(pdf_docs):
    """Extracts text from each page of uploaded PDF files."""
    print("Extracts text from each page of uploaded PDF files.")
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text
