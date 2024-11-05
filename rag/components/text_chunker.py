from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_text_chunks(text):
    """Splits text into chunks for processing."""
    print("Splits text into chunks for processing.")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100)
    text_split = text_splitter.split_text(text)
    print("text", len(text_split))
    return text_split
