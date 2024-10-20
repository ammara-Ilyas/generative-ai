from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
# loader = PyPDFLoader(
#     "./data/chem.pdf",
# )

print("hello1")
try:
    # loader = TextLoader("data.txt")
    loader = PyPDFLoader(
        "./data/chem.pdf",
    )
    data = loader.load()
    print(data)
    print("hello2")
except FileNotFoundError as e:
    print("File not found : ", e)
except Exception as e:
    print("Error while loading text : ", e)

text_splitter = CharacterTextSplitter(
    # separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
    # length_function=len,
    # is_separator_regex=False,
)
texts = text_splitter.split_documents(data)
print(texts[0])
