from fastapi import FastAPI, Depends
from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
from .config.database import init_db
from .routes.questions import router as question_router
from .routes.past_paper import route as papers_router
from fastapi.middleware.cors import CORSMiddleware
# from .schems.past_paper import CreatPastPaperSchema
#
# from sqlmodel import Session, select
# from .config.database import get_session
# from .processor.past_papers import create_past_paper_db
# from langchain_community.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from google.api_core.exceptions import GoogleAPIError
from langchain_google_genai import GoogleGenerativeAI
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import asyncio
from pydantic import BaseModel
# from openai import OpenAIError
import re


load_dotenv()

# create table in database


def on_startup():
    # init_db()
    try:
        init_db()
    except Exception as e:
        print(f"Database connection failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    on_startup()
    yield

# Initialize the LangChain LLM (Language Learning Model)
# gemni_api_key = os.getenv("OPENAI_API_KEY")
# llm = OpenAI(temperature=0.7, api_key=gemni_api_key)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

openai_api_key = os.getenv("GEMNI_API_KEY")

llm = GoogleGenerativeAI(model="gemini-1.5-flash", api_key=openai_api_key)

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Include the router in the main FastAPI app
app.include_router(question_router, prefix="/api/v1", tags=["Questions"])
app.include_router(papers_router, prefix="/api/v1", tags=["Papers"])


@app.get("/")
def read_root():
    return {"message": "Welcome to the Questions API!"}


@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    # Ensure the 'uploaded_pdfs' directory exists
    os.makedirs("uploaded_pdfs", exist_ok=True)

    file_location = f"uploaded_pdfs/{file.filename}"
    try:
        with open(file_location, "wb") as file_object:
            file_object.write(await file.read())
        return {"filename": file.filename}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to upload file: {str(e)}")
# Generate MCQs based on the topic extracted from the PDF

# @app.get("/generate-mcqs/")
# async def generate_mcqs(subject: str, topic: str):
#     # Assume the PDF is named after the subject (or provide specific filename)
#     file_location = f"uploaded_pdfs/{subject}.pdf"

#     # Extract PDF text and find relevant content
#     try:
#         pdf_text = extract_pdf_text(file_location)
#     except FileNotFoundError:
#         raise HTTPException(status_code=404, detail="PDF file not found")

#     # Optionally, filter the text for the relevant topic here
#     filtered_text =  filter_text_by_topic(pdf_text, topic)
#     print("filter",filtered_text )

#     # Generate MCQs using LangChain
#     prompt = f"Create 5 multiple choice questions (MCQs) on the topic of '{topic}' in '{subject}'. Provide 4 options for each question and clearly indicate the correct answer. Here is the text:\n{filtered_text}"

#     # Ensure we await the coroutine
#     response = await generate_mcqs_with_retry(prompt)

#     return {"mcqs": response}


class UserInput(BaseModel):
    query: str


@app.post("/generate-mcqs/")
async def generate_mcqs(user_input: UserInput):
    # Extracting subject and topic using simple regex or NLP (can use more advanced NLP tools)
    subject, topic = extract_subject_topic(user_input.query)
    if not subject or not topic:
        return {"error": "Please provide a valid subject and topic in your query."}

    # Extract PDF text
    file_location = f"uploaded_pdfs/{subject}.pdf"
    try:
        pdf_text = extract_pdf_text(file_location)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="PDF file not found")

    # Filter the text by topic
    filtered_text = filter_text_by_topic(pdf_text, topic)
    if filtered_text == "Topic not found in the provided text.":
        return {"error": "The specified topic was not found in the provided PDF."}

    # Generate MCQs using LangChain
    prompt = f"Create 5 multiple choice questions (MCQs) on the topic of '{topic}' in '{
        subject}'. Provide 4 options for each question and clearly indicate the correct answer. Here is the text:\n{filtered_text}"
    response = await generate_mcqs_with_retry(prompt)

    return {"mcqs": response}


def extract_subject_topic(query: str):
    # Basic regex-based extraction (can be replaced with more advanced NLP)
    print("subject", query)
    subject_match = re.search(r"subject is (\w+)", query, re.IGNORECASE)
    topic_match = re.search(r"topic is ([\w\s]+)", query, re.IGNORECASE)

    subject = subject_match.group(1) if subject_match else None
    topic = topic_match.group(1) if topic_match else None
    print("subject find", subject)
    print("topic find", topic)

    return subject, topic

# Function to extract text from a PDF


def extract_pdf_text(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Filter the text by the given topic


def filter_text_by_topic(text: str, topic: str) -> str:
    # You can implement more advanced topic filtering (e.g., keyword-based filtering)
    if topic.lower() in text.lower():
        return text
    else:
        return "Topic not found in the provided text."

# Retry mechanism for generating MCQs


async def generate_mcqs_with_retry(prompt, max_retries=5):
    retries = 0
    while retries < max_retries:
        try:
            # Assuming Google Gemini's API uses the 'invoke' method to handle prompts
            response = llm.invoke(prompt)  # Add await to the API call
            return response
        except GoogleAPIError as e:  # Replace 'GoogleAPIError' with the appropriate error class
            retries += 1
            wait_time = 2 ** retries  # Exponential backoff
            print(f"Gemini API error: {str(e)}. Retrying in {
                  wait_time} seconds...")
            await asyncio.sleep(wait_time)
    raise Exception("Max retries exceeded")


# @app.post("/generate-questions/")
# async def generate_questions(filename: str, topic: str):
#     try:
#         # Load the PDF
#         file_location = f"uploaded_pdfs/{filename}.pdf"
#         if not os.path.exists(file_location):
#             raise HTTPException(status_code=404, detail="File not found")

#         with open(file_location, "rb") as file:
#             pdf_reader = PdfReader(file)
#             text = ""
#             for page in pdf_reader.pages:
#                 text += page.extract_text()

#         # Split the text for processing
#         splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#         chunks = splitter.split_text(text)

#         # Initialize the LLM (e.g., OpenAI GPT model)
#         prompt_template = f"CGenerate short questions on the topic '{topic}' from the following text:\n{text}"

#         # Filter chunks related to the topic
#         topic_related_chunks = [chunk for chunk in chunks if topic.lower() in chunk.lower()]

#         # Generate questions from topic-related chunks
#         questions = []
#         for chunk in topic_related_chunks:
#             prompt = prompt_template.format(topic=topic, text=chunk)
#             response = llm(prompt)
#             questions.extend(response.split("\n"))

#         if not questions:
#             raise HTTPException(status_code=404, detail="No questions found for the given topic")

#         return {"topic": topic, "questions": questions}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error generating questions: {str(e)}")


async def ocr_page(image):
    return pytesseract.image_to_string(image)


@app.post("/extract-exercise-questions/")
async def extract_exercise_questions(filename: str, chapter_name: str):
    try:
        file_location = f"uploaded_pdfs/{filename}.pdf"
        if not os.path.exists(file_location):
            raise HTTPException(status_code=404, detail="File not found")

        # Convert PDF to images
        images = convert_from_path(file_location, poppler_path=r'C:\poppler-24.08.0\Library\bin',
                                   first_page=1, last_page=5)
        tasks = [ocr_page(image) for image in images]
        ocr_results = await asyncio.gather(*tasks)

        # Combine all OCR text
        text = " ".join(ocr_results)

        # Debugging: Display text start and end
        print("First 500 characters of OCR text:", text[:500])
        print("Last 500 characters of OCR text:", text[-500:])

        # Preprocess the text
        clean_text = re.sub(r'\s+', ' ', text)

        # Continue with regex and question extraction
        chapter_pattern = rf"(Unit|Chapter)\s*\d+.*?{
            chapter_name}.*?(?=(Unit|Chapter)\s*\d+|$)"
        chapter_text = re.search(
            chapter_pattern, clean_text, re.DOTALL | re.IGNORECASE)

        if not chapter_text:
            raise HTTPException(status_code=404, detail="Chapter not found")

        chapter_content = chapter_text.group(0)

        exercise_pattern = r"Exercise\s*Questions?.*?(?=(Unit|Chapter)\s*\d+|$)"
        exercise_questions = re.findall(
            exercise_pattern, chapter_content, re.DOTALL)

        if not exercise_questions:
            raise HTTPException(
                status_code=404, detail="No exercise questions found in the chapter")

        return {"chapter": chapter_name, "exercise_questions": exercise_questions}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error extracting exercise questions: {str(e)}")


# @app.post("/extract-exercise-questions/")
# async def extract_exercise_questions(filename: str, chapter_name: str):
#     try:
#         file_location = f"uploaded_pdfs/{filename}.pdf"
#         if not os.path.exists(file_location):
#             raise HTTPException(status_code=404, detail="File not found")

#         # Convert PDF to images for OCR
#         images = convert_from_path(file_location , poppler_path=r'C:\poppler-24.08.0\Library\bin')
#         text = ""

#         # Perform OCR on each page
#         for image in images:
#             page_text = pytesseract.image_to_string(image)
#             text += page_text

#         # Debug: Display the start and end of OCR text
#         print("First 500 characters of OCR text:", text[:500])
#         print("Last 500 characters of OCR text:", text[-500:])

#         # Preprocess the text
#         clean_text = re.sub(r'\s+', ' ', text)

#         # Adjust regex to handle 'Unit' or 'Chapter'
#         chapter_pattern = rf"(Unit|Chapter)\s*\d+.*?{chapter_name}.*?(?=(Unit|Chapter)\s*\d+|$)"
#         chapter_text = re.search(chapter_pattern, clean_text, re.DOTALL | re.IGNORECASE)

#         if not chapter_text:
#             raise HTTPException(status_code=404, detail="Chapter not found")

#         chapter_content = chapter_text.group(0)

#         # Find exercise questions in the chapter text
#         exercise_pattern = r"Exercise\s*Questions?.*?(?=(Unit|Chapter)\s*\d+|$)"
#         exercise_questions = re.findall(exercise_pattern, chapter_content, re.DOTALL)

#         if not exercise_questions:
#             raise HTTPException(status_code=404, detail="No exercise questions found in the chapter")

#         return {"chapter": chapter_name, "exercise_questions": exercise_questions}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error extracting exercise questions: {str(e)}")
