import os
import re
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import chromadb
from typing import List
import google.generativeai as genai
from chromadb import Documents, EmbeddingFunction, Embeddings
import fitz  # PyMuPDF (for PDF handling)
import datetime  # For generating dynamic collection name
import uuid      # Alternative for unique identifier
from pymongo import MongoClient
import io


os.environ["GEMINI_API_KEY"] = "AIzaSyCjQihAK86WBUqnDzycuTWpE7gMZvOqJik"

# Initialize Flask app
app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt', 'docx'}

# Configure MongoDB connection
mongo_client = MongoClient("mongodb://localhost:27017/")  # Update with your MongoDB URI
db = mongo_client['your_database_name']  # Replace with your MongoDB database name
file_collection = db['file_storage']
chroma_collection_metadata = db['chroma_collection_metadata']

# Global variable to store the name of the latest collection
latest_collection_name = None

# Split text into chunks
def split_text(text: str):
    split_text = re.split(r'\n \n', text)
    return [chunk for chunk in split_text if chunk]

# Custom embedding function with Gemini API
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("Gemini API Key not provided. Set GEMINI_API_KEY as an environment variable.")
        
        genai.configure(api_key=gemini_api_key)
        model = "models/embedding-001"
        title = "Custom query"
        return genai.embed_content(
            model=model,
            content=input,
            task_type="retrieval_document",
            title=title
        )["embedding"]

# Function to create and populate ChromaDB with chunks
from bson import ObjectId

# Function to create and populate ChromaDB with chunks
def create_chroma_db(documents: List[str], name: str):
    # Store document chunks in MongoDB and get document IDs
    doc_ids = []
    for doc in documents:
        doc_id = file_collection.insert_one({"collection_name": name, "content": doc}).inserted_id
        doc_ids.append(doc_id)  # Store ObjectId directly

    # Initialize ChromaDB client (no `database_uri` argument used)
    chroma_client = chromadb.PersistentClient()
    db = chroma_client.create_collection(name=name, embedding_function=GeminiEmbeddingFunction())
    
    # Populate ChromaDB with document chunks from MongoDB
    for doc_id in doc_ids:
        # Convert ObjectId to match MongoDB format
        content = file_collection.find_one({"_id": ObjectId(doc_id)})
        if content:  # Check if document exists
            db.add(documents=[content["content"]], ids=[str(doc_id)])
        else:
            print(f"Warning: Document with ID {doc_id} not found in MongoDB.")

    # Save metadata for easy lookup
    chroma_collection_metadata.insert_one({"name": name, "doc_ids": [str(doc_id) for doc_id in doc_ids]})
    
    return db, name


# Function to load an existing ChromaDB collection
def load_chroma_collection(name: str):
    # Retrieve ChromaDB collection
    chroma_client = chromadb.PersistentClient()
    return chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())

# Retrieve relevant passages from ChromaDB
def get_relevant_passage(query: str, db, n_results: int):
    passages = db.query(query_texts=[query], n_results=n_results)['documents']
    return passages[0] if passages else []

# Create RAG prompt
def make_rag_prompt(query: str, relevant_passage: str) -> str:
    escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    
    prompt = ("""You are a helpful and informative bot that answers questions using text from the reference passage included below. \
Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
strike a friendly and conversational tone. \
If the passage is irrelevant to the answer, you may ignore it.
QUESTION: '{query}'
PASSAGE: '{relevant_passage}'

ANSWER:
""").format(query=query, relevant_passage=escaped)
    
    return prompt

# Generate response using Gemini API
def generate_response(prompt):
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-pro')
    answer = model.generate_content(prompt)
    return answer.text

# Generate answer using RAG pipeline
def generate_answer(db, query):
    relevant_text_chunks = get_relevant_passage(query, db, n_results=3)
    combined_text = "".join(relevant_text_chunks) if relevant_text_chunks else "No relevant information found."
    
    prompt = make_rag_prompt(query, relevant_passage=combined_text)
    response = generate_response(prompt)

    return response

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function to extract text from PDF (or file-like object)
def extract_text_from_pdf(file_content):
    with fitz.open("pdf", file_content) as doc:
        pdf_text = ""
        for page in doc:
            pdf_text += page.get_text()
    return pdf_text

@app.route("/")
def index():
    return render_template('chat.html')

# Route to handle document upload, splitting, and ChromaDB storage
@app.route("/upload", methods=["POST"])
def upload_document():
    global latest_collection_name  # Use global to update the latest collection name

    if 'document' not in request.files:
        return "No file part", 400

    file = request.files['document']
    if file.filename == '':
        return "No selected file", 400

    # Check if the file is allowed
    if not allowed_file(file.filename):
        return "Invalid file type", 400

    filename = secure_filename(file.filename)
    file_content = file.read()  # Read file content into memory
    file_metadata = {
        "filename": filename,
        "upload_date": datetime.datetime.now(),
        "file_type": filename.split('.')[-1],
        "file_content": file_content  # Store binary data
    }
    file_metadata_id = file_collection.insert_one(file_metadata).inserted_id

    # Determine file type and extract text accordingly
    if filename.endswith('.pdf'):
        pdf_text = extract_text_from_pdf(io.BytesIO(file_content))
    elif filename.endswith('.txt'):
        pdf_text = file_content.decode('utf-8')
    elif filename.endswith('.docx'):
        from docx import Document
        doc = Document(io.BytesIO(file_content))
        pdf_text = "\n".join([para.text for para in doc.paragraphs])
    else:
        return "Unsupported file type", 400

    # Split the document into chunks
    chunked_text = split_text(pdf_text)

    # Generate a unique collection name for each upload
    db_name = f"rag_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    db, name = create_chroma_db(documents=chunked_text, name=db_name)

    # Update latest collection name in MongoDB
    latest_collection_name = db_name
    chroma_collection_metadata.update_one(
        {"_id": file_metadata_id},
        {"$set": {"chroma_collection_name": db_name}}
    )

    return f"Document uploaded, processed, and indexed with collection name: {db_name}", 200

# Route to handle chat messages using RAG-based response generation
@app.route("/get", methods=["POST"])
def chat():
    global latest_collection_name  # Access the latest collection name

    user_message = request.form["msg"]

    # Check if a collection is available
    if not latest_collection_name:
        return "No document uploaded yet.", 400

    # Load the most recent ChromaDB collection
    db = load_chroma_collection(name=latest_collection_name)
    
    # Generate answer using RAG pipeline
    response = generate_answer(db=db, query=user_message)
    
    return response

if __name__ == "__main__":
    app.run(debug=True)
