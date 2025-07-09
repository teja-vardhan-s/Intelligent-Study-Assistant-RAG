import cohere
import faiss
import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
import docx
import ebooklib
from bs4 import BeautifulSoup
import logging
import os
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()

# Initialize the Cohere client
co = cohere.Client()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Streamlit UI
# Initialize session state for conversation history and document chunks/index
if "conversation" not in st.session_state:
    st.session_state["conversation"] = []
if "document_chunks" not in st.session_state:
    st.session_state["document_chunks"] = []
if "index" not in st.session_state:
    st.session_state["index"] = None
if "input_key" not in st.session_state:
    st.session_state["input_key"] = 0
if "last_answer" not in st.session_state:
    st.session_state["last_answer"] = None
if "processing" not in st.session_state:
    st.session_state["processing"] = False
if "processing_status" not in st.session_state:
    st.session_state["processing_status"] = ""

# Upload Documents
uploaded_files = st.file_uploader(
    "Upload your documents (PDFs, DOCs, EPUBs)",
    accept_multiple_files=True,
    type=["pdf", "docx", "epub"],
)


# Extract text from PDF
def extract_sections_from_pdf(file):
    sections = []
    current_section = ""

    # Open the PDF file in binary mode
    pdf_reader = PdfReader(file)

    # Iterate through each page
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text = page.extract_text()
        if not text:
            continue  # Skip pages with no text

        # Split text into lines
        lines = text.split("\n")

        for line in lines:
            if line.isupper() or line.strip() == "":
                if current_section:
                    sections.append(current_section.strip())
                    current_section = ""
            else:
                current_section += " " + line

    if current_section:
        sections.append(current_section.strip())

    return sections


# Extract text from Word (docx)
def extract_sections_from_docx(file_path):
    doc = docx.Document(file_path)
    sections = []
    current_section = ""

    for para in doc.paragraphs:
        if para.style.name.startswith("Heading"):
            if current_section:
                sections.append(current_section)
            current_section = ""
            sections.append(para.text)
        else:
            current_section += para.text + "\n"

    if current_section:
        sections.append(current_section)

    return sections


# Extract text from EPUB
def extract_sections_from_epub(file):
    book = ebooklib.epub.read_epub(file)
    sections = []

    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            text = soup.get_text()
            lines = text.split("\n")

            current_section = ""
            for line in lines:
                if line.strip() == "" or line.isupper():
                    if current_section:
                        sections.append(current_section.strip())
                        current_section = ""
                else:
                    current_section += " " + line.strip()

            if current_section:
                sections.append(current_section.strip())

    return sections


# Extract sections based on file type
def extract_sections(file):
    if file.name.endswith(".pdf"):
        return extract_sections_from_pdf(file)
    elif file.name.endswith(".docx"):
        return extract_sections_from_docx(file)
    elif file.name.endswith(".epub"):
        return extract_sections_from_epub(file)
    else:
        return "Unsupported file type."


# Generate embeddings for document chunks
def get_embeddings(text_list):
    embeddings = []
    for i in range(0, len(text_list), 50):
        batch = text_list[i : i + 50]
        response = co.embed(model="embed-english-v2.0", texts=batch)
        embeddings.extend(response.embeddings)
        time.sleep(1)
    return embeddings


def process_documents(uploaded_files):
    st.session_state["document_chunks"] = []  # Clear previous chunks
    st.session_state["processing"] = True
    st.session_state["processing_status"] = "Processing started"

    try:
        st.session_state["processing_status"] = "Processing files..."
        logging.info(st.session_state["processing_status"])

        # Sequential document extraction
        for file in uploaded_files:
            document_chunks = extract_sections(file)
            st.session_state["document_chunks"].extend(document_chunks)
            if document_chunks:
                logging.info(
                    f"Extracted {len(document_chunks)} chunks from {file.name}."
                )
            else:
                logging.warning(f"No text extracted from {file.name}.")

    except Exception as e:
        logging.error(f"Error during extraction: {str(e)}")
        st.error("An error occurred while processing the files. Please try again.")
        st.session_state["processing"] = False
        return

    # Generate embeddings after extraction...
    try:
        st.session_state["processing_status"] = "Generating embeddings..."
        document_embeddings = get_embeddings(st.session_state["document_chunks"])
        document_embeddings_np = np.array(document_embeddings).astype("float32")

        # Initialize FAISS index and add embeddings
        st.session_state["index"] = faiss.IndexFlatL2(document_embeddings_np.shape[1])
        st.session_state["index"].add(document_embeddings_np)

        st.success("Documents uploaded, chunked, and indexed successfully!")
    except Exception as e:
        logging.error(f"Error generating embeddings: {e}")
        st.error("An error occurred while generating embeddings. Please try again.")

    # End processing
    st.session_state["processing"] = False
    st.session_state["processing_status"] = "Processing complete"


# Start processing when files are uploaded
if uploaded_files and not st.session_state["processing"]:
    st.session_state["processing"] = True  # Set processing state to True
    process_documents(uploaded_files)

# Display a spinner or status message while processing is ongoing
if st.session_state["processing"]:
    with st.spinner(st.session_state["processing_status"]):
        st.write(st.session_state["processing_status"])


# Function to get embeddings (similarity search)
def get_query_embedding(query):
    response = co.embed(model="embed-english-v2.0", texts=[query])
    return response.embeddings[0]


# Perform similarity search with FAISS
def search_faiss(query, k=5, threshold=1.0):
    query_embedding = get_query_embedding(query)
    query_embedding_np = np.array([query_embedding]).astype('float32')
    distances, indices = st.session_state['index'].search(query_embedding_np, k)
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if dist < threshold:
            results.append(st.session_state['document_chunks'][idx])
    
    return results


# Function to generate answer based on retrieved text
def generate_answer(query, retrieved_texts):
    prompt = f"Question: {query}\n\nRelevant text: {retrieved_texts}\n\nAnswer:"
    response = co.chat(
        model="command-r-plus",  # Or use another supported chat model
        message=prompt,
        max_tokens=300,
        temperature=0.5,
    )
    return response.text.strip()


# Clear button functionality
if st.button("Clear All"):
    st.session_state["conversation"] = []
    st.session_state["document_chunks"] = []
    st.session_state["index"] = None
    st.session_state["input_key"] = 0  # Reset input key
    st.success("Chat and uploaded documents have been cleared. You can start fresh!")

print(f"input_{st.session_state['input_key']}")
print(f"Index:{st.session_state['index']}")

# Chatbot Interface
if st.session_state["index"] is not None:
    for q, a in st.session_state["conversation"]:
        st.markdown(
            f"""
            <div style="border: 1px solid #ccc; border-radius: 10px; padding: 10px; margin: 10px 0; background-color: #232423;">
                <div style="border: 1px ; border-radius: 10px; padding: 5px; margin: 5px 0; background-color: #2b2b2b;">
                    <strong>You:</strong> {q}<br>
                </div>
                <div style="border: 1px ; border-radius: 10px; padding: 5px; margin: 5px 0; background-color: #2b2b2b;">
                    <strong>Chatbot:</strong> {a}<br>
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    query = st.text_input(
        "Ask me anything:", key=f"input_{st.session_state['input_key']}"
    )

    if query:
        with st.spinner("Retrieving relevant sections..."):
            retrieved_chunks = search_faiss(query)
            if not retrieved_chunks:
                st.warning("Your question doesn't seem to match any content in the uploaded documents.")
            else:
                combined_texts = "\n".join(retrieved_chunks)
                answer = generate_answer(query, combined_texts)

                st.session_state["conversation"].append((query, answer))
                st.session_state["last_answer"] = answer

                st.session_state["input_key"] += 1

        st.query_params.dummy = st.session_state["input_key"]

        st.markdown(
            f"""
            <div style="border: 1px solid #ccc; border-radius: 10px; padding: 10px; margin: 10px 0; background-color: #232423;">
                <div style="border: 1px ; border-radius: 10px; padding: 5px; margin: 5px 0; background-color: #2b2b2b;">
                    <strong>You:</strong> {query}<br>
                </div>
                <div style="border: 1px ; border-radius: 10px; padding: 5px; margin: 5px 0; background-color: #2b2b2b;">
                    <strong>Chatbot:</strong> {answer}<br>
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )

        if st.button("Regenerate Answer"):
            if st.session_state["last_answer"] is not None:
                with st.spinner("Regenerating answer..."):
                    retrieved_chunks = search_faiss(query, k=5)
                    combined_texts = "\n".join(retrieved_chunks)
                    new_answer = generate_answer(query, combined_texts)

                    st.session_state["conversation"].append((query, new_answer))
                    st.session_state["last_answer"] = new_answer
            else:
                st.warning("No answer available to regenerate.")

else:
    st.info("Upload documents to start asking questions.")
