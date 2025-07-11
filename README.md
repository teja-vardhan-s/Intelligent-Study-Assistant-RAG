# Intelligent Study Assistant

## Overview

The Intelligent Study Assistant is an AI-powered application designed to help students efficiently retrieve and summarize information from various document types, including PDFs, Word documents, and EPUBs. By leveraging natural language processing and machine learning techniques, the application allows users to ask questions and receive accurate answers based on the contents of their uploaded materials.

## Features

- **Document Upload**: Supports multiple file formats (PDF, DOCX, EPUB).
- **Text Extraction**: Extracts text from documents and segments it into manageable sections.
- **AI-Powered Q&A**: Utilizes the Cohere API to generate relevant answers based on user queries.
- **Embeddings and Similarity Search**: Creates embeddings for text chunks and performs similarity searches to find relevant sections.
- **User-Friendly Interface**: Built using Streamlit for a seamless user experience.

## Technologies Used

- Python
- Streamlit
- Cohere API
- FAISS (Facebook AI Similarity Search)
- PyPDF2, python-docx, ebooklib, BeautifulSoup

## Installation

To set up the Intelligent Study Assistant locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/teja-vardhan-s/Intelligent-Study-Assistant-RAG.git
   cd Intelligent-Study-Assistant-RAG
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up your Cohere API key in the environment:

   ```bash
   export CO_API_KEY='your_api_key_here'
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Open the application in your web browser.
2. Upload your documents using the file uploader.
3. Ask questions related to the content of your uploaded documents.
4. Retrieve relevant sections and receive answers generated by the AI.

## Future Enhancements

- User authentication to save favorite answers.
- Integration of more advanced LLMs for improved accuracy.
- Additional document formats support (e.g., TXT, HTML).
- Mobile-responsive design for accessibility.

## Acknowledgments

- Special thanks to the Cohere team for providing the API.
- Inspiration from various AI and educational technology resources.
