# StudyLens

A personalized RAG (Retrieval-Augmented Generation) chatbot designed to help you study your own course materials. Built with Python, Streamlit, and LangChain, it supports multiple LLM providers like Gemini, OpenAI, and Anthropic.

> **Important:** Please read this README fully before downloading or running the code. To make the AI work, you **must** provide your own API key in a `.env` file as described in the setup instructions below. The application will not function without it.

Simply add your syllabus, notes, or other documents to the `data` folder, and the app will create a searchable knowledge base, allowing you to ask questions and get answers based on your content.

## Features

- **Interactive Chat Interface**: A user-friendly web UI built with Streamlit.
- **Multi-Provider Support**: Natively supports Google Gemini, OpenAI, and Anthropic models.
- **Flexible Document Loading**: Ingests both PDF and plain text (`.txt`, `.md`) files.
- **Local Vector Storage**: Uses ChromaDB to store document embeddings locally in the `chroma_db` folder.
- **Easy Configuration**: Simple setup using a `.env` file for API keys and model preferences.
- **File Upload**: Directly upload PDF files through the web interface.

## Project Structure

Here is an overview of the key files in the project:

```
StudyLens/
├── 📂 app.py                  # The main Streamlit web application.
├── 📂 hello_rag.py             # A command-line script to test the RAG pipeline directly.
├── 📂 study_lens_utils.py      # Helper functions for managing API providers, keys, and models.
│
├── 📂 data/
│   └── 📜 sample_syllabus.txt  # Folder to store your course materials (PDF, TXT).
│
├── 📂 chroma_db/               # (Generated) Local vector database created by Chroma.
│
├── 📜 .env.example             # Template for your environment variables.
├── 📜 requirements.txt         # A list of all the required Python dependencies.
│
└── 🧪 Diagnostic Scripts/
    ├── 📂 check_models.py      # Lists available Gemini models for your API key.
    ├── 📂 inspect_langchain.py # Provides information about your LangChain installation.
    └── 📂 test_quota_error.py   # A simple test for the error handling utilities.
```

## Setup and Usage

Follow these steps to get the application running.

### 1. Install Dependencies

First, install all the required Python packages from `requirements.txt`.

```bash
   pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy the `.env.example` file to a new file named `.env`.

```bash
cp .env.example .env
```

Open the `.env` file and:
1.  Set the `PROVIDER` to `gemini`, `openai`, or `anthropic`.
2.  Add your API key for the chosen provider (e.g., `GEMINI_API_KEY=...`).

3.  (Optional) Specify a default model to use (e.g., `GEMINI_MODEL=gemini-pro-latest`).

### 3. Add Your Documents

Place your course syllabus, notes, or other materials (PDF or `.txt` files) into the `data/` directory. A sample file is included to get you started.

### 4. Run the Application

Launch the Streamlit app from your terminal:

```bash
streamlit run app.py
```

The application will open in your web browser. The first time you ask a question, it will build the knowledge base from the documents in the `data` folder.

### Testing the RAG Pipeline

To test the core logic without the web interface, you can run the `hello_rag.py` script. This is useful for debugging.

```bash
python hello_rag.py
```
