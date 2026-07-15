import os
from pathlib import Path
from shutil import copyfileobj

import streamlit as st

with open(Path(__file__).with_name("styles.css"), "r", encoding="utf-8") as style_file:
    st.markdown(f"<style>{style_file.read()}</style>", unsafe_allow_html=True)
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from study_lens_utils import (
    create_chat_model,
    create_embeddings,
    format_model_error,
    is_quota_error,
    normalize_provider,
    resolve_api_key,
    resolve_model_name,
)

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "chroma_db"


def load_documents_from_dir(data_dir: Path):
    docs = []
    if not data_dir.exists():
        return docs

    for path in sorted(data_dir.iterdir()):
        if path.is_dir():
            continue

        if path.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(path)).load())
        elif path.suffix.lower() in {".txt", ".md", ".json"}:
            loader = TextLoader(str(path), encoding="utf-8")
            docs.extend(loader.load())

    return docs


def save_uploaded_pdf(uploaded_file):
    if uploaded_file is None:
        return None

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    destination = DATA_DIR / uploaded_file.name
    if destination.exists():
        destination = DATA_DIR / f"{Path(uploaded_file.name).stem}_uploaded.pdf"

    with destination.open("wb") as target_file:
        copyfileobj(uploaded_file, target_file)

    return destination


@st.cache_resource
def build_rag_chain():
    provider = normalize_provider(st.session_state.get("provider"))
    api_key = resolve_api_key(provider, st.session_state.get("api_key"))
    if not api_key:
        raise RuntimeError(
            f"Missing API key for provider '{provider}'. Add it to your .env file or enter it in the sidebar."
        )

    docs = load_documents_from_dir(DATA_DIR)
    if not docs:
        raise RuntimeError(
            "No documents were found in the data folder. Add a PDF or text syllabus file to data/."
        )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    try:
        embeddings = create_embeddings(provider, api_key)
        vector_store = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory=str(DB_DIR),
        )
    except Exception as exc:
        raise RuntimeError(f"Unable to create the embedding index for provider '{provider}'. Last error: {exc}") from exc

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    model_name = resolve_model_name(provider, st.session_state.get("model_name"))
    try:
        llm = create_chat_model(provider, api_key, model_name)
    except Exception as exc:
        raise RuntimeError(f"Unable to initialize the chat model for provider '{provider}'. Last error: {exc}") from exc

    prompt = ChatPromptTemplate.from_template(
        "You are an expert college study assistant. Use the retrieved context to answer the user. "
        "If the answer is not in the context, say that you do not know.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


default_provider = normalize_provider(os.getenv("PROVIDER", "gemini"))
if "provider" not in st.session_state:
    st.session_state.provider = default_provider
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "model_name" not in st.session_state:
    st.session_state.model_name = resolve_model_name(default_provider, None)

st.set_page_config(
    page_title="StudyLens RAG Chatbot",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items=None,
)

st.markdown(
    """
    <div class="study-lens-shell">
      <div class="hero-card">
        <div class="eyebrow">Smart study companion</div>
        <h1>StudyLens</h1>
        <p>Turn your syllabus and notes into a polished, private study assistant that answers questions from your own course materials.</p>
        <div class="hero-badges">
          <span class="badge">📄 PDF support</span>
          <span class="badge">🧠 Your own API key</span>
          <span class="badge">⚡ Fast answers</span>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-card">
      <h3>Upload course materials</h3>
      <p>Add a syllabus or lecture PDF to build your personal knowledge base for smarter study support.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.button("Add PDF"):
    st.session_state.show_upload_box = True

if st.session_state.get("show_upload_box", False):
    st.markdown(
        """
        <div class="upload-box">
            <div class="icon">📄</div>
            <div class="title">Drop PDF here</div>
            <div class="subtitle">Drag and drop your syllabus or course notes</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")
    if uploaded_file is not None:
        saved_path = save_uploaded_pdf(uploaded_file)
        if saved_path:
            build_rag_chain.clear()
            st.success(f"Saved {uploaded_file.name} to {saved_path} and rebuilt the knowledge base.")
            st.session_state.show_upload_box = False
        else:
            st.error("The upload could not be saved.")

if st.button("Rebuild knowledge base"):
    build_rag_chain.clear()
    st.success("Knowledge base cache cleared. The next query will rebuild the index.")

query = st.text_area("Ask a question", placeholder="When is the midterm exam? What is the grading policy?")

if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question first.")
    else:
        try:
            chain = build_rag_chain()
            answer = chain.invoke(query)
            st.subheader("Answer")
            st.write(answer)
        except Exception as exc:
            if is_quota_error(exc):
                st.error(format_model_error(exc, st.session_state.get("provider")))
            else:
                st.error(f"Could not answer the question: {exc}")
