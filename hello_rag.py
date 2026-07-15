import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
except ImportError:
    from langchain_community.embeddings import GoogleGenAIEmbeddings as GoogleGenerativeAIEmbeddings

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file. Please create one from .env.example.")
os.environ.setdefault("GOOGLE_API_KEY", api_key)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def load_documents_from_dir(data_dir: Path):
    """Loads documents from a directory, supporting PDF and text files."""
    docs = []
    if not data_dir.exists():
        return docs

    print(f"Loading documents from: {data_dir}")
    for path in sorted(data_dir.iterdir()):
        if path.is_dir():
            continue

        if path.suffix.lower() == ".pdf":
            print(f"  - Loading PDF: {path.name}")
            docs.extend(PyPDFLoader(str(path)).load())
        elif path.suffix.lower() in {".txt", ".md", ".json"}:
            print(f"  - Loading text: {path.name}")
            loader = TextLoader(str(path), encoding="utf-8")
            docs.extend(loader.load())

    return docs


docs = load_documents_from_dir(DATA_DIR)
if not docs:
    print("\nNo documents found in the data/ directory. Please add a syllabus or notes file.")
    exit()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

print(f"\nSplit {len(docs)} document(s) into {len(chunks)} chunks.")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

print("Created Chroma vector store and retriever.")

llm = ChatGoogleGenerativeAI(model="gemini-pro-latest", temperature=0)
print(f"Initialized chat model: {llm.model}")

prompt = ChatPromptTemplate.from_template(
    "You are an expert college study assistant. Use the retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

print("\nInvoking RAG chain...")
response = rag_chain.invoke("When is the mid-term exam and what is the grading policy?")
print("\n--- RAG Response ---")
print(response)
