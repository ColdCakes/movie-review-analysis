import os
from langchain_community.document_loaders import PlaywrightURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
EMBED_MODEL = "all-MiniLM-L6-v2"


def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def load_and_index(urls: list[str]) -> Chroma:
    loader = PlaywrightURLLoader(urls, remove_selectors=["nav", "footer", "script", "style"])
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        chunks,
        _get_embeddings(),
        persist_directory=PERSIST_DIR,
    )
    return vectorstore


def load_existing() -> Chroma | None:
    if not os.path.exists(PERSIST_DIR):
        return None
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=_get_embeddings())
