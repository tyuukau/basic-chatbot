import chainlit as cl

from chainlit.types import AskFileResponse

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100
)
embedding = HuggingFaceEmbeddings()


def process_file(file: AskFileResponse):
    loaders = {"text/plain": TextLoader, "application/pdf": PyPDFLoader}

    loader_class = loaders.get(file.type)

    if loader_class:
        loader = loader_class(file.path)
        documents = loader.load()
        docs = text_splitter.split_documents(documents)
        for i, doc in enumerate(docs):
            doc.metadata["source"] = f"source_{i}"
        return docs

    else:
        raise ValueError(f"Unsupported file type: {file.type}")


def get_vector_db(file: AskFileResponse) -> Chroma:
    docs = process_file(file)
    cl.user_session.set("docs", docs)
    vector_db = Chroma.from_documents(documents=docs, embedding=embedding)
    return vector_db
