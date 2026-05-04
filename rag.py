from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.tools import tool
import tempfile
import os

embeddings = OpenAIEmbeddings()
vectorstore = None

def process_pdf(uploaded_file):
    global vectorstore

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(pages)

    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="./chroma_db"
    )

    os.unlink(tmp_path)
    return len(chunks)

@tool
def search_course_materials(query: str) -> str:
    """
    Search through the student's uploaded course materials and lecture notes.
    Use this when the student asks about their course content, lectures, or curriculum.
    """
    global vectorstore
    if vectorstore is None:
        return "No course materials uploaded yet. Please upload a PDF in the sidebar first."

    docs = vectorstore.similarity_search(query, k=3)
    if not docs:
        return "No relevant content found in the uploaded materials."

    results = "\n\n".join([
        f"📄 Page {doc.metadata.get('page', '?')+1}:\n{doc.page_content}"
        for doc in docs
    ])
    return f"Found relevant content from your course materials:\n\n{results}"