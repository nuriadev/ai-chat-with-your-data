import os
import tempfile
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize embeddings and LLM
embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
llm = ChatOpenAI(temperature=0, openai_api_key=API_KEY)

def load_documents(uploaded_file):
    """
    Save the uploaded file to a temporary path and load documents using the appropriate loader.

    Args:
        uploaded_file: Streamlit UploadedFile object with .read() and .name attributes.

    Returns:
        List of Document objects loaded from the file.
    """
    # Determine file extension
    file_name = uploaded_file.name
    _, ext = os.path.splitext(file_name)
    # Write to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Use appropriate loader based on extension
    if ext.lower() == ".pdf":
        loader = PyPDFLoader(tmp_path)
    else:
        loader = TextLoader(tmp_path)
    return loader.load()

def get_response(uploaded_file, question: str) -> str:
    """
    Build a FAISS index over the uploaded document
    and answer the user's question via a RetrievalQA chain.

    Args:
        uploaded_file: Uploaded file from Streamlit.
        question: User's query string.

    Returns:
        Answer string from the LLM.
    """
    docs = load_documents(uploaded_file)
    vectorstore = FAISS.from_documents(docs, embeddings)
    qa = RetrievalQA.from_chain_type(
        llm, chain_type="stuff", retriever=vectorstore.as_retriever()
    )
    return qa.run(question)
