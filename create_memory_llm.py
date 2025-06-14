from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

def load_pdf_files(data_path):
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data directory not found at {data_path}")
            
        loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader)
        documents = loader.load()
        
        if not documents:
            raise ValueError("No PDF documents found in the data directory")
            
        return documents
    except Exception as e:
        print(f"Error loading PDF files: {e}")
        raise

def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)

def main():
    try:
        # Load and process documents
        documents = load_pdf_files(DATA_PATH)
        text_chunks = create_chunks(documents)
        
        # Initialize embedding model
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}  # or 'cuda' if available
        )
        
        # Create and save vectorstore
        db = FAISS.from_documents(text_chunks, embedding_model)
        db.save_local(DB_FAISS_PATH)
        print(f"Vectorstore successfully created at {DB_FAISS_PATH}")
        
    except Exception as e:
        print(f"Error in vectorstore creation: {e}")
        exit(1)

if __name__ == "__main__":
    main()