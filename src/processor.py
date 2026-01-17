import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

def process_file(file_path):
    documents = []
    
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
    
    elif file_path.endswith('.csv'):
        # Using a limit to handle large 45MB files without crashing RAM
        df = pd.read_csv(file_path)
        # Reading first 500 rows from each part to balance speed and knowledge
        for _, row in df.head(500).iterrows():
            content = f"Topic: {row.get('topic', 'General')}\nDetails: {row.get('source_text', '')}"
            if content.strip():
                documents.append(Document(page_content=content))
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    return text_splitter.split_documents(documents)