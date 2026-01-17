import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.prompts import SYSTEM_PROMPT

# Load environment variables from .env file
load_dotenv()

def create_vector_store(chunks):
    """
    Creates a local vector database using HuggingFace embeddings.
    This runs on your CPU, so it doesn't use any API quota.
    """
    # Local embedding model (downloaded automatically on first run)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="vector_db"
    )
    return vector_db

def get_answer(query, vector_db):
    """
    Sends the user query and retrieved context to Groq's Llama 3 model.
    """
    # 1. Verify API Key exists
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found! Check your .env file in the root folder.")

    # 2. Initialize the Groq LLM
    # Llama-3.3-70b is smart, fast, and has a great free tier
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        groq_api_key=api_key
    )
    
    # 3. Define the Prompt Structure
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Context: {context}\n\nQuestion: {input}")
    ])

    # 4. Setup the Retriever
    # 'k=3' means it picks the top 3 most relevant chunks from your notes
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # 5. Build the RAG Chain
    chain = (
        {"context": retriever, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke(query)