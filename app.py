import streamlit as st
import os
import glob
from dotenv import load_dotenv

from src.utils import initialize_folders, save_uploaded_file
from src.processor import process_file
from src.brain import create_vector_store, get_answer

load_dotenv()
initialize_folders()

# --- AUTO-LOAD ALL CSV PARTS ---
@st.cache_resource
def load_all_datasets():
    # Finds all files starting with 'ai_flashcards' in your uploads folder
    files = glob.glob("data/uploads/ai_flashcards_notes_dataset_v*.csv")
    
    if files:
        all_chunks = []
        # Create a progress bar for the 3 parts
        progress_text = f"Analyzing {len(files)} datasets..."
        my_bar = st.sidebar.progress(0, text=progress_text)
        
        for idx, file in enumerate(files):
            chunks = process_file(file)
            all_chunks.extend(chunks)
            my_bar.progress((idx + 1) / len(files))
        
        return create_vector_store(all_chunks)
    return None

st.set_page_config(page_title="AI Study Buddy", page_icon="🎓", layout="wide")
st.title("📚 AI-Powered Study Buddy")

# Initialize the "Brain" automatically
if "vector_db" not in st.session_state:
    db = load_all_datasets()
    if db:
        st.session_state.vector_db = db
        st.sidebar.success("✅ All Dataset Parts Loaded!")
    else:
        st.sidebar.warning("No datasets found in data/uploads/")

# Sidebar for manual uploads
with st.sidebar:
    st.header("Add Extra Notes")
    uploaded_file = st.file_uploader("Upload PDF/CSV", type=["pdf", "csv"])
    if uploaded_file and st.button("Train on New File"):
        with st.spinner("Processing..."):
            path = save_uploaded_file(uploaded_file)
            new_chunks = process_file(path)
            # If DB exists, add to it; otherwise create it
            if "vector_db" in st.session_state:
                st.session_state.vector_db.add_documents(new_chunks)
            else:
                st.session_state.vector_db = create_vector_store(new_chunks)
            st.success("Memory Updated!")

# Chat Interface
if "vector_db" in st.session_state:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Searching notes..."):
                response = get_answer(user_input, st.session_state.vector_db)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})