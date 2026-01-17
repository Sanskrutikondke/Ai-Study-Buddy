import os

def initialize_folders():
    folders = ["data/uploads", "data/processed", "vector_db"]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

def save_uploaded_file(uploaded_file):
    path = os.path.join("data/uploads", uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path