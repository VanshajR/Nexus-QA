import os
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Load Hugging Face API token
hf_token = os.getenv("HF_TOKEN", "")
selected_hf_model = "all-MiniLM-L6-v2"  # Default model

def process_intents():
    """Load and process intents.json"""
    with open("intents.json", "r", encoding="utf-8") as f:
        intents = json.load(f)

    docs = []
    for intent in intents['intents']:
        question = " ".join(intent['patterns'])
        answer = intent['responses'][0] 
        docs.append(f"Q: {question}\nA: {answer}")

    return docs

def create_faiss():
    """Create and save FAISS index"""
    print("🔄 Processing intents & creating FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name=selected_hf_model)

    docs = process_intents()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    final_docs = text_splitter.create_documents(docs)

    vector_store = FAISS.from_documents(final_docs, embeddings)
    vector_store.save_local("faiss_index")
    print("✅ FAISS index created successfully!")

if __name__ == "__main__":
    create_faiss()
