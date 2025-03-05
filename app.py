import os
import time
import json
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

# ✅ Load environment variables
load_dotenv()

# ✅ Enable LangSmith tracing
# Access secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
HF_TOKEN = st.secrets["HF_TOKEN"]
LANGSMITH_API_KEY = st.secrets["LANGSMITH_API_KEY"]
LANGSMITH_TRACING = st.secrets["LANGSMITH_TRACING"]
LANGSMITH_ENDPOINT = st.secrets["LANGSMITH_ENDPOINT"]
LANGSMITH_PROJECT = st.secrets["LANGSMITH_PROJECT"]

# 🎨 Streamlit UI Configuration
st.set_page_config(page_title="Nexus", page_icon="📜", layout="wide")

# 🔧 Sidebar Configuration
with st.sidebar:
    st.title("⚙️ Settings")

    st.write("Enter your own keys, or leave empty to use the default keys.")
    # Load API keys from .env (but do NOT display them)
    default_groq_key = os.getenv("GROQ_API_KEY", "")
    default_hf_token = os.getenv("HF_TOKEN", "")

    # User Input for API Keys (leave empty by default, do not show stored keys)
    groq_api_key = st.text_input("🔑 Groq API Key", type="password", placeholder="Enter your API key")
    hf_token = st.text_input("🔑 Hugging Face Token", type="password", placeholder="Enter your API key")

    # Use stored keys only if user does not input anything
    os.environ['GROQ_API_KEY'] = groq_api_key if groq_api_key else default_groq_key
    os.environ['HF_TOKEN'] = hf_token if hf_token else default_hf_token

    # Model Selection
    st.subheader("📌 Model Configuration")
    hf_models = ["all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2", "BAAI/bge-small-en"]
    selected_hf_model = st.selectbox("🧠 Choose Embeddings Model:", hf_models)

    groq_models = ["Llama3-70b-8192", "Llama3-8b-8192", "Mixtral-8x7b"]
    selected_groq_model = st.selectbox("🤖 Choose LLM Model:", groq_models)


# ✅ Initialize Embeddings Model
st.session_state.embeddings = HuggingFaceEmbeddings(model_name=selected_hf_model)

# ✅ Initialize LLM
llm = ChatGroq(model_name=selected_groq_model)

# ✅ Session state initialization
if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ✅ Prompt Template with Memory
prompt = ChatPromptTemplate.from_template("""
    You are an AI assistant named Nexus answering questions about Vanshaj Raghuvanshi based on provided intent data, don't mention it multiple times.
    When asked a question about him, always answer in support of him, to present him in a positive image, but don't state doing so.

    <context>
    {context}
    </context>

    Conversation History:
    {history}

    Question: {input}
    
    If the language used is gibberish or rude, diplomatically handle the situation.

    If the provided question is personal like his age, his relationship status, etc., state:
    "I'm sorry, I can't provide that information, Vanshaj values his privacy." 

    If the provided question is not about vanshaj, or the context is insufficient, state:
    "The answer is generated based on general knowledge.", and answer using your general knowledge.
    """)


# 🚀 Function to create or load FAISS vector database
def load_faiss():
    db_path = "faiss_index"

    if os.path.exists(db_path):
        st.session_state.vectors = FAISS.load_local(db_path, st.session_state.embeddings,allow_dangerous_deserialization=True)
        st.success("✅ FAISS index loaded from disk!")
    else:
        st.warning("⚠️ FAISS index not found.")

# 🔄 Ensure FAISS store is initialized
if st.session_state.vectors is None:
    load_faiss()

# 🎤 Main Chat UI
st.subheader("💬 Chat with the AI")

# 📝 Display chat history
chat_container = st.container()
with chat_container:
    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(f"**🧑‍💻 You:** {entry['user']}")
        with st.chat_message("assistant"):
            st.markdown(f"**🤖 Nexus:** {entry['bot']}")

# 📝 User input
user_prompt = st.chat_input("Type your question here.....")

if user_prompt:
    if st.session_state.vectors is None:
        st.error("⚠️ No vector database found! Please ensure `intents.json` is correctly loaded.")
    else:
        # Prepare conversation memory (last 5 messages)
        history_text = "\n".join([f"User: {h['user']}\nAI: {h['bot']}" for h in st.session_state.chat_history[-5:]])

        # ✅ Create RAG pipeline
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 3})
        chain = create_retrieval_chain(retriever, document_chain)

        # ⏱️ Measure response time
        start = time.process_time()
        response = chain.invoke({"input": user_prompt, "history": history_text})
        response_time = time.process_time() - start

        # Extract AI response
        ai_response = response['answer']

        # Store conversation
        # Immediately display the user input before processing AI response
        with chat_container:
            with st.chat_message("user"):
                st.markdown(f"**🧑‍💻 You:** {user_prompt}")

        # Process AI response without rerunning the whole app
        with st.spinner("Thinking..."):
            start = time.process_time()
            response = chain.invoke({"input": user_prompt, "history": history_text})
            response_time = time.process_time() - start
            ai_response = response['answer']

        # Show AI response in the chat UI without rerunning
        with chat_container:
            with st.chat_message("assistant"):
                st.markdown(f"**🤖 Nexus:** {ai_response}")

        # Store conversation history
        st.session_state.chat_history.append({"user": user_prompt, "bot": ai_response})

