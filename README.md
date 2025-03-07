# Nexus - Q&A Chatbot

Nexus is an AI-powered chatbot designed to answer questions about **Vanshaj Raghuvanshi**. It uses **FAISS** for efficient vector search and **Retrieval-Augmented Generation (RAG)** with Hugging Face embeddings and Groq's LLM models.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chat-nexus.streamlit.app)

## 🚀 Features

- **Retrieval-Augmented Generation (RAG)** for accurate Q&A
- **FAISS Vector Database** for efficient similarity search
- **Customizable API Keys** (Use your own or default keys)
- **Multiple Embeddings & LLM Models** to choose from
- **Streamlit UI** for easy interaction

## 🛠️ Setup Instructions

### 1️⃣ Install Dependencies
Ensure you have Python installed, then install the required packages:

```bash
pip install -r requirements.txt
```

### 2️⃣ Set Up API Keys
You can either:
1. **Use default API keys** (may expire).
2. **Provide your own API keys**:
   - Get a **Groq API Key**: [Groq API](https://console.groq.com/keys)
   - Get a **Hugging Face Token**: [Hugging Face](https://huggingface.co/settings/tokens)

#### 🔑 Setting API Keys Locally
Create a `.env` file in the project root and add:

```ini
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
```

#### 🔑 Setting API Keys for Streamlit Deployment
If deploying on **Streamlit Cloud**, add the API keys as secrets:

1. Go to **Streamlit Cloud → Your App → Edit Secrets**.
2. Add the following:
   ```ini
   GROQ_API_KEY="your_groq_api_key"
   HF_TOKEN="your_huggingface_token"
   ```

## 🔍 FAISS Database Creation

Before running the chatbot, you **must** create the FAISS vector database:

```bash
python create_faiss.py
```

This generates the required `faiss_index/` folder.

## ▶️ Running the Chatbot

Once FAISS is created, start the chatbot:

```bash
streamlit run app.py
```

## 📂 Project Structure

```
Nexus-QA/
├── faiss_index/
│   ├── index.faiss
│   └── index.pkl
├── .env
├── app.py
├── create_faiss.py
├── intents.json
└── requirements.txt
```
---

## 📜 License

This project is licensed under the MIT License.

