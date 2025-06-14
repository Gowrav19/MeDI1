# 🏥 MediChatbot - AI-Powered Medical Assistant

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-00A67E?style=for-the-badge)
![HuggingFace](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)

An intelligent medical chatbot powered by Mistral-7B (via HuggingFace) and LangChain, with Streamlit web interface. Provides evidence-based medical answers using Retrieval-Augmented Generation (RAG).

## ✨ Features

- **Medical Q&A** - Get concise answers to medical questions
- **RAG Architecture** - Combines LLM with medical knowledge base
- **Secure** - No patient data storage
- **Responsive UI** - Clean Streamlit interface

## 🛠️ Tech Stack

- **Backend**: Python, LangChain, HuggingFace Inference API
- **Frontend**: Streamlit
- **LLM**: Mistral-7B-Instruct-v0.3
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector DB**: FAISS

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- [Pipenv](https://pipenv.pypa.io/) (or pip)
- HuggingFace API token

### Installation
```bash
git clone https://github.com/Gowrav19/MeDI1.git
cd MediChatbot

#First of all we need get all the required libraries
pipenv install langchain langchain_community langchain_huggingface faiss-cpu pypdf
pipenv install huggingface_hub
pipenv install streamlit

# Using pipenv (recommended)
pipenv install
pipenv shell

# Or with pip
pip install -r requirements.txt
# Run both the files
python run create__memory__llm.py #it creates the vector store
python run connect__memory__llm.py

#run the main appilcation
streamlit run medibot.py 
