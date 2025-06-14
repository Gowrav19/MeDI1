import os
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Constants
DB_FAISS_PATH = "vectorstore/db_faiss"
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
CUSTOM_PROMPT_TEMPLATE = """
Use the following medical context to answer the question professionally.
If you don't know the answer, say you don't know - don't make up answers.

Context: {context}
Question: {question}

Provide a concise medical answer:
"""

@st.cache_resource
def get_vectorstore():
    try:
        return FAISS.load_local(
            DB_FAISS_PATH,
            HuggingFaceEmbeddings(
                model_name='sentence-transformers/all-MiniLM-L6-v2',
                model_kwargs={'device': 'cpu'}
            ),
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error(f"Vector store error: {str(e)}")
        return None

def set_custom_prompt():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

@st.cache_resource
def load_llm():
    try:
        api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not api_token:
            st.error("API token not configured")
            return None
            
        return HuggingFaceEndpoint(
            repo_id=HUGGINGFACE_REPO_ID,
            temperature=0.5,
            max_new_tokens=512,
            huggingfacehub_api_token=api_token,
            task="text-generation"
        )
    except Exception as e:
        st.error(f"LLM loading error: {str(e)}")
        return None

def initialize_chat():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        st.session_state.sources = []
        st.session_state.messages.append({
            'role': 'assistant',
            'content': "Hello! I'm your medical assistant. How can I help you today?"
        })

def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
            
            # Display sources if available
            if message['role'] == 'assistant' and st.session_state.sources:
                with st.expander("Reference Sources"):
                    for doc in st.session_state.sources[-1]:
                        st.write(f"- {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'N/A')})")

def process_user_query(prompt):
    try:
        vectorstore = get_vectorstore()
        llm = load_llm()
        
        if not vectorstore or not llm:
            return "System error - please try again later"
            
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': set_custom_prompt()}
        )

        response = qa_chain.invoke({'query': prompt})
        st.session_state.sources.append(response["source_documents"])
        return response["result"]
        
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    st.set_page_config(
        page_title="Medical Chatbot",
        page_icon="🩺",
        layout="centered"
    )
    st.title("Medical Chatbot Assistant 🏥")
    
    initialize_chat()
    display_chat_history()
    
    if prompt := st.chat_input("Ask your medical question..."):
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        
        response = process_user_query(prompt)
        st.session_state.messages.append({'role': 'assistant', 'content': response})
        
        st.rerun()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")