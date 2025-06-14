import os
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

def load_llm():
    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not api_token:
        raise ValueError("HuggingFace API token not found in environment variables")
    
    return HuggingFaceEndpoint(
        endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3",
        task="text-generation",
        max_new_tokens=512,
        temperature=0.5,
        huggingfacehub_api_token=api_token
    )

CUSTOM_PROMPT = """Use this medical context to answer professionally. If unsure, say you don't know.
Context: {context}
Question: {question}
Concise medical answer:"""

def load_vectorstore():
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        return FAISS.load_local(
            "vectorstore/db_faiss",
            embedding_model,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"Vectorstore loading failed: {e}")
        exit(1)

def create_qa_chain(db):
    prompt = PromptTemplate(
        template=CUSTOM_PROMPT,
        input_variables=["context", "question"]
    )
    
    return RetrievalQA.from_chain_type(
        llm=load_llm(),
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

def main():
    try:
        db = load_vectorstore()
        qa_chain = create_qa_chain(db)
        
        while True:
            query = input("\nMedical question (or 'quit' to exit): ").strip()
            if query.lower() in ['quit', 'exit']:
                break
                
            if not query:
                print("Please enter a valid question.")
                continue
                
            response = qa_chain.invoke({"query": query})
            print(f"\nANSWER: {response['result']}\n")
            print("SOURCES:")
            for doc in response["source_documents"]:
                print(f"- {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'N/A')})")
                
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()