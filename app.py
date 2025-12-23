import streamlit as st
import os
from pypdf import PdfReader 
from dotenv import load_dotenv

# Imports pour LangChain 1.x / 0.3+
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# --- CHANGEMENT ICI POUR LES CHAÎNES ---
try:
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains import create_retrieval_chain
except ImportError:
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
    from langchain_classic.chains.retrieval import create_retrieval_chain

# Initialisation
load_dotenv() 
api_key = os.getenv("hugging_face_key")

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "model" not in st.session_state:
    st.session_state["model"] = "LLAMA-2"
if "retrieve_chain" not in st.session_state:
    st.session_state["retrieve_chain"] = None

# Interface utilisateur
st.set_page_config(page_title="EMSI Chatbot", layout="centered")
st.title("EMSI Chatbot")
st.markdown("Bienvenue dans le chatbot EMSI. Posez votre question ci-dessous !")

# Configuration Sidebar
with st.sidebar:
    st.header("Paramètres du Chatbot")
    model_choice = st.sidebar.selectbox(
        "Select a model", options=["LLAMA-2", "GPT-4", "Hugging Face"], index=0
    )
    st.session_state["model"] = model_choice
    
    max_tokens = st.sidebar.slider("Max Tokens", min_value=128, max_value=2048, value=512)
    docs_pdf = st.file_uploader(label="Upload file", accept_multiple_files=True)

# Traitement des PDF
if docs_pdf and st.session_state.retrieve_chain is None:
    pdf_content = ""
    for pdf in docs_pdf:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                pdf_content += text

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(pdf_content)

    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)

    # Configuration du modèle LLM
    if st.session_state["model"] == "Hugging Face":
        llm = HuggingFaceEndpoint(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            task="conversational",
            max_new_tokens=512,
            huggingfacehub_api_token=api_key
        )
    elif st.session_state["model"] == "GPT-4":
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            task="text-generation",
            max_new_tokens=max_tokens,
            huggingfacehub_api_token=api_key
        )
    else:
        llm = ChatOllama(model="llama3.2", temperature=0.8)

    prompt = ChatPromptTemplate.from_template("""
    Réponds à la question suivante en utilisant uniquement le contexte fourni :
    Contexte : {context}
    Question : {input}
    """)
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever()
    st.session_state.retrieve_chain = create_retrieval_chain(retriever, document_chain)

# --- 9. Affichage de l'historique avec indication du modèle ---
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        # On affiche le modèle en petit sous la réponse de l'assistant
        if msg["role"] == "assistant" and "model_used" in msg:
            st.caption(f"Généré par : {msg['model_used']}")

# --- 10. Gestion de la nouvelle question ---
question = st.chat_input("Ask a question:")

if question:
    if st.session_state.retrieve_chain is None:
        st.warning("Veuillez d'abord uploader un fichier PDF.")
    else:
        st.chat_message("user").write(question)
        
        try:
            with st.spinner("L'assistant réfléchit..."):
                response = st.session_state.retrieve_chain.invoke({"input": question})
                answer = response["answer"]
                current_model = st.session_state["model"]
                
                # Sauvegarder avec le nom du modèle utilisé 
                st.session_state["messages"].append({"role": "user", "content": question})
                st.session_state["messages"].append({
                    "role": "assistant", 
                    "content": answer,
                    "model_used": current_model # On enregistre quel modèle a répondu
                })
                
                # Affichage immédiat de la réponse
                with st.chat_message("assistant"):
                    st.write(answer)
                    st.caption(f"Généré par : {current_model}")
                
        except Exception as e:
            st.error(f"Error: {e}")