import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List
from pypdf import PdfReader
from dotenv import load_dotenv

# Vos imports LangChain habituels
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
try:
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains import create_retrieval_chain
except ImportError:
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
    from langchain_classic.chains.retrieval import create_retrieval_chain
app = FastAPI()
load_dotenv()
api_key = os.getenv("hugging_face_key")

# Stockage global (pour simuler la session_state de Streamlit)
class GlobalStore:
    retrieve_chain = None
    messages = []
    current_model = "LLAMA-2"

store = GlobalStore()

# Modèle pour la requête de question
class QuestionRequest(BaseModel):
    question: str
    model_choice: str  # "LLAMA-2", "GPT-4", ou "Hugging Face"

# --- ENDPOINT 1 : UPLOAD & INDEXATION ---
@app.post("/upload")
async def upload_pdf(files: List[UploadFile] = File(...)):
    try:
        pdf_content = ""
        for file in files:
            # Sauvegarde temporaire pour lecture
            temp_path = f"temp_{file.filename}"
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            reader = PdfReader(temp_path)
            for page in reader.pages:
                pdf_content += page.extract_text() or ""
            os.remove(temp_path)

        # Splitter
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(pdf_content)

        # Embeddings (CPU mode comme avant)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}
        )
        vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
        
        # On garde le retriever en mémoire
        store.retriever = vector_store.as_retriever()
        return {"message": f"{len(files)} fichiers indexés avec succès."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- ENDPOINT 2 : ASK ---
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    if not hasattr(store, 'retriever'):
        raise HTTPException(status_code=400, detail="Veuillez uploader un PDF d'abord.")

    try:
        # Configuration dynamique du modèle selon le choix Flutter
        if request.model_choice == "Hugging Face":
            llm = HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-beta", task="conversational", huggingfacehub_api_token=api_key)
        elif request.model_choice == "GPT-4":
            llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.3", task="text-generation", huggingfacehub_api_token=api_key)
        else:
            llm = ChatOllama(model="llama3.2", temperature=0.8)

        prompt = ChatPromptTemplate.from_template("Réponds à la question en utilisant le contexte : {context}\nQuestion : {input}")
        document_chain = create_stuff_documents_chain(llm, prompt)
        chain = create_retrieval_chain(store.retriever, document_chain)

        response = chain.invoke({"input": request.question})
        
        # Gestion de l'historique
        store.messages.append({"role": "user", "content": request.question})
        store.messages.append({"role": "assistant", "content": response["answer"], "model": request.model_choice})

        return {
            "answer": response["answer"],
            "model_used": request.model_choice,
            "history": store.messages
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- ENDPOINT 3 : RESET ---
@app.post("/reset")
async def reset_session():
    store.messages = []
    if hasattr(store, 'retriever'):
        del store.retriever
    return {"message": "Session réinitialisée."}