import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import tempfile

embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

st.title("🤖 NOTEBOOK KLM CLONE")

fichier_upload = st.file_uploader("Upload file")


checkbox_activ_desac_llm = st.toggle("Activer/Désactiver le LLM")

st.sidebar.title("Historiques")

if checkbox_activ_desac_llm : 
    st.write(checkbox_activ_desac_llm)


boutton  = st.button("indexer")

if boutton and fichier_upload: 
    #stockage temporaire du fichier 
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as fichier_temp:
        fichier_temp.write(fichier_upload.getvalue())
        chemin_pdf = fichier_temp.name
    loader = PyMuPDFLoader(chemin_pdf)    
    documents = loader.load()

    #ajout du nom du fichier dans le metadata du document 
    for doc in documents:
        doc.metadata["source"] = fichier_upload.name

    #division des textes en chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )    

    chunks = text_splitter.split_documents(documents)
    
    #enregistrement dans la base de donnée vectorielle 
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory="./chroma_db"
    )


conversation = st.chat_input("Conversation")
st.write(conversation)

