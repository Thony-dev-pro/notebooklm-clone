import streamlit as st

st.title("🤖 NOTEBOOK KLM CLONE")

fichier_upload = st.file_uploader("Posez votre question")

checkbox_activ_desac_llm = st.toggle("Activer/Désactiver le LLM")

st.sidebar.title("Historiques")

if fichier_upload:
    st.write("Fichier :")
    st.write(fichier_upload)

if checkbox_activ_desac_llm : 
    st.write(checkbox_activ_desac_llm)


st.button("indexer")

conversation = st.chat_input("Conversation")
st.write(conversation)

