# notebooklm-clone

## Installation des dépendances

```bash
pip install langchain langchain-community langchain-text-splitters langchain-huggingface langchain-chroma chromadb sentence-transformers streamlit pymupdf
```

## Fonctionnement

Les documents PDF sont découpés en chunks (`chunk_size=1000`, `chunk_overlap=200`) puis enregistrés dans ChromaDB avec le metadata `source` contenant le nom du fichier original.

## Prérequis

Ollama doit tourner en arrière-plan. Installe-le sur [ollama.com](https://ollama.com) puis lance le modèle Mistral :

```bash
ollama run mistral
```