\# 🤖 AI Career Agent



An AI-powered career assistant that uses \*\*Retrieval-Augmented Generation (RAG)\*\* to provide career guidance based on a curated career knowledge base.



The application combines \*\*LangChain, ChromaDB, Hugging Face embeddings, Google Gemini and Streamlit\*\* to create an interactive career assistant.



\## 🚀 Features



\- 💬 Ask career-related questions through a web interface

\- 🔎 Semantic search using vector embeddings

\- 📚 Retrieval-Augmented Generation (RAG)

\- 🧠 Hugging Face sentence-transformer embeddings

\- 🗄️ ChromaDB vector database

\- 🔗 LangChain-based RAG pipeline

\- 🤖 Google Gemini integration

\- 🔄 Local Hugging Face model fallback when Gemini quota is unavailable

\- 📖 Displays retrieved knowledge used for the answer

\- 🖥️ Interactive Streamlit interface

\- 🔐 Environment variables for API key protection



\## 🏗️ Architecture



```text

&#x20;               User Question

&#x20;                     │

&#x20;                     ▼

&#x20;            Streamlit Web UI

&#x20;                     │

&#x20;                     ▼

&#x20;            LangChain RAG Pipeline

&#x20;                     │

&#x20;                     ▼

&#x20;             ChromaDB Vector Search

&#x20;                     │

&#x20;                     ▼

&#x20;         Relevant Career Knowledge

&#x20;                     │

&#x20;                     ▼

&#x20;             RAG Prompt Creation

&#x20;                     │

&#x20;                     ▼

&#x20;             Google Gemini API

&#x20;                     │

&#x20;               ┌─────┴─────┐

&#x20;               │           │

&#x20;            Available    Quota Error

&#x20;               │           │

&#x20;               ▼           ▼

&#x20;            Gemini    Local Hugging

&#x20;                        Face Model

&#x20;               │           │

&#x20;               └─────┬─────┘

&#x20;                     ▼

&#x20;                Final Answer

