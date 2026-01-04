# Coding Flow and Steps for Movie Trivia Chatbot

## Overview
This document outlines the step-by-step process followed to develop the Movie Trivia Chatbot, a Retrieval-Augmented Generation (RAG) application using LangChain, Chroma vector store, and Ollama models.

## Project Setup Steps

### 1. Environment Preparation
- Set up Python virtual environment
- Install required dependencies from `requirements.txt`
- Configure environment variables in `.env` file (LangSmith API key)
- Install and configure Ollama with required models (llama3.2:3b and granite-embedding:latest)

### 2. Project Structure Creation
- Create main application file `movie_chatbot.py`
- Organize project directory with necessary files:
  - `README.md` for documentation
  - `requirements.txt` for dependencies
  - `.gitignore` for version control
  - `movies_trivia.pdf` as knowledge base

## Implementation Steps

### 3. PDF Processing Pipeline
- Implement `load_and_split_pdf()` function using PyPDFLoader
- Configure CharacterTextSplitter with chunk_size=150 and chunk_overlap=20
- Process the movie trivia PDF into manageable text chunks

### 4. Vector Database Setup
- Create `build_or_load_vectorstore()` function
- Initialize Chroma vector store with Ollama embeddings
- Implement persistence to `movie_trivia_db/` directory
- Add document chunks to collection if database is empty

### 5. LLM and Chain Configuration
- Set up `get_llm()` function with ChatOllama (llama3.2:3b model)
- Create custom prompt template for movie trivia context
- Build RetrievalQA chain with:
  - Vector store retriever (k=3)
  - Custom prompt template
  - "Stuff" chain type

### 6. Chat History Management
- Implement InMemoryChatMessageHistory for conversation context
- Create `answer_question()` function with LangSmith tracing
- Format chat history for inclusion in queries

### 7. Gradio UI Development
- Design chat interface with chatbot component
- Implement `gradio_chat()` function for message handling
- Add clear history functionality
- Configure submit and clear button interactions

### 8. Application Initialization
- Create `init_pipeline()` function to orchestrate:
  - PDF loading and splitting
  - Vector store building/loading
  - LLM initialization
  - QA chain construction
- Set up global QA chain variable

## Key Technical Decisions

### 9. Architecture Choices
- RAG approach for accurate, context-aware responses
- Chroma for efficient vector storage and retrieval
- Ollama for local LLM and embedding models
- Gradio for user-friendly web interface
- LangSmith for tracing and monitoring

### 10. Configuration Parameters
- Chunk size: 150 characters with 20-character overlap
- Retrieval k: 3 most relevant documents
- Temperature: 0.2 for consistent responses
- Collection name: "movie_trivia"

## Testing and Validation

### 11. Functionality Testing
- Test PDF loading and text splitting
- Verify vector store creation and persistence
- Validate LLM responses with sample queries
- Check chat history maintenance
- Test Gradio interface interactions

### 12. Error Handling
- Implement checks for PDF file existence
- Handle Ollama connection issues
- Manage LangSmith API key validation
- Add graceful error messages for user

## Deployment and Usage

### 13. Local Deployment
- Run `python movie_chatbot.py`
- Access Gradio interface at local URL
- Verify all components work together





