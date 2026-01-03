# Movie Trivia Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers movie trivia questions based on content from a PDF knowledge base using LangChain, Chroma vector store, and Ollama models.

## Features

- **PDF Ingestion**: Loads and processes `movies_trivia.pdf` using PyPDFLoader and CharacterTextSplitter
- **Vector Store**: Uses Chroma with Ollama embeddings for efficient document retrieval
- **LLM Integration**: Powered by Ollama's llama3.2:3b model for natural language responses
- **Chat History**: Maintains conversation context using InMemoryChatMessageHistory
- **LangSmith Tracing**: Integrated tracing for monitoring and debugging
- **Gradio UI**: User-friendly web interface with chat functionality

## Requirements

- Python 3.8+
- Ollama (with llama3.2:3b and granite-embedding:latest models)
- LangSmith API key (for tracing)

## Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root with:
   ```
   LANGCHAIN_API_KEY=your_langsmith_api_key_here
   ```

4. **Install and run Ollama**:
   - Download and install Ollama from https://ollama.ai/
   - Pull the required models:
     ```bash
     ollama pull llama3.2:3b
     ollama pull granite-embedding:latest
     ```

## Usage

1. **Ensure `movies_trivia.pdf` is in the project directory**

2. **Run the chatbot**:
   ```bash
   python movie_chatbot.py
   ```

3. **Access the web interface**:
   - Open the provided local URL (typically http://127.0.0.1:7860) in your browser
   - Start asking movie trivia questions!

## How It Works

1. **PDF Processing**: The PDF is loaded and split into chunks of 150 characters with 20-character overlap
2. **Vectorization**: Document chunks are embedded using Ollama's granite-embedding model and stored in Chroma
3. **Question Answering**: User questions are processed through a RetrievalQA chain that:
   - Retrieves relevant document chunks
   - Uses the custom prompt template
   - Generates responses via the llama3.2:3b model
4. **Chat History**: Previous messages are maintained and included in context for follow-up questions
5. **Tracing**: All interactions are logged via LangSmith for analysis

## Project Structure

```
movie-trivia-chatbot/
├── movie_chatbot.py          # Main application file
├── requirements.txt          # Python dependencies
├── movies_trivia.pdf         # Movie trivia knowledge base
├── movie_trivia_db/          # Chroma vector database (auto-generated)
├── .env                      # Environment variables (create this)
└── README.md                 # This file
```

## Dependencies

- langchain-community
- langchain-chroma
- langchain-text-splitters
- langchain-ollama
- langchain-core
- langchain-classic
- langsmith
- python-dotenv
- gradio

## Troubleshooting

- **Ollama connection issues**: Ensure Ollama is running and models are pulled
- **LangSmith errors**: Verify your API key in the `.env` file
- **PDF loading errors**: Check that `movies_trivia.pdf` exists and is readable
- **Port conflicts**: Gradio may use different ports if 7860 is occupied



