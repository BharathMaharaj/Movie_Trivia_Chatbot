import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_classic.chains import RetrievalQA
from langsmith import traceable
import gradio as gr

load_dotenv()

# REQUIRED: LangSmith env vars
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Movie Trivia Chatbot"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

PDF_PATH = "movies_trivia.pdf"
CHROMA_DIR = "./movie_trivia_db"
COLLECTION_NAME = "movie_trivia"
OLLAMA_LLM_MODEL = "llama3.2:3b"
OLLAMA_EMBED_MODEL = "granite-embedding:latest"

# ------------- PDF INGESTION -----------------
def load_and_split_pdf(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=150, chunk_overlap=20)  # REQUIRED
    return splitter.split_documents(docs)

# ------------- VECTOR STORE -----------------
def build_or_load_vectorstore(chunks):
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)  # REQUIRED
    vector_db = Chroma(
        collection_name=COLLECTION_NAME,  # REQUIRED
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    if vector_db._collection.count() == 0:
        vector_db.add_documents(chunks)
        vector_db.persist()
    return vector_db

# ------------- LLM & CHAIN -----------------
def get_llm():
    return ChatOllama(model=OLLAMA_LLM_MODEL, temperature=0.2)

PROMPT_TEMPLATE = """
You are MovieBot, an expert movie trivia assistant.
Use ONLY the context below to answer the question.

Context:
{context}

Question: {question}
Answer:"""

qa_prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]  # REQUIRED format
)

def build_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    # REQUIRED: RetrievalQA with custom prompt template
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": qa_prompt},
        return_source_documents=False
    )

# ------------- CHAT HISTORY -----------------
chat_history = InMemoryChatMessageHistory()

@traceable  # REQUIRED
def answer_question(qa_chain, question: str) -> str:
    # Build history string from chat_history
    history_list = []
    for msg in chat_history.messages:
        role = 'Human' if msg.type == 'human' else 'AI'
        history_list.append(f"{role}: {msg.content}")

    history_str = "\n".join(history_list)
    full_question = f"Chat history:\n{history_str}\n\nQuestion: {question}"

    result = qa_chain.invoke({"query": full_question})
    return result["result"]

GLOBAL_QA_CHAIN = None

def gradio_chat(message, history):
    global GLOBAL_QA_CHAIN

    # Add to chat history
    chat_history.add_user_message(message)
    answer = answer_question(GLOBAL_QA_CHAIN, message)
    chat_history.add_ai_message(answer)

    # Append to Gradio history in dict format
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    return history, ""

def clear_chat():
    global chat_history
    chat_history.clear()
    return [], ""

# ------------- INIT -----------------
def init_pipeline():
    global GLOBAL_QA_CHAIN
    print("1. Loading PDF...")
    chunks = load_and_split_pdf(PDF_PATH)

    print("2. Building Chroma...")
    vector_db = build_or_load_vectorstore(chunks)

    print("3. Loading Ollama...")
    llm = get_llm()

    print("4. Building RetrievalQA...")
    GLOBAL_QA_CHAIN = build_qa_chain(vector_db, llm)

init_pipeline()

# ------------- GRADIO UI -----------------
with gr.Blocks(title="Movie Trivia Bot") as demo:
    gr.Markdown("# 🎬 MovieBot - Movie Trivia Expert")

    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(
        placeholder="Ask about movies from movies_trivia.pdf...",
        label="Your Question"
    )
    with gr.Row():
        submit_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear History")

    submit_btn.click(gradio_chat, inputs=[msg, chatbot], outputs=[chatbot, msg])
    msg.submit(gradio_chat, inputs=[msg, chatbot], outputs=[chatbot, msg])
    clear_btn.click(clear_chat, outputs=[chatbot, msg])

    gr.Markdown("**Hello, I am MovieBot, your movie trivia expert. Ask me anything about films!**")

if __name__ == "__main__":
    demo.launch(share=False)
