import os
import fitz  # PyMuPDF
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text content.
    """
    pdf_document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        text += page.get_text("text")
    return text

# Function to split the text into manageable chunks
def split_text(text, chunk_size=1000, chunk_overlap=200):
    """
    Splits text into chunks.

    Args:
        text (str): Text to split.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The overlap between chunks.

    Returns:
        list: A list of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)

# Function to create embeddings and store them in a vector database
def create_vector_store(chunks):
    """
    Creates a vector store for the text chunks.

    Args:
        chunks (list): List of text chunks.

    Returns:
        FAISS: The vector store containing the embeddings.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store

# Function to set up the LangChain for question answering
def create_qa_chain(vector_store):
    """
    Creates a Conversational Retrieval Chain for QA.

    Args:
        vector_store (FAISS): The vector store containing the embeddings.

    Returns:
        ConversationalRetrievalChain: The QA chain.
    """
    chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=vector_store.as_retriever()
    )
    return qa_chain

# Streamlit interface
def main():
    st.set_page_config(page_title="Chat with Your Document", layout="wide")

    # Sidebar for API Key and File Upload
    with st.sidebar:
        st.header("Settings")
        google_api_key = st.text_input("Google API Key", type="password")
        uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
        
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key

    if uploaded_file is not None:
        # Save uploaded file to a temporary file
        with open("uploaded_document.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Extract text from the PDF
        text = extract_text_from_pdf("uploaded_document.pdf")
        
        # Split text into chunks
        chunks = split_text(text)
        
        # Create vector store and QA chain
        vector_store = create_vector_store(chunks)
        qa_chain = create_qa_chain(vector_store)
        
        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        st.title("Chat with Your Document")
        
        # Chat interface
        st.subheader("Chatbot Developed By Abdullah Mirza")
        
        # Display chat history
        for i, (q, a) in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                st.write(a)

        # User input
        query = st.chat_input("Type your question here...")

        if query:
            # Run the query through the QA chain
            response = qa_chain({"question": query, "chat_history": st.session_state.chat_history})
            
            # Update chat history
            st.session_state.chat_history.append((query, response['answer']))
            
            # Display the response
            with st.chat_message("assistant"):
                st.write(response['answer'])

if __name__ == "__main__":
    main()
