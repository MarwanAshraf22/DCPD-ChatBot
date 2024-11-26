import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# Streamlit page configuration
st.set_page_config(
    page_title="DCPD ChatBot",
    page_icon='logo.jpg',
    layout='centered'
)
st.title("DCPD ChatBot")
st.markdown("""
You can only ask questions about tournaments, results, and general information about Dubai Club for People of Determination.
""")

# Sidebar with club info
with st.sidebar:
    st.header("Dubai Club for People of Determination")
    st.image("logo.jpg")

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
def get_openai_client():
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1, openai_api_key=api_key)

# Extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        loader = PyMuPDFLoader(pdf)
        documents = loader.load()  # Returns a list of documents
        for doc in documents:
            text += doc.page_content  # Assuming page_content contains the extracted text
    return text

# Extract text from TXT files
def get_txt_text(txt_files):
    text = ""
    for txt_file in txt_files:
        with open(txt_file, 'r', encoding='utf-8') as file:
            text += file.read()
    return text

# Combine text from all sources
def get_combined_text(pdf_docs, txt_files):
    pdf_text = get_pdf_text(pdf_docs)
    txt_text = get_txt_text(txt_files)
    return pdf_text + "\n" + txt_text

# Split text into manageable chunks for vector storage
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

# Store text embeddings in vector database
def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Chain for question answering
def get_conversational_chain():
    prompt_template = """   
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the context, say, "Answer is not available in the context."

    Context:
    {context}
    Question: 
    {question}
    Answer:
"""


    model = get_openai_client()
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Process user questions
def handle_user_input(user_question):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# Main function
def main():
    with st.spinner("Processing documents..."):
        pdf_docs = ["documents/information.pdf", "documents/results.pdf"]  # List of paths to PDF files
        txt_files = ["documents/schedule.txt"]  # List of paths to TXT files
        try:
            # Extract and combine text from all files
            raw_text = get_combined_text(pdf_docs, txt_files)
            # Split the extracted text into manageable chunks
            text_chunks = get_text_chunks(raw_text)
            # Store the text chunks in the FAISS vector store
            get_vector_store(text_chunks)
        except Exception as e:
            st.error(f"Error processing the files: {e}")

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display chat history
    for message in st.session_state.messages:
        role, content = message["role"], message["content"]
        with st.chat_message(role):
            st.markdown(content)

    # Capture new user question
    if user_question := st.chat_input("Ask a question about the documents:"):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Get response from model
        with st.chat_message("assistant"):
            response = handle_user_input(user_question)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
