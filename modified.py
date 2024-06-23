import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()

os.environ['NVIDIA_API_KEY'] = os.getenv('NVIDIA_API_KEY')

llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

def vector_embedding():
    if "vectors" not in st.session_state:
        with st.spinner('Loading and processing documents...'):
            st.session_state.embeddings = NVIDIAEmbeddings()
            st.session_state.loader = PyPDFDirectoryLoader(r".\us_census")
            st.session_state.docs = st.session_state.loader.load()
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.success('FAISS Vector Store DB is ready, used NVIDIA Embedding!!')

# Set page configuration
st.set_page_config(
    page_title="NVIDIA NIM Demo",
    page_icon=":robot_face:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Header
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: #3e7fe2;'>NVIDIA NIM Demo</h1>
        <p style='font-size: 18px;'>An interface to interact with NVIDIA AI models</p>
    </div>
    <hr style='border: 1px solid #3e7fe2;'>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    
    st.header("Options")
    if st.button("Create Embedding"):
        vector_embedding()
    # st.write("FAISS Vector Store DB is ready,used NVIDIAEmbedding!!")

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Provide the most accurate response for the question.
<context>
{context}
<context>
Questions: {input}
"""
)

# Main content
st.markdown("## Ask Your Questions")
prompt1 = st.text_input("Fire the Questions!!: ")

if prompt1:
    import time
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    with st.spinner('Processing your question...'):
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        response_time = time.process_time() - start

    st.markdown(f"### Response (Time: {response_time:.2f} seconds)")
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(f"**Document {i+1}**")
            st.write(doc.page_content)
            st.write("<hr>", unsafe_allow_html=True)
