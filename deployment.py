import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
from streamlit_option_menu import option_menu
import os
import openai
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import pinecone
from langchain.vectorstores import Pinecone
import streamlit as st

# Load environment variables from .env file
load_dotenv()

# Initalize the OpenAI API Key and define the API endpoint to use
openai.api_key = os.getenv("OPENAI_API_KEY")
embeddings = HuggingFaceEmbeddings()
# initialize pinecone
pinecone.init(
    api_key="3fde37e8-29da-404c-bdb6-4fd500240c6b",  # find at app.pinecone.io
    environment= "gcp-starter"  # next to api key in console
)
index = Pinecone.from_existing_index('mandat', embeddings)
# semantic search
def get_similiar_docs(query,k=6,score=True):
  if score:
    similar_docs = index.similarity_search_with_score(query,k=6)
  else:
    similar_docs = index.similarity_search(query,k=6)
  return similar_docs
#ChatOpen AI
model_name = "gpt-3.5-turbo"
#model_name = "gpt-4"
#model_name = ' babbage-002'
llm = ChatOpenAI(model_name=model_name,temperature=0)

# LOGO_IMAGE = './bluspeed.png'
# st.set_page_config(layout="wide")
# #Disable Warning
# st.set_option('deprecation.showPyplotGlobalUse', False)
# st.markdown(
#     f"""
#     <div style="text-align: center;">
#     <img class="logo-img" src="data:png;base64,{base64.b64encode(open(LOGO_IMAGE, 'rb').read()).decode()}">
#     </div>
#     """,
#     unsafe_allow_html=True
# )
#Store Prompt
if 'messages' not in st.session_state:
    st.session_state.messages = []
#Historical Message
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])
#Menu
menu = option_menu(None, ["Home", "RAG Example", "Chatbot BLU"], 
    icons=['house', "newspaper", 'robot'], 
    menu_icon="cast", default_index=0, orientation="horizontal",
    styles={
        "container": {"padding": "0!important"},
        "icon": {"color": "white", "font-size": "15px"}, 
        "nav-link": {"font-size": "15   px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "blue"},
    })
if menu == "Home":
    st.title("Home")
    st.header("Abstract— ")
    st.write("This study aim for performance measurement utilizing a vector database and a chatbot. The Industrial Revolution 4.0 has introduced various technologies that greatly impact production processes, with big data and analytics being the most significant. Chatbots emerge as innovative and smart solutions, allowing for more efficient interaction with public services. The Directorate of Financial Management Development of Public Service Agencies (PPKBLU) plays a pivotal role in providing financial and services advisory, aiming to improve the financial management of Public Service Agencies (BLU). This research compares the performance of existing LLM models in answering questions related to regulations concerning BLU. Using a vector database, questions are assessed and answered by the LLM model, considering cosine similarity scores. The best-performing model, gpt-4, is selected for the deployment process.")
    st.write("-----------\n\nThis project uses generative AI enhanced with specific knowledge on PMK 129 Tahun 2020. Using prompt engineering, we trained this AI with specific information beyond the general knowledge base of chatGPT.")
                
    st.write('\n\n\n© 2023 Ibnu Pujiono and Irfan Murtadho Agtyaputra.')
    st.write('\n\n\nDisclaimer: Chatbot may produce inaccurate information outside the scope of topics it was trained on.')
    text = "*Magister Teknologi Informasi UI 2023.*"
    st.markdown(text)
if menu == "RAG Example":
    st.title("RAG Example 📄")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        raw_text = ''
        # Loop through each uploaded file
        for uploaded_file in uploaded_files:

            # Read the PDF
            pdf_reader = PdfReader(uploaded_file)

            # Loop through each page in the PDF
            for i, page in enumerate(pdf_reader.pages):

                # Extract the text from the page
                text = page.extract_text()
            
                # If there is text, add it to the raw text
                if text:
                    raw_text += text
                
        # Split text into smaller chucks to index them
        text_splitter = CharacterTextSplitter(
                        separator="\n", # line break
                        chunk_size = 1000,
                    # Striding over the text
                    chunk_overlap = 200,  
                    length_function=len,
        )
        
        texts = text_splitter.split_text(raw_text)
        
        # Download embeddings from OPENAI
        embeddings = HuggingFaceEmbeddings()
        
        # Create a FAISS vector store with all the documents and their embeddings
        docsearch = FAISS.from_texts(texts, embeddings)
        
        # Load the question answering chain and stuff it with the documents
        chain = load_qa_chain(OpenAI(), chain_type="stuff", verbose=True) 

        query = st.text_input("Ask a question or give an instruction")
        
        if st.button("Ask doc"):
            # Perform a similarity search to find the 6 most similar documents "chunks of text" in the corpus of documents in the vector store
            docs = docsearch.similarity_search(query, k=6)
            
            # Run the question answering chain on the 6 most similar documents based on the user's query
            answer = chain.run(input_documents=docs, question=query)
            
            # Print the answer and display the 6 most similar "chunks of text" vectors 
            with st.spinner('Wait for it...'):
                time.sleep(2)
                st.success('Done!') 
                st.write(answer, docs[0:6])
if menu == "Chatbot BLU":
    st.title("Chatbot BLU")
    retriever = index.as_retriever()
    #Create QA chain
    #query = st.text_input("Ask a question or give an instruction")
    #"Apa yang dimaksud dengan Pejabat Pengelola dalam konteks Badan Layanan Umum (BLU)?"

    chat_history = []
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=index.as_retriever(search_kwargs={"k": 6}),
    )
    # if st.button("Ask Chatbot BLU"):
    #     with st.spinner('Wait for it...'):
    #         time.sleep(2)
    #         st.success('Done!') 
    #         st.write(qa.run(query))
    prompt = st.chat_input("Tanya Chatbot BLU")
    #Show Prompt
    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user','content':prompt})
        response = qa.run(prompt)
        st.chat_message('assistant').markdown(response)
        st.session_state.messages.append({'role':'assistant','content':response})