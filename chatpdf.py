import os
import openai
import time
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

import streamlit as st

# Load environment variables from .env file
load_dotenv()

# Initalize the OpenAI API Key and define the API endpoint to use
openai.api_key = os.getenv("OPENAI_API_KEY")

# Streamlit Code for UI - Upload PDF(s)
st.title('ChatPDF :microphone:')
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
    embeddings = OpenAIEmbeddings() # Default model "text-embedding-ada-002"
    
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