import os
from apikey import apikey

import streamlit as st
import chromadb

from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI 
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.chains import retrieval_qa
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chains.retrieval_qa.base import RetrievalQA


os.environ["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"] = apikey

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']


st.title('SupremeLending POC')
uploaded_file = st.file_uploader('upload a file: ',type=['pdf', 'docx', 'txt'])
add_file=st.button('Add File', on_click=clear_history)

if uploaded_file and add_file:
    bytes_data = uploaded_file.read()
    file_name=os.path.join('./SupremeDocuments/general/', uploaded_file.name)
    with open(file_name, 'wb') as f:
        f.write(bytes_data)

    name, extension = os.path.splitext(file_name)
    if extension == '.pdf':
        from langchain_community.document_loaders.pdf import PyPDFLoader
        loader = PyPDFLoader(file_name)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file_name)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader 
        loader = TextLoader(file_name)
    else:
        st.write('Document format is not supported!')    

    #loader = TextLoader('./SupremeDocuments/sample1.txt')
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embedding=embeddings)

    #llm = OpenAI(temperature=0)
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)

    retriever = vector_store.as_retriever()
    #chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    crc = ConversationalRetrievalChain.from_llm(llm,retriever=retriever)
    st.session_state.crc = crc
    st.success('File uploaded, chunked and embedded successfully')

question = st.text_input('Input your question')

if question:
    if 'crc' in st.session_state:
        crc = st.session_state.crc
        if 'history' not in st.session_state:
            st.session_state['history'] = []

        #response = chain.run(question)
        response = crc.run({'question':question,'chat_history':st.session_state['history']})

        st.session_state['history'].append((question,response))
        st.write(response)

        #st.write(st.session_state['history'])
        for prompts in st.session_state['history']:
            st.write("Question: " + prompts[0])
            st.write("Answer: " + prompts[1])
            