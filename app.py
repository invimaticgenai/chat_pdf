import streamlit as st
import os
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 Chat with PDF')
    st.markdown('''
    ## About
    - Forget searching for information. 
    Simply upload your PDF and start asking questions. 
    InvimaticGPT will analyze the document and provide you clear, 
    concise answers in seconds. It's that easy.
    
    - [Invimatic](https://www.invimatic.com/)
    - [OpenAI](https://platform.openai.com/docs/models) 
    ''')
    add_vertical_space(5)
    st.write(' by INVIMATIC GenAi labs')

def process_query():
    query = st.session_state.query
    if query:
        st.session_state['chat_history'].append(f"You: {query}")
        docs = st.session_state.VectorStore.similarity_search(query=query, k=3)
        llm = OpenAI()
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=query)
            print(cb)
        st.session_state['chat_history'].append(f"Bot: {response}")
        st.session_state.query = ""  # Reset query input

def main():
    st.header("InvimaticGPT 💬")

    # Initialize chat history and VectorStore
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'VectorStore' not in st.session_state:
        st.session_state.VectorStore = None

    # Upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None and st.session_state.VectorStore is None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # Rebuild the FAISS object
        embeddings = OpenAIEmbeddings()
        st.session_state.VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

    # Display chat history
    for message in st.session_state['chat_history']:
        st.text(message)

    # User input at the bottom
    st.text_input("Ask questions about your PDF file:", key="query", on_change=process_query)

if __name__ == '__main__':
    main()
