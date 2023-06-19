import streamlit as st
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import openai
from functions import *

# Display the logo image in the second column
col1, col2 = st.columns([1,1])
#col1.image('logo-admin.png', width=300)
col1.title('PDF Q&A')
col2.subheader('Ask questions to your PDF documents | Powered by Chat GPT 3.5 turbo')

with st.expander("How it works ?"):
    st.markdown('''
                This tools aims to parse your own PDF documents, so that you can train LLMs on them and ask specific questions on their contents.\n
                - Provide an API key from Open AI. It costs some :money: but not so much. If you dont have one, sign up on https://openai.com/ and get USD / EUR 5 free credits.
                - Select a PDF document you want to talk to.
                - Ask a question
                - Get the model's answer ... and see the sources + pages that were used.


                Here below are the main tools / libraries that have been used:
                * **PyPDFLoader, RecursiveCharacterTextSplitter** for PDF loading and parsing into smaller chunks
                * **LangChain** for text preprocessing / pipeline
                * **OpenAIEmbeddings** for embedding chunks into vectors
                * **Chroma** vector database to store and search into vectors embedding (cost & time savings)
                * **LLM: gpt-3.5-turbo** ChatModel from OpenAI
                * **RetrievalQA** to call the model on chunks with highest similarity only instead of passing the full text (cost & time savings)

                ''')


st.subheader(f':closed_lock_with_key: API Key')
openai_key = st.text_input("Enter your OpenAI API key",type='password')
# Set the API key as an environment variable

### CHECK IF API KEY PROVIDED ###
if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key
    openai.api_key = os.environ["OPENAI_API_KEY"]
    st.markdown('API key loaded')

    st.subheader(f':open_file_folder:  PDF Uploader')
    pdf_file = st.file_uploader('Select your PDF document')

    ### CHECK IF PDF DOCUMENT UPLOADED ###
    if pdf_file:
        # Reset the prompt

        with open('temp.pdf', 'wb') as temp_file:
            temp_file.write(pdf_file.read())

        fn = str(pdf_file.name).split('.')[0] # get file name to be used as folder to save doc vectors

        ### PARSING ###
        documents, texts = parse_pdf('temp.pdf')
        st.write(f"{len(texts)} text chunks created from {len(documents)} pages found in the pdf...")


        from langchain.vectorstores import Chroma
        from langchain.embeddings import OpenAIEmbeddings
        # Embedding function
        open_ai_ef = OpenAIEmbeddings(model = 'text-embedding-ada-002',openai_api_key=openai_key) # default model - performing well and much cheaper

        ### EMBEDDING ###
        # Check if already in a persistent directory (file already loaded previously)
        # if not, create new vectordb
        if os.path.exists(fn) is False:
            with st.spinner("Creating the vectors embedding ..."):
                persist_directory = fn
                vectordb = Chroma.from_documents(texts,
                                embedding=open_ai_ef,
                                persist_directory=persist_directory)
                vectordb.persist()
                #vectordb = None
            st.write(f"Vectors embedding done. Stored in Vector DB.")

        # if True, load it instead of re-creating new embeddings
        else:
            persist_directory = fn
            vectordb = Chroma(persist_directory=persist_directory,embedding_function=open_ai_ef)
            st.write(f"File already found. Existing vectors have been loaded.")


        st.markdown('---')
        ### PROMPT ###
        st.subheader(f':thinking_face:  Question')
        prompt = st.text_input('Input your question here')

        ### GET ANSWER FROM MODEL ###
        if prompt:
            st.markdown('---')
            # Run Model for answer if prompt
            with st.spinner("Looking into this for you ..."):
                answer = get_answer(prompt,vectordb)
                st.subheader(f':rocket:  Answer')
                st.markdown(f'**Question:** {prompt}')
                st.write(answer['result'])

                with st.expander('Sources and Pages'):
                    st.write([st.write(f'Page:{doc.metadata["page"]+1}\n---------------------------\n{doc.page_content}') for doc in answer['source_documents']][0])
