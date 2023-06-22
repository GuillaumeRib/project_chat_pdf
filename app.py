import streamlit as st
import streamlit_analytics
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import openai
from functions import *

streamlit_analytics.start_tracking()
# your streamlit code here

# Display the logo image in the second column
col1, col2 = st.columns([1,5])
col1.image('logo.png')
col2.title('PDF Assistant')
st.subheader("Ask questions to your PDF | Powered by Open AI's ChatGPT")

with st.expander("How it works"):
    st.markdown('''
                This app gives you the opportunity to ask questions about your own PDF documents' content.
                1. Provide an API key from OpenAI.
                - It incurs a cost, but you can utilize the free credits (USD 5) provided by OpenAI when signing up (https://openai.com/). Registration should take no more than 1-2 minutes.
                - The underlying models have been selected based on their strong performance-to-cost ratio (see pricing information and cost examples below).
                2. Select a PDF document with which you want to interact.
                3. Ask your question and get an answer from ChatGPT LLM together with sources and the corresponding page numbers.
                ''')

    st.markdown('---')

    st.markdown('''
                **OpenAI Pricing info**:
                - You can find up-to-date pricing information here: https://openai.com/pricing
                - The embedding model used is Ada v2, which provides a highly affordable option while maintaining accuracy.
                - The LLM model used is gpt-3.5-turbo, offering a great balance between cost savings and performance/accuracy (20 times cheaper than GPT-4).

                **Pricing example**:
                - Embedding a 90-page PDF and asking 5 questions (including answers) should cost approximately USD 0.02.
               ''')

    st.markdown('---')

    st.markdown('''
                Here below are the main tools / libraries that have been used:
                * **PyPDFLoader, RecursiveCharacterTextSplitter** for PDF loading and parsing into smaller chunks
                * **LangChain** for text preprocessing / pipeline
                * **OpenAIEmbeddings** for embedding chunks into vectors
                * **Chroma** vector database to store and search into vectors embedding (cost & time savings)
                * **LLM: gpt-3.5-turbo** ChatModel from OpenAI
                * **RetrievalQA** to call the model on chunks with highest similarity only instead of passing the full text (cost & time savings)
                ''')


st.subheader(f':closed_lock_with_key: API Key')
openai_key = st.text_input("Enter your OpenAI key and start talking to your PDF",type='password')


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
        with open('temp.pdf', 'wb') as temp_file:
            temp_file.write(pdf_file.read())

        fn = str(pdf_file.name).split('.')[0] # get file name to be used as folder to save doc vectors

        ### PARSING ###
        documents, texts = parse_pdf('temp.pdf')
        n_chunks = len(texts)
        st.write(f"{n_chunks} text chunks created from {len(documents)} pages found in the pdf...")

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
                answer = get_answer(prompt,vectordb,n_chunks)
                st.subheader(f':rocket:  Answer')
                st.markdown(f'**Question:** {prompt}')
                st.write(answer['result'])

                with st.expander('Sources and Pages'):
                    st.write([st.write(f'Page:{doc.metadata["page"]+1}\n---------------------------\n{doc.page_content}') for doc in answer['source_documents']][0])
else:
    sign_up_link = "https://platform.openai.com/signup?launch"
    st.markdown("No API key yet? [Sign up here](" + sign_up_link + ")")


streamlit_analytics.stop_tracking()
