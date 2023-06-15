import streamlit as st
import os

openai_key = st.text_input('Input your OpenAI API key here')
os.environ["OPENAI_API_KEY"] = openai_key


file_path = TEMP

# From raw PDF doc, load, split into smaller chunks before passing to Vectorizer & LLM
from langchain.document_loaders import PyPDFLoader,UnstructuredPDFLoader
loader = PyPDFLoader(file_path)
documents = loader.load()
print(len(documents))

# Split the docs in chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter #text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
texts = text_splitter.split_documents(documents[0:135])

# embedding function
from langchain.embeddings import OpenAIEmbeddings
open_ai_ef = OpenAIEmbeddings(model = 'text-embedding-ada-002') # default model - performing well and much cheaper

# Chroma DB - Vectors DB
from langchain.vectorstores import Chroma
store = Chroma.from_documents(texts,
                              embedding=open_ai_ef,
                              collection_name='pdf_app',
                             ) #store vectors into ChromaDB

# Select llm model
from langchain.chat_models import ChatOpenAI
# Chat Model
# "gpt-4, gpt-4-0613, gpt-4-32k, gpt-4-32k-0613, gpt-3.5-turbo, gpt-3.5-turbo-0613, gpt-3.5-turbo-16k, gpt-3.5-turbo-16k-0613"
chat_open_ai_llm = ChatOpenAI(model='gpt-3.5-turbo',temperature=0.2) #init llm with temperature argument (control creativity)



# Select number of chunks to pass to llm model after similarity performed on embeddings
k = 8
retriever = store.as_retriever(search_type='similarity',  # similarity or mmr
                               search_kwargs={'k':k}
                              )


# To select only the chunks with high similarity
from langchain.chains import RetrievalQA
qa = RetrievalQA.from_chain_type(
    llm=chat_open_ai_llm,
    chain_type='stuff', #refine / stuff (no refine for gpt 3.5 turbo)
    retriever= retriever,
    return_source_documents=True,
)

# Prompt
prompt = st.text_input('Input your question here')

if prompt:
    # Run Model for answer if prompt
    answer = qa(prompt)

    # Output on screen
    st.write((answer['result'])
