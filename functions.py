def parse_pdf(temp_file):
    # From raw PDF doc, load, split into smaller chunks before passing to Vectorizer & LLM
    from langchain.document_loaders import PyPDFLoader
    loader = PyPDFLoader(temp_file)
    documents = loader.load()

    # Split the docs in chunks
    from langchain.text_splitter import RecursiveCharacterTextSplitter #text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    return documents, texts



def get_answer(prompt,vectordb,n_chunks):
    # Select llm model
    from langchain.chat_models import ChatOpenAI
    chat_open_ai_llm = ChatOpenAI(model='gpt-3.5-turbo',temperature=0.5)

    # Select number of chunks to pass to llm model after similarity performed on embeddings
    k = min([4,n_chunks])
    retriever = vectordb.as_retriever(search_type='similarity',
                                   search_kwargs={'k':k}
                                            )

    from langchain.chains import RetrievalQA
    qa = RetrievalQA.from_chain_type(
        llm=chat_open_ai_llm,
        chain_type='stuff', #refine / stuff (no refine for gpt 3.5 turbo)
        retriever= retriever,
        return_source_documents=True,
    )
    answer = qa(prompt)
    return answer
