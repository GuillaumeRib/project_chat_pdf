# Project Description
- Name: project_chat_pdf

# Objective
- Ask question to any PDF you have & get the answer by Open AI chat-gpt-3.5-turbo LLM.
- Load, parse, extract PDF contents
- Use Vectors embedding, Vectors databases, and LLM models in a question / answer retrieval context.
- Build a Front-End app using **Streamlit** where the user can enter it's OpenAI API key and ask question to his/her own PDF doc.

# Data Source / Main libraries used
- Any PDF document
- PyPDFLoader, RecursiveCharacterTextSplitter for PDF loading and parsing into smaller chunks
- LangChain for text preprocessing / pipeline
- OpenAIEmbeddings for embedding chunks into vectors
- Chroma vector database to store and search into vectors embedding (cost & time savings)
- LLM: gpt-3.5-turbo ChatModel from OpenAI
- RetrievalQA to call the model on chunks with highest similarity only instead of passing the full text (cost & time savings)

# Type of analysis
- Data extraction / Document parsing / Vector Database/ NLP / LLM / Front-end app

# Front-end
- Live app: https://guillaumerib-project-chat-pdf-app.streamlit.app/
- App may need to wake-up first if not used recently. Thanks you for your patience.
