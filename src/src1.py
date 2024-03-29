import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI

# Functions for the page:1_Chat_with_Documents.py.

def get_pdf_text(pdf_docs):
    """
    Extracts text from PDF documents.

    Parameters:
    - pdf_docs (list of str): A list of paths to PDF documents.

    Returns:
    - text (str): Concatenated text extracted from all the pages of the PDF documents.

    Example:
    >>> pdf_docs = ["document1.pdf", "document2.pdf"]
    >>> text = get_pdf_text(pdf_docs)
    >>> print(text)
    This is the text extracted from document1.pdf followed by the text extracted from document2.pdf.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text,chunk_size=1000,chunk_overlap=200):
    """
    Splits text into chunks of specified size with an optional overlap.

    Parameters:
    - text (str): The text to be split into chunks.
    - chunk_size (int): The desired size of each chunk. Default is 1000.
    - chunk_overlap (int): The amount of overlap between adjacent chunks. Default is 200.

    Returns:
    - chunks (list of str): List containing text chunks.

    Example:
    >>> text = "This is a long piece of text that needs to be split into smaller chunks."
    >>> chunks = get_text_chunks(text, chunk_size=20, chunk_overlap=5)
    >>> print(chunks)
    ['This is a long pi', 'long piece of text', 'text that needs to', 'needs to be split', 'split into smaller', 'smaller chunks.']
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    """
    Generates a vector store from text chunks using pre-trained embeddings.

    Parameters:
    - text_chunks (list of str): List containing text chunks.

    Returns:
    - vectorstore (FAISS): Vector store generated from the text chunks.

    Example:
    >>> text_chunks = ['chunk 1 text', 'chunk 2 text', 'chunk 3 text']
    >>> vectorstore = get_vectorstore(text_chunks)
    >>> print(vectorstore)
    <FAISS vector store object>
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks,embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore,llm_model,temperature,k):
    """
    Generates a conversation chain for conversational retrieval.

    Parameters:
    - vectorstore (FAISS): Vector store generated from text chunks.
    - llm_model (str): Name or identifier of the language model.
    - temperature (float): Temperature parameter for language model generation.
    - k (int): Number of nearest neighbors to retrieve from the vector store.

    Returns:
    - conversation_chain (ConversationalRetrievalChain): Conversation chain for conversational retrieval.

    Example:
    >>> vectorstore = <FAISS vector store object>
    >>> llm_model = "gpt-3.5-turbo"
    >>> temperature = 0.7
    >>> k = 5
    >>> conversation_chain = get_conversation_chain(vectorstore, llm_model, temperature, k)
    >>> print(conversation_chain)
    <ConversationalRetrievalChain object>
    """
    llm = ChatOpenAI(model_name=llm_model,temperature=temperature)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever = vectorstore.as_retriever(search_kwargs={'k': k}),
        memory = memory
    )
    return conversation_chain

def handle_userinput(user_question):
    """
    Handles user input by initiating a conversation with the AI model and updating the chat history.

    Parameters:
    - user_question (str): The question or input provided by the user.

    Returns:
    - None

    Example:
    >>> user_question = "What is the weather like today?"
    >>> handle_userinput(user_question)
    User: What is the weather like today?
    Assistant: The response generated by the AI model.
    """
    response = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
             with st.chat_message("user"):
                 st.markdown(message.content)
        else:
             with st.chat_message("assistant"):
                 st.markdown(message.content)