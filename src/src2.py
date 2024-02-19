from PyPDF2 import PdfReader
import tiktoken
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain

# Functions for the page:2_Text_Summarization.py.

def txt_splt(text,split_size):
    """
    Splits text into smaller documents of specified size.

    Parameters:
    - text (str): The text to be split into smaller documents.
    - split_size (int): The desired size of each split document.

    Returns:
    - docs (list of Document): List containing smaller documents.

    Example:
    >>> text = "This is a long piece of text that needs to be split into smaller documents."
    >>> split_size = 20
    >>> docs = txt_splt(text, split_size)
    >>> print(docs)
    [Document(page_content='This is a long pie'), Document(page_content='ce of text that needs'), Document(page_content=' to be split into small'), Document(page_content='er documents.')]
    """
    text_splitter = TokenTextSplitter(chunk_size=split_size, chunk_overlap=0)
    texts = text_splitter.split_text(text)
    docs = [Document(page_content=t) for t in texts]
    return docs

def token_count(text):
    """
    Calculates the number of tokens in the given text.

    Parameters:
    - text (str): The text for which token count is to be calculated.

    Returns:
    - count (int): The number of tokens in the text.

    Example:
    >>> text = "This is a sample sentence."
    >>> count = token_count(text)
    >>> print(count)
    6
    """
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))

def summarizer(llm_model, temperature, prompt, docs, type,refine_prompt=None):
    """
    Summarizes documents using a language model-based summarization approach.

    Parameters:
    - llm_model (str): Name or identifier of the language model.
    - temperature (float): Temperature parameter for language model generation.
    - prompt (str): Prompt for the summarization task.
    - docs (list of Document): List of documents to be summarized.
    - type (str): Type of summarization approach ('stuff', 'map_reduce', or other).
    - refine_prompt (str): Optional prompt for refining the summarization.

    Returns:
    - summary (str): Summarized text.

    Example:
    >>> llm_model = "gpt-3.5-turbo"
    >>> temperature = 0.7
    >>> prompt = "Summarize the given documents."
    >>> docs = [Document(page_content='Document 1 content'), Document(page_content='Document 2 content')]
    >>> type = 'map_reduce'
    >>> refine_prompt = "Refine the summary further."
    >>> summary = summarizer(llm_model, temperature, prompt, docs, type, refine_prompt)
    >>> print(summary)
    'Summarized content based on the provided documents and prompts.'
    """
    llm = ChatOpenAI(model_name=llm_model,temperature=temperature)
    if type == 'stuff':
        chain = load_summarize_chain(llm, chain_type=type, prompt=prompt)
    elif type == "map_reduce":
        chain = load_summarize_chain(llm, chain_type=type, map_prompt=prompt, combine_prompt=prompt)
    else:
        chain = load_summarize_chain(llm, chain_type=type, question_prompt=prompt, refine_prompt=refine_prompt)
    return chain.run(docs)