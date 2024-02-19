import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import src.src2 as src 
from langchain.prompts import PromptTemplate

def main():
    load_dotenv()
    st.set_page_config(page_title="Summarize", page_icon=":receipt:")
    with st.sidebar:
        st.image("img/logo_sq.png")
        st.markdown("This is a portfolio project by Felipe Martins. If you want to see the code of this app and other data science projects check my [GitHub](https://github.com/felipebita).")
        st.markdown("This is just an example tool. Please, do not abuse on my OpenAI credits, use it only for testing purposes.")

    st.header("Text Summarization :receipt:")
    with st.expander("Model Options"):
        st.session_state.llm_model = st.radio('LLM model:',["gpt-3.5-turbo","gpt-4"],index=0,)
        st.session_state.temperature = st.slider('Temperature:', 0.0, 1.0, step=0.1, value=0.7)
        if st.session_state.llm_model == "gpt-3.5-turbo":
            st.session_state.split_size = st.slider('Split Size (tokens):', 200, 4000, step=100, value=2000)
        else:
            st.session_state.split_size = st.slider('Split Size (tokens):', 2000, 8000, step=100, value=4000)   

    st.write("""If your text is smaller than the split size, the method used for summarization is going to be 'stuff'. 
            If it is longer, the option to chose between 'map_reduce' and 'refine' is going to be available.""")
    
    txt_sum = st.text_area(
            "Text to summarize.",
            "Inser your text here."
            )
    st.write(f"""Your text has '{len(txt_sum)}' characters and '{src.token_count(txt_sum)}' tokens.""")

    txt_prompt = st.text_area(
            "Define your prompt.",
            """Write a concise summary of the following text:
            '{text}'""")
    prompt = PromptTemplate.from_template(txt_prompt)

    if src.token_count(txt_sum) > st.session_state.split_size:
        st.session_state.type = st.radio('Summarization Method:',["map_reduce","refine"],index=0,)
        if st.session_state.type == "refine":
            refine_txt = st.text_area(
                "Define your refine prompt.",
                """Your job is to produce a final summary.
                We have provided an existing summary up to a certain point: {existing_answer}\n
                We have the opportunity to refine the existing summary (only if needed) with some more context below.
                ------------
                {text}
                ------------
                Given the new context, refine the original summary. If the context isn't useful, return the original summary.""")
            refine_prompt = PromptTemplate.from_template(refine_txt)
    else:
        st.session_state.type = 'stuff'
 

    if st.button("Process",key='runmodel'):
        with st.spinner("Processing"):
            if st.session_state.type == "stuff":
                summarized = src.summarizer(llm_model=st.session_state.llm_model, 
                                            temperature=st.session_state.temperature, 
                                            prompt=prompt,
                                            docs=src.txt_splt(txt_sum,st.session_state.split_size),
                                            type=st.session_state.type)
                txt_prompt = st.text_area("Here is your summarization and you can edit it. \n\n", summarized)
            elif st.session_state.type == "map_reduce":
                summarized = src.summarizer(llm_model=st.session_state.llm_model, 
                                            temperature=st.session_state.temperature, 
                                            prompt=prompt,
                                            docs=src.txt_splt(txt_sum,st.session_state.split_size),
                                            type=st.session_state.type)
                txt_prompt = st.text_area("Here is your summarization and you can edit it. \n\n", summarized)
            else:
                summarized = src.summarizer(llm_model=st.session_state.llm_model,
                                            temperature=st.session_state.temperature, 
                                            prompt=prompt,
                                            docs=src.txt_splt(txt_sum,st.session_state.split_size),
                                            type=st.session_state.type,refine_prompt=refine_prompt)
                txt_prompt = st.text_area("Here is your summarization and you can edit it. \n\n", summarized)

if __name__ == '__main__':
    main()
    