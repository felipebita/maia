import streamlit as st
from dotenv import load_dotenv
import src.src1 as src

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with Documents", page_icon=":books:")
    st.markdown("""
        <style>
            #MainMenu {visibility: hidden;}
            div[data-testid="stSidebarNav"] {display: none;}
        </style>
        """, unsafe_allow_html=True)
    
    st.sidebar.image("img/logo_sq.png")

    # Create the sidebar navigation
    pages = {
    "🏠 Home": "Home.py",
    "📚 Chat with Documents": "pages/1_Chat_with_Documents.py",
    "📝 Text Summarization": "pages/2_Text_Summarization.py",
    "🖼️ Image Creation": "pages/3_Image_Creation.py",
    "📹 Sentinel": "pages/4_Sentinel.py",
    "🔄 Agentic Workflows": "pages/5_Agentic_Workflows.py"
    }

    # Add some styling to make the sidebar links more visible
    st.markdown(
    """
    <style>
    .css-1d391kg {
        padding-top: 0rem;
    }
    .css-1d391kg a {
        text-decoration: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )

    # Create links for each page
    for page_name, page_path in pages.items():
        st.sidebar.page_link(page_path, label=page_name)

    # Add the messages in the sidebar
    st.sidebar.markdown("---")  # This adds a horizontal line for separation
    st.sidebar.markdown("This is a portfolio project by Felipe Martins. If you want to see the code of this app and other data science projects check my [GitHub](https://github.com/felipebita).")
    st.sidebar.markdown("This is just an example tool. Please, do not abuse on my OpenAI credits, use it only for testing purposes.")

    # Hide fullscreen button for image
    hide_img_fs = '''
    <style>
    button[title="View fullscreen"]{
        visibility: hidden;}
    </style>
    '''
    st.markdown(hide_img_fs, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with Documents :books:")
  
    with st.expander("Database Options"):
        pdf_docs = st.file_uploader("Upload your files (only PDF)",accept_multiple_files=True)
        chunk_size = st.number_input('Chunk Size:',min_value=200, max_value=2000,value=1000)
        overlap_size = st.number_input('Overlap:', min_value = 40, max_value=400, value=200)

        if st.button("Process",key='database'):
            with st.spinner("Processing"):
                # get the pdf text
                raw_text = src.get_pdf_text(pdf_docs)
                    
                # get the text chunks
                text_chunks = src.get_text_chunks(raw_text, chunk_size, overlap_size)

                # create the vector store
                st.session_state.vectorstore = src.get_vectorstore(text_chunks)

            st.success('Done! Proceed to Model Options')

    with st.expander("Model Options"):
        llm_model = st.radio('LLM model:',["gpt-3.5-turbo","gpt-4",],index=0,)
        temperature = st.slider('Temperature:', 0.0, 1.0, step=0.1, value=0.7)
        k = st.slider('Chunks to Retrieve:', 0, 10, step=1, value=4)

        if st.button("Process",key='chat'):
            with st.spinner("Processing"):
                # create converstion chain
                st.session_state.conversation = src.get_conversation_chain(st.session_state.vectorstore,llm_model,temperature,k)
            st.success('Now you can use the chat!')
  
    user_question = st.chat_input("Ask a question about your documents.")
    if user_question:
        # start conversation
        src.handle_userinput(user_question)     

if __name__ == '__main__':
    main()