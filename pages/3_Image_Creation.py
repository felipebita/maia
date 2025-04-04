import streamlit as st
from dotenv import load_dotenv
import src.src3 as src


def main():
    load_dotenv()
    st.set_page_config(page_title="Image Creation", page_icon=":frame_with_picture:")

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

    st.header("Image Creation :frame_with_picture:")
    
    with st.expander("Model Options"):
        st.session_state.i_model = st.radio('Image model:',["dall-e-2","dall-e-3"],index=0)
        if st.session_state.i_model == "dall-e-2":
            st.session_state.i_size = st.selectbox('Size of the image:', ("256x256", "512x512", "1024x1024"))
        else:
            st.session_state.i_size = st.selectbox('Size of the image:', ("1024x1024", "1792x1024","1024x1792")) 
            st.session_state.i_qual = st.radio('Image quality:',["standard","hd",],index=0) 
            st.session_state.i_style = st.radio('Image style:',["vivid","natural",],index=0)
        st.session_state.enh_bot = st.toggle('Activate Enhanced Prompt')
        if st.session_state.enh_bot:
            st.session_state.pen_model = st.radio('Model:',["gpt-3.5-turbo","gpt-4"],index=0)
            st.session_state.pen_temp = st.slider('Temperature:', 0.0, 1.0, 0.1)

    txt_prompt = st.text_area("Describe your image.")

    if st.session_state.enh_bot:
        if st.button("Enhance",key='enhancer_2'):
            with st.spinner("Processing"):
                st.session_state.p_enhanced = src.prompt_enhancer(st.session_state.pen_model,st.session_state.pen_temp,txt_prompt)
    else:
        if st.button("Generate Image",key='runmodel'):
                with st.spinner("Processing"):
                    if st.session_state.i_model == "dall-e-2":
                        st.image(src.get_images_2(model=st.session_state.i_model,prompt=txt_prompt,size=st.session_state.i_size))
                    else:
                        st.image(src.get_images_3(model=st.session_state.i_model,prompt=txt_prompt,size=st.session_state.i_size,quality=st.session_state.i_qual,style=st.session_state.i_style))

    if "p_enhanced" in st.session_state:
        p_enhanced_box = st.text_area("Enhanced Prompt",st.session_state.p_enhanced)
        st.write(f"""Your prompt has '{len(p_enhanced_box)}' characters. The limit is 1000.""")
 
        if st.button("Generate Image (Enhanced)",key='runmodel_enh'):
            with st.spinner("Processing"):
                if st.session_state.i_model == "dall-e-2":
                    st.image(src.get_images_2(model=st.session_state.i_model,prompt=p_enhanced_box,size=st.session_state.i_size))
                else:
                    st.image(src.get_images_3(model=st.session_state.i_model,prompt=p_enhanced_box,size=st.session_state.i_size,quality=st.session_state.i_qual,style=st.session_state.i_style))
    
if __name__ == '__main__':
    main()
    