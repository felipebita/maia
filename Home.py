import streamlit as st
from st_pages import Page, show_pages

def main():
    st.set_page_config(page_title="MAIA", page_icon="🏠")
    left_co, cent_co,last_co = st.columns([1,5,1])
    with cent_co:
        st.image("img/logo_wd.png")
        hide_img_fs = '''
            <style>
            button[title="View fullscreen"]{
                visibility: hidden;}
            </style>
            '''

        st.markdown(hide_img_fs, unsafe_allow_html=True)
    
    st.write(
    """
    Introducing MAIA, your all-in-one solution for seamless interaction with information and harnessing the power of artificial intelligence! 
    Our app offers a unique set of features designed to enhance productivity and streamline tasks.

    **1. Chat with Documents:** 
        Say goodbye to the hassle of searching through lengthy PDF documents! With our Document Chat feature, effortlessly ask questions and receive instant, relevant answers. Whether you're a student, researcher, or professional, this tool transforms how you interact with information. Simply type your query, and let our AI navigate through documents to provide precise answers.

    **2. Summarization:** 
        Save time and enhance understanding with our Summarization feature. Our AI-powered tool analyzes text, distilling it down to key points and delivering concise summaries. Ideal for research, studying, or quick knowledge absorption, this feature empowers you to focus on essential information.

    **3. Image Creation:** 
        Unleash creativity with our Image Creation feature. Transform ideas into visual representations effortlessly! Whether designing presentations, social media posts, or enhancing content, our AI-driven tool has you covered.

    **4. Sentinel:** 
        Sentinel employs artificial intelligence to identify and count objects in images and videos. Whether analyzing surveillance footage, conducting inventory checks, or monitoring wildlife, Sentinel provides accurate object detection and counting capabilities. Efficiently analyze visual data and extract valuable insights with Sentinel.
    """
    )

    with st.sidebar: 
        show_pages(
        [
            Page("Home.py", "Home", "🏠"),
            Page("pages/1_Chat_with_Documents.py", "Chat with Documents", ":books:"),
            Page("pages/2_Text_Summarization.py", "Text Summarization", ":receipt:"),
            Page("pages/3_Image_Creation.py", "Image Creation", ":frame_with_picture:"),
            Page("pages/4_Sentinel.py", "Sentinel", ":video_camera:")
        ])
        st.image("img/logo_sq.png")
        hide_img_fs = '''
            <style>
            button[title="View fullscreen"]{
                visibility: hidden;}
            </style>
            '''

        st.markdown(hide_img_fs, unsafe_allow_html=True)
        st.markdown("This is a portfolio project by Felipe Martins. If you want to see the code of this app and other data science projects check my [GitHub](https://github.com/felipebita).")
        st.markdown("This is just an example tool. Please, do not abuse on my OpenAI credits, use it only for testing purposes.")

if __name__ == '__main__':
    main()