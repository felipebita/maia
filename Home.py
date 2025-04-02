import streamlit as st

def main():
    st.set_page_config(page_title="MAIA", page_icon="🏠")
    # Hide default menu
    
    st.markdown("""
        <style>
            #MainMenu {visibility: hidden;}
            div[data-testid="stSidebarNav"] {display: none;}
        </style>
        """, unsafe_allow_html=True)

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
    
    **5. Agentic Workflows:** 
        Streamline complex processes with intelligent, sequential decision-making. Automate text analysis, essay grading, and SQL-based data exploration with adaptive AI workflows designed for precision and efficiency.
    """
    )


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

if __name__ == '__main__':
    main()


