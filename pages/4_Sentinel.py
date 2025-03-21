import streamlit as st
import PIL
import tempfile
from ultralytics import YOLO
import src.src4 as src

model_path = 'weights/yolov8n.pt'

def main():
    st.set_page_config(page_title="Sentinel", page_icon=":video_camera:")
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

    st.header("Sentinel :video_camera:")
    
    with st.expander("Identifiable objects"):
        model = src.load_model(model_path)
        st.write(model.names)

    tab1, tab2 = st.tabs(["Image","Video"])
    with tab1:
        source_img = st.file_uploader("Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
        confidence_img = float(st.slider("Select Model Confidence", 25, 100, 40, key="img_slide")) / 100

        col1, col2 = st.columns(2)    
        with col1:
            if source_img:
                uploaded_image = PIL.Image.open(source_img)
                st.image(uploaded_image,
                        caption="Uploaded Image",
                        use_column_width=True
                        )
            
        if st.button('Detect Objects', key="img_but"):
            model = src.load_model(model_path)
            res, res_plotted = src.image_detect(model,
                                              uploaded_image,
                                              confidence_img
                                              )
            df, class_counts_df = src.results_img_df(res)                   
            with col1:
                st.image(res_plotted,
                        caption='Analyzed Image',
                        use_column_width=True
                        )
            with col2:
                st.dataframe(data=df)
                st.dataframe(data=class_counts_df)

    with tab2:    
        disp_tracker = st.radio("Display Tracker", ('Yes', 'No'))
        if disp_tracker == "Yes":
            tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))

        
        source_vid = st.file_uploader("Choose an video...", type=("mp4"))      
        example_videos = st.checkbox("Try with example video.")

        confidence_vid = float(st.slider("Select Model Confidence", 25, 100, 40, key="vid_slide")) / 100

        if example_videos:
            example = st.selectbox("Select the example video.",("People Walking", "Car Traffic"))
            if example == "People Walking":
                vid_path = 'midia/people_walking.mp4'
            else:
                vid_path = 'midia/car_traffic.mp4'
            st.video(vid_path)

        if source_vid :
            tfile= tempfile.NamedTemporaryFile(delete=False)
            tfile.write(source_vid.read())
            vid_path = tfile.name    
            st.video(vid_path)     

        if st.button('Detect Objects', key="vid_but"):
            model = src.load_model(model_path)
            if disp_tracker == "Yes":
                src.play_video(conf = confidence_vid, 
                                model = model, 
                                file = vid_path,
                                disp_tracker = disp_tracker,
                                tracker = tracker_type)
            else:
                src.play_video(conf = confidence_vid, 
                                model = model, 
                                file = tfile.name,
                                disp_tracker = disp_tracker)


if __name__ == '__main__':
    main()
    