import streamlit as st
from ultralytics import YOLO
import cv2
import pandas as pd

def load_model(model_path):
    """
    Loads a YOLO model from the specified path.

    Parameters:
    - model_path (str): Path to the YOLO model.

    Returns:
    - model (YOLO): Loaded YOLO model object, or None if loading fails.

    Example:
    >>> model_path = "path/to/yolo/model"
    >>> yolo_model = load_model(model_path)
    >>> if yolo_model:
    >>>     print("Model loaded successfully.")
    >>> else:
    >>>     print("Failed to load the model.")
    """
    try:
        model = YOLO(model_path)
        return model
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)
        return None
    
def image_detect(model,uploaded_image,confidence_img):
    """
    Performs object detection on the uploaded image using the specified YOLO model.

    Parameters:
    - model (YOLO): Loaded YOLO model object.
    - uploaded_image (PIL.Image.Image or str): Uploaded image for object detection.
    - confidence_img (float): Confidence threshold for object detection.

    Returns:
    - res (list): List of detection results.
    - res_plotted (numpy.ndarray): Image with detection results plotted.

    Example:
    >>> yolo_model = load_model("path/to/yolo/model")
    >>> uploaded_image = Image.open("uploaded_image.jpg")
    >>> confidence_img = 0.5
    >>> detection_results, plotted_image = image_detect(yolo_model, uploaded_image, confidence_img)
    >>> print(detection_results)
    [{'label': 'person', 'confidence': 0.85, 'box': [x1, y1, x2, y2]}, {'label': 'car', 'confidence': 0.75, 'box': [x1, y1, x2, y2]}, ...]
    >>> plt.imshow(plotted_image)
    >>> plt.show()
    """
    res = model.predict(uploaded_image, 
                        conf=confidence_img)
    res_plotted = res[0].plot()[:, :, ::-1]
    return res, res_plotted

def results_img_df(res):
    """
    Converts detection results to a DataFrame and calculates class counts.

    Parameters:
    - res (list): List of detection results.

    Returns:
    - df (pandas.DataFrame): DataFrame containing detection results.
    - class_counts_df (pandas.DataFrame): DataFrame containing class counts.

    Example:
    >>> detection_results, _ = image_detect(yolo_model, uploaded_image, confidence_img)
    >>> df, class_counts_df = results_img_df(detection_results)
    >>> print(df.head())
       class  confidence             xyxy
    0  person        0.85  [x1, y1, x2, y2]
    1     car        0.75  [x1, y1, x2, y2]
    2  person        0.65  [x1, y1, x2, y2]
    3     dog        0.60  [x1, y1, x2, y2]
    4  person        0.55  [x1, y1, x2, y2]
    >>> print(class_counts_df.head())
       class  count
    0  person     12
    1     car      5
    """
    xyxy = []
    conf = []
    cls = []
    
    for box in res[0].boxes:
        xyxy.extend(box.xyxy.tolist())
        conf.extend(box.conf.tolist())
        cls.extend(box.cls.tolist())
    
    df = pd.DataFrame({
        'class': cls,
        'confidence': conf,
        'xyxy': xyxy 
        })
    df['class'] = df['class'].map(res[0].names)
    
    class_counts = df['class'].value_counts()
    class_counts_df = class_counts.reset_index()
    class_counts_df.columns = ['class', 'count']
    
    return df, class_counts_df

def run_model(conf, model, image, disp_tracker=None, tracker=None):
    """
    Runs object detection or tracking on the input image using the specified model.

    Parameters:
    - conf (float): Confidence threshold for object detection or tracking.
    - model (YOLO): Loaded YOLO model object.
    - image (numpy.ndarray): Input image to perform detection or tracking on.
    - disp_tracker (str or None): Option to display object tracking results. Default is None.
    - tracker (str or None): Tracker algorithm for object tracking. Default is None.

    Returns:
    - res (list): List of detection or tracking results.

    Example:
    >>> confidence_threshold = 0.5
    >>> yolo_model = load_model("path/to/yolo/model")
    >>> input_image = cv2.imread("input_image.jpg")
    >>> detection_results = run_model(confidence_threshold, yolo_model, input_image)
    >>> print(detection_results)
    [{'label': 'person', 'confidence': 0.85, 'box': [x1, y1, x2, y2]}, {'label': 'car', 'confidence': 0.75, 'box': [x1, y1, x2, y2]}, ...]
    """
    image = cv2.resize(image, (720, int(720*(9/16))))
    if disp_tracker == 'Yes': 
        res = model.track(image, 
                            conf=conf, 
                            persist=True, 
                            tracker=tracker)
    else:
        res = model.predict(image, conf = conf)
    return res             

def display_frames(res,st_frame): 
    """
    Displays the frames with detected objects.

    Parameters:
    - res (list): List of detection results.
    - st_frame (Streamlit.image): Streamlit image element for displaying the frames.

    Returns:
    - None

    Example:
    >>> detection_results, _ = image_detect(yolo_model, uploaded_image, confidence_img)
    >>> display_frames(detection_results, st_frame)
    """
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                    caption='Detected Video',
                    channels="BGR",
                    use_column_width=True
                    )

def count_objects(res):
    """
    Counts the objects detected in the detection results.

    Parameters:
    - res (list): List of detection results.

    Returns:
    - df (pandas.DataFrame): DataFrame containing object counts.

    Example:
    >>> detection_results, _ = image_detect(yolo_model, uploaded_image, confidence_img)
    >>> object_counts_df = count_objects(detection_results)
    >>> print(object_counts_df.head())
       id  class
    0   0  person
    1   1    car
    2   2  person
    3   3    dog
    4   4  person
    """
    cls = []
    id = []
    for box in res[0].boxes:
        cls.extend(box.cls.tolist())
        id.extend(box.id.tolist())
    df = pd.DataFrame({
        'id': id,
        'class': cls
        })
    df['class'] = df['class'].map(res[0].names)
    return df
    
def play_video(conf, model, file, disp_tracker=None, tracker=None):
    """
    Plays a video file and performs object detection or tracking on each frame.

    Parameters:
    - conf (float): Confidence threshold for object detection or tracking.
    - model (YOLO): Loaded YOLO model object.
    - file (str): Path to the video file.
    - disp_tracker (str or None): Option to display object tracking results. Default is None.
    - tracker (str or None): Tracker algorithm for object tracking. Default is None.

    Returns:
    - None

    Example:
    >>> confidence_threshold = 0.5
    >>> yolo_model = load_model("path/to/yolo/model")
    >>> video_file = "path/to/video/file.mp4"
    >>> play_video(confidence_threshold, yolo_model, video_file, disp_tracker="Yes", tracker="kcf")
    """
    try:
        vid_cap = cv2.VideoCapture(file)
        st_frame = st.empty()
        st_table = st.empty()
        final_df = pd.DataFrame(columns=['id', 'class'])
        while (vid_cap.isOpened()):
            success, image = vid_cap.read()
            if success:
                if disp_tracker == "Yes":
                    res = run_model(conf = conf,
                                    model = model,
                                    image = image,
                                    disp_tracker = disp_tracker,
                                    tracker = tracker
                                    )
                    display_frames(res,st_frame)
                    df = count_objects(res)
                    final_df = pd.concat([final_df, df], ignore_index=True)
                    unique_counts_df = final_df.groupby('class')['id'].nunique().reset_index(name='count').sort_values(by='count', ascending=False)
                    st_table.dataframe(data=unique_counts_df)
                else:
                    res = run_model(conf = conf,
                                    model = model,
                                    image = image
                                    )
                    display_frames(res,st_frame)
            else:
                vid_cap.release()
                break
    except Exception as e:
        st.error("Error loading video: " + str(e))