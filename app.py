import streamlit as st

import cv2
from ultralytics import YOLO
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

import numpy as np
import pandas as pd
import time

st.title("SmartVision AI - Intelligent Multi-Class Object Recognition System")
labels = {0: 'airplane',
            1: 'bed',
            2: 'bench',
            3: 'bicycle',
            4: 'bird',
            5: 'bottle',
            6: 'bowl',
            7: 'bus',
            8: 'cake',
            9: 'car',
            10: 'cat',
            11: 'chair',
            12: 'couch',
            13: 'cow',
            14: 'cup',
            15: 'dog',
            16: 'elephant',
            17: 'horse',
            18: 'motorcycle',
            19: 'person',
            20: 'pizza',
            21: 'potted plant',
            22: 'stop sign',
            23: 'traffic light',
            24: 'train',
            25: 'truck'}

@st.cache_resource
def getModel(modelName):
    if modelName == "YOLO":
        model = YOLO('YOLO\best.pt')
    else:
        model = tf.keras.models.load_model(f'{modelName}\\best_model.keras')
    
    return model

def predict(modelName, img_array):
    model = getModel(modelName)

    # 1. Warm up (ignore this one)
    model.predict(img_array)

    # 2. Start the timer
    start_time = time.time()

    # 3. Make the actual prediction
    predictions = model.predict(img_array)

    # 4. End the timer
    end_time = time.time()

    # Calculate duration in milliseconds
    duration = (end_time - start_time) * 1000
    return predictions, duration

with st.sidebar:
    
    st.title("Navigation")
    page = st.radio(
        "Go to",
        [
            "Home",
            "Webcam detection"
        ],
        index=0,
        width='stretch'
    )
if page == "Home":
    st.subheader('Object detection and recognition')
    uploaded_file = st.file_uploader("**Select Image**", width=200)
    if st.button("Detect and Classify"):
        yolo = getModel('YOLO')

        #Detection
        results = yolo.predict(
            source=Image.open(uploaded_file), 
            save=True,
            device='cpu',
            imgsz=640
        )

        for r in results:
            im_array = r.plot() 
                
            # Convert BGR to RGB for Streamlit display
            im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
                
        # Show the final image with detections
        st.image(im_rgb, caption="Processed Image")

        vgg = getModel('EfficientNetB0')
        # 1. Open and resize the image to 224x224 (VGG standard)
        img = Image.open(uploaded_file).resize((224, 224))

        # 2. Convert PIL image to a Numpy array (Shape: 224, 224, 3)
        img_array = img_to_array(img)

        # 3. Add batch dimension (Shape: 1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)

        # 4. Predict
        pVGG16, dVGG16 = predict('VGG16', keras.applications.vgg16.preprocess_input(img_array))
        cVGG16 = labels[np.argmax(pVGG16, axis=1)[0]]

        pResnet50, dResnet50 = predict('Resnet50', keras.applications.resnet50.preprocess_input(img_array))
        cResnet50 = labels[np.argmax(pResnet50, axis=1)[0]]

        pMobileNet, dMobileNet = predict('MobileNet', keras.applications.mobilenet_v2.preprocess_input(img_array))
        cMobileNet = labels[np.argmax(pMobileNet, axis=1)[0]]

        pEfficientNetB0, dEfficientNetB0 = predict('EfficientNetB0', keras.applications.efficientnet.preprocess_input(img_array))
        cEfficientNetB0 = labels[np.argmax(pEfficientNetB0, axis=1)[0]]
       
        data = pd.DataFrame({
            'Model': ['VGG16', 'Resnet50', 'MobileNet','EfficientNetB0'],
            'Duration': [dVGG16, dResnet50, dMobileNet, dEfficientNetB0],
            'Class': [cVGG16, cResnet50, cMobileNet, cEfficientNetB0]
        })

        st.dataframe(data, hide_index=True)
else:

    # --- PAGE CONFIG ---
    st.set_page_config(page_title="YOLOv8 Real-Time Detection", layout="wide")
    st.subheader('Webcam Object Detection')

    # --- 1. LOAD MODEL (Cached) ---
    model = getModel('YOLO')

    # --- 2. SIDEBAR CONTROLS ---
    st.sidebar.header("Settings")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    iou_threshold = st.sidebar.slider("IOU Threshold (NMS)", 0.0, 1.0, 0.45, 0.05)

    # --- 3. WEBRTC CALLBACK ---
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")

        # Run YOLOv8 Inference
        # we pass the slider values into the model call
        results = model.predict(
            source=img, 
            conf=conf_threshold, 
            iou=iou_threshold,
            imgsz=320 # You can lower this to 320 for even more speed on CPU
        )

        # Use YOLO's built-in plotting to draw boxes and labels
        annotated_frame = results[0].plot()

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    # --- 4. STREAMER CONFIG ---
    # STUN servers help bypass firewalls for the webcam stream
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    webrtc_streamer(
        key="yolo-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True, # Critical for maintaining high FPS
    )
    
    # --- 5. CLASS LIST ---
    if st.sidebar.checkbox("Show Class List"):
        st.sidebar.write(model.names)