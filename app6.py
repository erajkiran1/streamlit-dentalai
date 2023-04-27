import streamlit as st
import torch
from PIL import Image
from io import *
import glob
from datetime import datetime
import os
#import wget
import time
import cv2
from ultralytics import YOLO

def load_and_predict():
    image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])
    col1, col2 = st.columns(2)
    if image_file is not None:
        img = Image.open(image_file)
        with col1:
            st.image(img, caption='Uploaded Image', use_column_width='always')
            ts = datetime.timestamp(datetime.now())
            imgpath = os.path.join('uploads/', str(ts)+image_file.name)
            outputpath = os.path.join('outputs/', os.path.basename(imgpath))
            with open(imgpath, mode="wb") as f:
                f.write(image_file.getbuffer())
        
        
#-------------------Load IOP Model----------------
            #st.write("Please wait, model is Loading...!")
            model=YOLO("models/best.pt")

#----------------Predict-------------------------
            st.write("Please wait while computer vision model is working for you...!")
            result=model.predict(imgpath)
        
#------------Plot Results-----------------------
            with col2:
                mask_plotted=result[0].plot()
                st.image(mask_plotted,caption="Detected Image",use_column_width=True)
                cv2.imwrite(outputpath,mask_plotted)

def main():
    st.title(" ")
    st.title("Ivory-IOP instance Segmentations")
    load_and_predict()

if __name__=='__main__':
    main()