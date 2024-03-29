import os
import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load the YOLO model
model = YOLO('/home/sysadm/Downloads/building/runs/detect/train/weights/best.pt')

# Decoding according to the .yaml file class names order
decoding_of_predictions = {0: 'undamagedcommercialbuilding', 
                           1: 'undamagedresidentialbuilding',
                           2: 'damagedresidentialbuilding',
                           3: 'damagedcommercialbuilding'}

# Define colors for different classes
class_colors = {'undamagedcommercialbuilding': (0, 1, 0),  # Green
                'undamagedresidentialbuilding': (1, 0, 0),  # Red
                'damagedresidentialbuilding': (0, 0, 1),  # Blue
                'damagedcommercialbuilding': (0.5, 0.5, 0.5)}  # Grey

def main():
    st.title("Object Detection with YOLOv8")
    st.markdown("<style> p{margin: 10px auto; text-align: justify; font-size:20px;}</style>", unsafe_allow_html=True)      
    st.markdown("<p>🚀Welcome to the introduction page of our project! In this project, we will be exploring the YOLO (You Only Look Once) algorithm. YOLO is known for its ability to detect objects in an image in a single pass, making it a highly efficient and accurate object detection algorithm.🎯</p>", unsafe_allow_html=True)  
    st.markdown("<p>The latest version of YOLO, YOLOv8, released in January 2023 by Ultralytics, has introduced several modifications that have further improved its performance. 🌟</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Detect Objects"):
            detect_objects(image)

def detect_objects(image):
    # Resize the image to 512x512
    image_resized = image.resize((512, 512))

    # Convert the PIL image to numpy array
    img_np = np.array(image_resized)

    # Perform object detection
    results = model.predict(img_np, save=True, iou=0.5, save_txt=True, conf=0.25)

    # Plot the image
    plt.figure(figsize=(8, 8))
    plt.imshow(image_resized)

    for r in results:
        conf_list = r.boxes.conf.numpy().tolist()
        clss_list = r.boxes.cls.numpy().tolist()
        original_list = clss_list
        updated_list = [decoding_of_predictions[int(element)] for element in original_list]

        bounding_boxes = r.boxes.xyxy.numpy()
        confidences = conf_list
        class_names = updated_list

        # Draw bounding boxes on the image
        for bbox, conf, cls in zip(bounding_boxes, confidences, class_names):
            x1, y1, x2, y2 = bbox.astype(int)
            box_color = class_colors[cls]  # Get color based on class

            # Draw bounding box
            plt.rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=box_color, facecolor='none')

            # Determine text color based on box color
            text_color = 'black' if sum(box_color) > 1.5 else 'white'

            # Draw class label
            plt.text(x1, y1 - 10, f'{cls} {conf:.2f}', color=text_color, fontsize=10, ha='left')

    plt.axis('off')
    st.pyplot(plt)

if __name__ == "__main__":
    main()

