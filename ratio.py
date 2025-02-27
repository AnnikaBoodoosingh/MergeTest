import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Streamlit File Upload feature
uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read the uploaded image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Convert from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (300, 300))

    # Convert to HSV color space
    hsv = cv2.cvtColor(image_resized, cv2.COLOR_RGB2HSV)

    # --- SHADOW REMOVAL ---
    lower_shadow = np.array([0, 0, 0])  # Lower bound for shadow detection
    upper_shadow = np.array([180, 255, 100])  # Upper bound for shadow detection
    shadow_mask = cv2.inRange(hsv, lower_shadow, upper_shadow)
    image_no_shadow = cv2.bitwise_and(image_resized, image_resized, mask=~shadow_mask)

    # --- MASK CREATION ---
    lower_yellow = np.array([15, 100, 50])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    lower_brown = np.array([2, 3, 10])
    upper_brown = np.array([38, 255, 200])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)

    diseased_mask = cv2.bitwise_or(yellow_mask, brown_mask)
    kernel = np.ones((3, 3), np.uint8)
    diseased_mask_expanded = cv2.dilate(diseased_mask, kernel, iterations=1)

    lower_green = np.array([40, 100, 50])
    upper_green = np.array([70, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_near_disease = cv2.bitwise_and(green_mask, diseased_mask_expanded)

    diseased_mask = cv2.bitwise_or(diseased_mask, green_near_disease)
    diseased_only = np.zeros_like(image_resized)
    diseased_only[diseased_mask != 0] = image_resized[diseased_mask != 0]

    diseased_pixels = np.count_nonzero(np.all(diseased_only != 0, axis=-1))

    leaf_mask = np.all(image_resized != [0, 0, 0], axis=-1)
    total_leaf_pixels = np.count_nonzero(leaf_mask)

    severity_ratio = (diseased_pixels / total_leaf_pixels) * 100 if total_leaf_pixels > 0 else 0

    # Display Results
    st.image(uploaded_file, caption="Original Image", use_column_width=True)
    st.image(diseased_only, caption="Diseased Area", use_column_width=True)

    st.write(f"Diseased Pixels: {diseased_pixels}")
    st.write(f"Total Leaf Pixels: {total_leaf_pixels}")
    st.write(f"Severity Ratio: {severity_ratio}%")
